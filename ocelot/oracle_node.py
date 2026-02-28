#!/usr/bin/env python3
"""Oracle node — privileged ground-truth pose tracker.

Reads face billboard world positions directly from Gazebo via ros_gz_bridge
(/model/face_N/pose).  Uses closed-form analytic FK to transform each face
world position into the camera frame, selects the tracking target based on the
episode label_key, computes pan/tilt angular errors, and publishes /cmd_vel.

No face detector is used — this is the "privileged teacher" whose behaviour
the VLA student will imitate.  It has the same /cmd_vel interface as
tracker_node and drives the same cmd_vel_adapter → joints pipeline.

Target selection per label_key (set via ros2 param at episode start):
    track / multi_attr → face_0 (the oracle-designated target)
    slow               → face_0 at half max_velocity
    multi_left         → whichever face has the most positive camera-frame Y
    multi_right        → whichever face has the most negative camera-frame Y
    multi_closest      → whichever face is closest to the camera

This makes oracle behaviour stateless at the frame level: the selected target
can change mid-episode as faces move, which is exactly what a stateless VLA
will need to learn.

Start disabled (default) and enable at runtime:
    ros2 param set /oracle_node enabled true

Or launch with use_oracle:=true (sim_launch.py) to start enabled directly.

FK chain (robot spawned at world origin, matches urdf/pan_tilt.urdf):
    base_link  @ world (0, 0, 0)
    pan_joint  : +0.03 m Z from base_link, revolute around Z, angle = θ_pan
    tilt_joint : +0.03 m Z from pan_link,  revolute around Y, angle = θ_tilt
    camera_link: +0.02 m X, +0.01 m Z from tilt_link, fixed (no rotation)

Sign convention (matches cmd_vel_adapter and tracker_node):
    angular.z  → pan_joint velocity  (positive = CCW = camera turns left)
    angular.y  → tilt_joint velocity (positive = nose-down)
    pan_err  > 0 → face is LEFT  of camera → turn left  (positive angular.z) ✓
    tilt_err > 0 → face is ABOVE camera    → tilt up    (negative angular.y) ✓
"""

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, Twist
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from sensor_msgs.msg import JointState

# URDF joint / link offsets (metres) — must match urdf/pan_tilt.urdf
_PAN_Z = 0.03   # base_link  → pan_joint origin, Z
_TILT_Z = 0.03  # pan_link   → tilt_joint origin, Z (in pan_link frame)
_CAM_X = 0.02   # tilt_link  → camera_link, X
_CAM_Z = 0.01   # tilt_link  → camera_link, Z

# Maximum number of face_N topics the oracle subscribes to.
_MAX_FACES = 3


def _rz(theta: float) -> np.ndarray:
    """Rotation matrix around Z axis (pan joint)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def _ry(theta: float) -> np.ndarray:
    """Rotation matrix around Y axis (tilt joint)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]])


class OracleNode(Node):
    def __init__(self) -> None:
        super().__init__('oracle_node')

        self.declare_parameter('enabled', False)
        self.declare_parameter('kp_pan', 5.0)        # rad/s per radian of pan error
        self.declare_parameter('kp_tilt', 2.5)       # rad/s per radian of tilt error
        self.declare_parameter('max_velocity', 0.3)  # rad/s clamp
        self.declare_parameter('deadband_rad', 0.002) # ~0.11° — suppress sub-pixel chatter
        self.declare_parameter('label_key', 'track')  # controls target selection
        self.declare_parameter('num_faces', 1)        # faces active in current episode

        self._pan_angle: float = 0.0
        self._tilt_angle: float = 0.0
        # Pose buffer for each face slot.  Populated by /model/face_N/pose callbacks.
        # Reset to None at startup; episode_runner sets num_faces so only the
        # face slots in [0, num_faces) are considered each control tick.
        self._face_positions: dict[str, np.ndarray | None] = {
            f'face_{i}': None for i in range(_MAX_FACES)
        }
        self._tick: int = 0  # diagnostic counter

        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)
        for i in range(_MAX_FACES):
            name = f'face_{i}'
            self.create_subscription(
                Pose,
                f'/model/{name}/pose',
                lambda msg, n=name: self._face_pose_cb(n, msg),
                10,
            )
        self._pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Control loop at 20 Hz (decoupled from sensor callbacks).
        # Face poses arrive at ~10 Hz (PosePublisher update_rate).
        # Joint states arrive at ~50 Hz from ros2_control.
        self.create_timer(0.05, self._control_loop)

        # Reset face pose buffers whenever num_faces changes so stale poses
        # from the previous episode don't contaminate target selection.
        self.add_on_set_parameters_callback(self._on_params_changed)

        self.get_logger().info('Oracle node ready — set enabled:=true to activate')

    # ── parameter change callback ───────────────────────────────────────────

    def _on_params_changed(self, params) -> SetParametersResult:
        for p in params:
            if p.name == 'num_faces':
                # New episode starting — reset all face pose buffers so stale
                # poses from the previous episode don't pollute target selection.
                for key in self._face_positions:
                    self._face_positions[key] = None
                self.get_logger().debug(
                    f'num_faces → {p.value}: face pose buffers reset'
                )
        return SetParametersResult(successful=True)

    # ── callbacks ──────────────────────────────────────────────────────────

    def _joint_cb(self, msg: JointState) -> None:
        for i, name in enumerate(msg.name):
            if name == 'pan_joint':
                self._pan_angle = msg.position[i]
            elif name == 'tilt_joint':
                self._tilt_angle = msg.position[i]

    def _face_pose_cb(self, name: str, msg: Pose) -> None:
        self._face_positions[name] = np.array([
            msg.position.x,
            msg.position.y,
            msg.position.z,
        ])

    # ── control loop ───────────────────────────────────────────────────────

    def _control_loop(self) -> None:
        if not self.get_parameter('enabled').value:
            return

        label_key = self.get_parameter('label_key').value
        num_faces = int(self.get_parameter('num_faces').value)

        # Collect positions for faces active in this episode.
        available: dict[str, np.ndarray] = {}
        for i in range(num_faces):
            n = f'face_{i}'
            if self._face_positions[n] is not None:
                available[n] = self._face_positions[n]

        if not available:
            return

        # Rotation matrices for current joint angles.
        R_pan = _rz(self._pan_angle)
        R_tilt = _ry(self._tilt_angle)
        # Camera frame orientation in world frame: apply pan first, then tilt.
        R_cam = R_pan @ R_tilt

        # Tilt joint centre in world frame.
        P_tilt = np.array([0.0, 0.0, _PAN_Z + _TILT_Z])

        # Camera origin in world frame.
        P_cam = P_tilt + R_cam @ np.array([_CAM_X, 0.0, _CAM_Z])

        # Camera-frame displacement vector for every available face.
        d_cams: dict[str, np.ndarray] = {
            n: R_cam.T @ (pos - P_cam) for n, pos in available.items()
        }

        # Select the tracking target based on the episode label_key.
        # Positional selections are computed in camera frame so the definition
        # of "left/right/closest" matches what the VLA sees in the image.
        if label_key == 'multi_left':
            # Leftmost = most positive camera-frame Y (camera +Y is to the left).
            target = max(d_cams, key=lambda n: d_cams[n][1])
        elif label_key == 'multi_right':
            # Rightmost = most negative camera-frame Y.
            target = min(d_cams, key=lambda n: d_cams[n][1])
        else:
            # track, multi_attr, slow, unknown: always face_0.
            target = 'face_0'
            if target not in d_cams:
                return

        d_cam = d_cams[target]

        if d_cam[0] <= 0.0:
            # Face is at or behind the camera plane — stop and wait.
            self._pub.publish(Twist())
            return

        # Angular errors (radians).
        #   pan_err  > 0 → face is in +Y direction (left)  → rotate camera left
        #   tilt_err > 0 → face is in +Z direction (above) → tilt camera up
        pan_err = np.arctan2(d_cam[1], d_cam[0])
        tilt_err = np.arctan2(d_cam[2], d_cam[0])

        kp_pan = self.get_parameter('kp_pan').value
        kp_tilt = self.get_parameter('kp_tilt').value
        max_vel = self.get_parameter('max_velocity').value
        if label_key == 'slow':
            max_vel *= 0.5
        deadband = self.get_parameter('deadband_rad').value

        twist = Twist()
        if abs(pan_err) > deadband:
            # Positive pan_err → face left → positive angular.z → camera turns left ✓
            twist.angular.z = float(np.clip(kp_pan * pan_err, -max_vel, max_vel))
        if abs(tilt_err) > deadband:
            # Positive tilt_err → face above → negative angular.y → tilt joint nose-up ✓
            twist.angular.y = float(np.clip(-kp_tilt * tilt_err, -max_vel, max_vel))

        self._pub.publish(twist)
        self._tick += 1
        # Log at INFO for first 500 ticks (~25s) for diagnostic visibility, then DEBUG
        log_fn = self.get_logger().info if self._tick <= 500 else self.get_logger().debug
        face_pos = available[target]
        log_fn(
            f'[t{self._tick}] target={target} '
            f'joints=({self._pan_angle:.3f},{self._tilt_angle:.3f}) '
            f'face=({face_pos[0]:.2f},{face_pos[1]:.2f},{face_pos[2]:.2f}) '
            f'd_cam=({d_cam[0]:.2f},{d_cam[1]:.2f},{d_cam[2]:.2f}) '
            f'err=({np.degrees(pan_err):.1f}°,{np.degrees(tilt_err):.1f}°) '
            f'vel=({twist.angular.z:+.2f},{twist.angular.y:+.2f})'
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OracleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
