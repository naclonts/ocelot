#!/usr/bin/env python3
"""Oracle node — privileged ground-truth pose tracker.

Reads the face billboard world position directly from Gazebo via
ros_gz_bridge (/model/face_0/pose).  Uses closed-form analytic FK to
transform the face world position into the camera frame, computes pan/tilt
angular errors, and publishes /cmd_vel.

No face detector is used — this is the "privileged teacher" whose behaviour
the VLA student will imitate.  It has the same /cmd_vel interface as
tracker_node and drives the same cmd_vel_adapter → joints pipeline.

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
from rclpy.node import Node
from sensor_msgs.msg import JointState

# URDF joint / link offsets (metres) — must match urdf/pan_tilt.urdf
_PAN_Z = 0.03   # base_link  → pan_joint origin, Z
_TILT_Z = 0.03  # pan_link   → tilt_joint origin, Z (in pan_link frame)
_CAM_X = 0.02   # tilt_link  → camera_link, X
_CAM_Z = 0.01   # tilt_link  → camera_link, Z


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
        self.declare_parameter('max_velocity', 0.3)  # rad/s clamp — moderate speed for training
        self.declare_parameter('deadband_rad', 0.002) # ~0.11° — suppress sub-pixel chatter
        self.declare_parameter('label_key', 'track')  # 'slow' → halve max_velocity

        self._pan_angle: float = 0.0
        self._tilt_angle: float = 0.0
        self._face_pos: np.ndarray | None = None
        self._tick: int = 0  # diagnostic counter

        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)
        self.create_subscription(Pose, '/model/face_0/pose', self._face_pose_cb, 10)
        self._pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Control loop at 20 Hz (decoupled from sensor callbacks).
        # Face pose arrives at ~10 Hz (PosePublisher update_rate in model.sdf).
        # Joint states arrive at ~50 Hz from ros2_control.
        self.create_timer(0.05, self._control_loop)

        self.get_logger().info('Oracle node ready — set enabled:=true to activate')

    # ── callbacks ──────────────────────────────────────────────────────────

    def _joint_cb(self, msg: JointState) -> None:
        for i, name in enumerate(msg.name):
            if name == 'pan_joint':
                self._pan_angle = msg.position[i]
            elif name == 'tilt_joint':
                self._tilt_angle = msg.position[i]

    def _face_pose_cb(self, msg: Pose) -> None:
        self._face_pos = np.array([
            msg.position.x,
            msg.position.y,
            msg.position.z,
        ])

    # ── control loop ───────────────────────────────────────────────────────

    def _control_loop(self) -> None:
        if not self.get_parameter('enabled').value or self._face_pos is None:
            return

        # Rotation matrices for current joint angles.
        R_pan = _rz(self._pan_angle)
        R_tilt = _ry(self._tilt_angle)
        # Camera frame orientation in world frame: apply pan first, then tilt.
        R_cam = R_pan @ R_tilt

        # Tilt joint centre in world frame.
        # Rz leaves the Z component unchanged, so both Z offsets stack directly
        # regardless of current pan angle.
        P_tilt = np.array([0.0, 0.0, _PAN_Z + _TILT_Z])

        # Camera origin in world frame.
        P_cam = P_tilt + R_cam @ np.array([_CAM_X, 0.0, _CAM_Z])

        # Face direction vector expressed in camera frame.
        # Camera looks along +X in camera_link (camera_joint has rpy=0,0,0).
        d_cam = R_cam.T @ (self._face_pos - P_cam)

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
        if self.get_parameter('label_key').value == 'slow':
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
        # Log at INFO for first 500 ticks (~25s) for diagnostic visibility, then downgrade to DEBUG
        log_fn = self.get_logger().info if self._tick <= 500 else self.get_logger().debug
        log_fn(
            f'[t{self._tick}] '
            f'joints=({self._pan_angle:.3f},{self._tilt_angle:.3f}) '
            f'face=({self._face_pos[0]:.2f},{self._face_pos[1]:.2f},{self._face_pos[2]:.2f}) '
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
