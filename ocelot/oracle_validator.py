#!/usr/bin/env python3
"""Oracle validator — measures face tracking error in pixel space.

Runs alongside oracle_node in a second terminal.  Uses the same closed-form FK
as oracle_node to project the face billboard world position into image pixel
coordinates, then reports how far that pixel is from the image center.

The oracle's goal is to drive that error to zero, so pixel distance from center
IS the tracking error.

Camera intrinsics (derived from urdf/pan_tilt.urdf):
    width=640, height=480, horizontal_fov=2.0944 rad (120°)
    fx = fy = (w/2) / tan(hfov/2) = 320 / tan(1.0472) ≈ 184.8
    cx=320, cy=240

Camera frame convention (camera_link, rpy=0,0,0 on tilt_link):
    +X = forward (optical axis)
    +Y = left
    +Z = up

Pinhole projection (d_cam in camera frame):
    u = cx  +  fx * (-d_cam[1] / d_cam[0])   # -Y because image X goes right
    v = cy  +  fy * (-d_cam[2] / d_cam[0])   # -Z because image Y goes down

Usage (while sim_launch.py is running with use_oracle:=true):
    ros2 run ocelot oracle_validator
"""

import math
import statistics

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from sensor_msgs.msg import JointState

# ── Camera intrinsics (from urdf/pan_tilt.urdf) ───────────────────────────────
_W, _H = 640, 480
_HFOV = 2.0944  # radians (120°)
_FX = _FY = (_W / 2) / math.tan(_HFOV / 2)   # ≈ 184.8 px
_CX, _CY = _W / 2, _H / 2                     # 320, 240

# ── FK constants (must match oracle_node.py and urdf/pan_tilt.urdf) ──────────
_PAN_Z  = 0.03  # base_link → pan_joint, Z offset
_TILT_Z = 0.03  # pan_link  → tilt_joint, Z offset (in pan frame)
_CAM_X  = 0.02  # tilt_link → camera_link, X offset
_CAM_Z  = 0.01  # tilt_link → camera_link, Z offset

# ── Reporting ─────────────────────────────────────────────────────────────────
_REPORT_INTERVAL_S = 10.0   # print rolling stats every N seconds
_HISTORY_LIMIT     = 10_000 # cap in-memory sample list


def _rz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def _ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]])


class OracleValidator(Node):
    def __init__(self) -> None:
        super().__init__('oracle_validator')

        self._pan_angle:  float = 0.0
        self._tilt_angle: float = 0.0
        self._face_pos:   np.ndarray | None = None

        self._errors: list[float] = []   # pixel distances from center
        self._n_behind = 0               # samples where face was behind camera

        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)
        self.create_subscription(Pose, '/model/face_0/pose', self._face_pose_cb, 10)

        self.create_timer(0.05, self._measure)                        # 20 Hz
        self.create_timer(_REPORT_INTERVAL_S, self._report)

        self.get_logger().info(
            f'Oracle validator running — reporting every {_REPORT_INTERVAL_S:.0f} s'
        )

    # ── callbacks ──────────────────────────────────────────────────────────────

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

    # ── measurement ────────────────────────────────────────────────────────────

    def _measure(self) -> None:
        if self._face_pos is None:
            return

        R_pan  = _rz(self._pan_angle)
        R_tilt = _ry(self._tilt_angle)
        R_cam  = R_pan @ R_tilt

        P_tilt = np.array([0.0, 0.0, _PAN_Z + _TILT_Z])
        P_cam  = P_tilt + R_cam @ np.array([_CAM_X, 0.0, _CAM_Z])
        d_cam  = R_cam.T @ (self._face_pos - P_cam)

        if d_cam[0] <= 0.0:
            self._n_behind += 1
            return

        # Project to pixel coordinates.
        # Camera +X is forward, +Y is left, +Z is up.
        # Image u increases rightward  → negate d_cam[1]
        # Image v increases downward   → negate d_cam[2]
        u = _CX + _FX * (-d_cam[1] / d_cam[0])
        v = _CY + _FY * (-d_cam[2] / d_cam[0])

        px_err = math.hypot(u - _CX, v - _CY)
        self._errors.append(px_err)

        # Trim list so we don't grow unboundedly during long runs.
        if len(self._errors) > _HISTORY_LIMIT:
            self._errors = self._errors[-_HISTORY_LIMIT:]

        self.get_logger().debug(
            f'face_px=({u:.1f},{v:.1f})  err={px_err:.1f} px'
        )

    # ── reporting ──────────────────────────────────────────────────────────────

    def _report(self) -> None:
        errors = self._errors[-200:] # last 10 seconds
        n = len(errors)
        if n == 0:
            self.get_logger().info('No samples yet — waiting for face pose data.')
            return

        mean_err = statistics.mean(errors)
        std_err  = statistics.stdev(errors) if n > 1 else 0.0
        max_err  = max(errors)
        p95_err  = sorted(errors)[int(0.95 * n)]

        gate = mean_err < 5.0
        gate_str = 'PASS (<5 px)' if gate else 'FAIL (>=5 px)'

        self.get_logger().info(
            f'\n--- Oracle Tracking Error ({n} samples) ---\n'
            f'  mean : {mean_err:6.2f} px   [{gate_str}]\n'
            f'  std  : {std_err:6.2f} px\n'
            f'  p95  : {p95_err:6.2f} px\n'
            f'  max  : {max_err:6.2f} px\n'
            f'  behind-camera skips: {self._n_behind}'
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OracleValidator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Print final summary on Ctrl-C.
        n = len(node._errors)
        if n > 0:
            mean_err = statistics.mean(node._errors)
            gate = mean_err < 5.0
            node.get_logger().info(
                f'\n=== Final summary ({n} samples) ===\n'
                f'  mean error : {mean_err:.2f} px  '
                f'[{"PASS" if gate else "FAIL"} — goal <5 px]\n'
                f'  max  error : {max(node._errors):.2f} px'
            )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
