#!/usr/bin/env python3
"""VLA inference node — runs the trained ONNX model in the sim loop.

Subscribes to /camera/image_raw, runs ONNX inference per frame, publishes
/cmd_vel (Twist) with (pan_vel → angular.z, tilt_vel → angular.y).

This node replaces tracker_node and oracle_node when validating the trained
model in simulation.  It requires no torch or transformers — only onnxruntime
and numpy (both available in the sim container).

Parameters
----------
~checkpoint : str
    Absolute path to the ONNX model (e.g. /ws/src/ocelot/runs/v0.0-smoke/best.onnx).
~token_cache : str
    Path to the companion JSON token cache (<checkpoint-stem>_tokens.json).
    Defaults to <checkpoint directory>/<checkpoint stem>_tokens.json.
~command : str
    Language command to use for every frame.  Must be one of the keys in the
    token cache (or the closest match is used).  Default: "track the face".
~max_vel : float
    Clip the model output to ±max_vel rad/s.  Default: 0.3 (matches oracle
    max_velocity used during training data collection).
~max_accel : float
    Maximum velocity change per frame in rad/s² (assumes 10 Hz camera).
    Limits how quickly the commanded velocity can ramp up or down.
    Default: 0.0 (disabled).  Set > 0 to enable (e.g. 1.5).
~deadband : float
    Snap model output to 0.0 when |vel| < deadband (rad/s).  Default: 0.03.
    Addresses the model's inability to output exact zeros near the oracle
    deadband region (~20% of training labels are exactly 0.0).
~enabled : bool
    Publish /cmd_vel only when True.  Default: True.  Set False to pause without
    stopping the node (e.g. ros2 param set /vla_node enabled false).

Usage (inside sim container after colcon build):
    ros2 launch ocelot sim_launch.py use_vla:=true headless:=true

Or run directly:
    ros2 run ocelot vla_node \\
        --ros-args \\
        -p checkpoint:=/ws/src/ocelot/runs/v0.0-smoke/best.onnx \\
        -p command:="track the face"
"""

from __future__ import annotations

import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image

import numpy as np

from ocelot.vla_inference import VLAInferenceEngine, find_best_command, preprocess_bgr

_preprocess = preprocess_bgr
_find_best_command = find_best_command


class VLANode(Node):
    def __init__(self) -> None:
        super().__init__("vla_node")

        self.declare_parameter("checkpoint",  "")
        self.declare_parameter("token_cache", "")
        self.declare_parameter("command",     "track the face")
        self.declare_parameter("max_vel",     0.3)
        self.declare_parameter("max_accel",   0.0)
        self.declare_parameter("deadband",    0.03)
        self.declare_parameter("enabled",     True)

        checkpoint_str = self.get_parameter("checkpoint").value
        if not checkpoint_str:
            self.get_logger().fatal(
                "~checkpoint parameter is required. "
                "Pass -p checkpoint:=/path/to/best.onnx"
            )
            raise RuntimeError("checkpoint not set")

        cmd_requested = self.get_parameter("command").value
        token_cache_str = self.get_parameter("token_cache").value or None
        try:
            self._engine = VLAInferenceEngine(
                checkpoint=checkpoint_str,
                token_cache=token_cache_str,
            )
        except ImportError as e:
            self.get_logger().fatal(
                "onnxruntime not installed. Run: pip3 install onnxruntime"
            )
            raise RuntimeError("onnxruntime missing") from e
        except FileNotFoundError as e:
            self.get_logger().fatal(
                f"{e}. Run train/export_onnx.py on the host first."
            )
            raise RuntimeError("token cache missing") from e

        self.get_logger().info(f"Loading ONNX model from {checkpoint_str} …")
        self.get_logger().info(
            f"ONNX session ready (provider: {self._engine.provider})"
        )
        cmd_actual = self._engine.resolve_command(cmd_requested)
        if cmd_actual != cmd_requested:
            self.get_logger().warn(
                f"Command {cmd_requested!r} not in token cache; "
                f"using {cmd_actual!r} instead."
            )
        self.get_logger().info(f"Command: {cmd_actual!r}")
        self._command = cmd_actual

        self._bridge  = CvBridge()
        self._pub     = self.create_publisher(Twist, "/cmd_vel", 10)
        self._tick    = 0
        self._prev_pan: float = 0.0
        self._prev_tilt: float = 0.0

        # Use queue depth of 1, as we only want to process the latest frame
        # when hardware falls behind on processing
        self.create_subscription(Image, "/camera/image_raw", self._image_cb, 1)
        self.get_logger().info("VLA node ready — subscribed to /camera/image_raw")

    # ── callbacks ──────────────────────────────────────────────────────────

    def _image_cb(self, msg: Image) -> None:
        if not self.get_parameter("enabled").value:
            self._prev_pan = 0.0
            self._prev_tilt = 0.0
            return

        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"cv_bridge conversion failed: {exc}")
            return

        max_vel = float(self.get_parameter("max_vel").value)
        deadband = float(self.get_parameter("deadband").value)

        result = self._engine.predict_bgr(bgr, self._command)
        pan_vel = float(result["pan_vel"])
        tilt_vel = float(result["tilt_vel"])

        pan_in_db  = abs(pan_vel)  < deadband
        tilt_in_db = abs(tilt_vel) < deadband
        if pan_in_db and tilt_in_db:
            self.get_logger().info(
                f"[t{self._tick + 1}] IN DEADBAND  raw=({pan_vel:+.4f},{tilt_vel:+.4f}) db=±{deadband:.3f}"
            )
        if pan_in_db:
            pan_vel = 0.0
        if tilt_in_db:
            tilt_vel = 0.0

        pan_vel  = float(np.clip(pan_vel, -max_vel, max_vel))
        tilt_vel = float(np.clip(tilt_vel, -max_vel, max_vel))

        # Rate-limit: cap how fast velocity can change per frame.
        # At 10 Hz camera, max_delta = max_accel / 10.
        # Disabled by default (max_accel=0.0) — see oce-wp85: the limiter
        # adds +30% MSE and +60% transition lag with no training-time
        # equivalent, causing overshoot in closed-loop operation.
        max_accel = float(self.get_parameter("max_accel").value)
        if max_accel > 0.0:
            max_delta = max_accel / 10.0
            pan_vel  = float(np.clip(pan_vel,  self._prev_pan - max_delta,
                                     self._prev_pan + max_delta))
            tilt_vel = float(np.clip(tilt_vel, self._prev_tilt - max_delta,
                                     self._prev_tilt + max_delta))
        self._prev_pan = pan_vel
        self._prev_tilt = tilt_vel

        twist = Twist()
        twist.angular.z = pan_vel
        twist.angular.y = tilt_vel
        self._pub.publish(twist)

        self._tick += 1
        # Log first 200 frames verbosely, then once per 50 frames
        if self._tick <= 200 or self._tick % 50 == 0:
            self.get_logger().info(
                f"[t{self._tick}] pan={pan_vel:+.3f}  tilt={tilt_vel:+.3f} rad/s"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VLANode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
