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
    Clip the model output to ±max_vel rad/s.  Default: 1.0.
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

import json
import logging
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image

# ImageNet normalisation — must match train/dataset.py
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

log = logging.getLogger(__name__)


def _preprocess(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 HxWx3 → float32 (1, 3, 224, 224) ImageNet-normalised."""
    import cv2  # available in sim container (python3-opencv)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (224, 224):
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    f = rgb.astype(np.float32) / 255.0
    f = (f - _IMAGENET_MEAN) / _IMAGENET_STD       # HWC, normalised
    f = f.transpose(2, 0, 1)[np.newaxis, ...]       # (1, 3, 224, 224)
    return np.ascontiguousarray(f)


def _find_best_command(requested: str, cache: dict[str, dict]) -> str:
    """Return the closest command in the cache (exact match first, then substring)."""
    if requested in cache:
        return requested
    # Substring match on whitespace-normalised strings
    req_lower = requested.lower().strip()
    for key in cache:
        if req_lower in key.lower() or key.lower() in req_lower:
            return key
    # Fall back to "track the face" or the first key
    for fallback in ("track the face", "look at the person", "follow the person"):
        if fallback in cache:
            return fallback
    return next(iter(cache))


class VLANode(Node):
    def __init__(self) -> None:
        super().__init__("vla_node")

        self.declare_parameter("checkpoint",  "")
        self.declare_parameter("token_cache", "")
        self.declare_parameter("command",     "track the face")
        self.declare_parameter("max_vel",     1.0)
        self.declare_parameter("enabled",     True)

        checkpoint_str = self.get_parameter("checkpoint").value
        if not checkpoint_str:
            self.get_logger().fatal(
                "~checkpoint parameter is required. "
                "Pass -p checkpoint:=/path/to/best.onnx"
            )
            raise RuntimeError("checkpoint not set")

        ckpt = Path(checkpoint_str)
        token_cache_str = self.get_parameter("token_cache").value
        token_cache_path = (
            Path(token_cache_str)
            if token_cache_str
            else ckpt.with_name(ckpt.stem + "_tokens.json")
        )

        # Load ONNX model
        try:
            import onnxruntime as ort
        except ImportError as e:
            self.get_logger().fatal(
                "onnxruntime not installed. Run: pip3 install onnxruntime"
            )
            raise RuntimeError("onnxruntime missing") from e

        self.get_logger().info(f"Loading ONNX model from {ckpt} …")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._sess = ort.InferenceSession(str(ckpt), providers=providers)
        used_provider = self._sess.get_providers()[0]
        self.get_logger().info(f"ONNX session ready (provider: {used_provider})")

        # Load token cache
        if not token_cache_path.exists():
            self.get_logger().fatal(
                f"Token cache not found: {token_cache_path}. "
                "Run train/export_onnx.py on the host first."
            )
            raise RuntimeError("token cache missing")

        token_cache: dict[str, dict] = json.loads(token_cache_path.read_text())
        cmd_requested = self.get_parameter("command").value
        cmd_actual    = _find_best_command(cmd_requested, token_cache)
        if cmd_actual != cmd_requested:
            self.get_logger().warn(
                f"Command {cmd_requested!r} not in token cache; "
                f"using {cmd_actual!r} instead."
            )
        self.get_logger().info(f"Command: {cmd_actual!r}")

        tokens = token_cache[cmd_actual]
        # Pre-build fixed numpy arrays (reused every frame — no allocation in the hot path)
        self._input_ids      = np.array([tokens["input_ids"]],      dtype=np.int64)  # (1,77)
        self._attention_mask = np.array([tokens["attention_mask"]], dtype=np.int64)  # (1,77)

        self._bridge  = CvBridge()
        self._pub     = self.create_publisher(Twist, "/cmd_vel", 10)
        self._tick    = 0

        self.create_subscription(Image, "/camera/image_raw", self._image_cb, 10)
        self.get_logger().info("VLA node ready — subscribed to /camera/image_raw")

    # ── callbacks ──────────────────────────────────────────────────────────

    def _image_cb(self, msg: Image) -> None:
        if not self.get_parameter("enabled").value:
            return

        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"cv_bridge conversion failed: {exc}")
            return

        frame = _preprocess(bgr)

        actions = self._sess.run(
            ["actions"],
            {
                "frames":         frame,
                "input_ids":      self._input_ids,
                "attention_mask": self._attention_mask,
            },
        )[0]  # (1, 2)

        max_vel = float(self.get_parameter("max_vel").value)
        pan_vel  = float(np.clip(actions[0, 0], -max_vel, max_vel))
        tilt_vel = float(np.clip(actions[0, 1], -max_vel, max_vel))

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
