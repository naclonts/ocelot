#!/usr/bin/env python3
"""Pi-side client node that sends camera frames to a remote VLA server."""

from __future__ import annotations

import json
from urllib.parse import quote
from urllib.request import Request, urlopen

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image


class RemoteVLAClientNode(Node):
    def __init__(self) -> None:
        super().__init__("remote_vla_client_node")

        self.declare_parameter("server_url", "http://127.0.0.1:8765/infer")
        self.declare_parameter("command", "track the face")
        self.declare_parameter("jpeg_quality", 85)
        self.declare_parameter("request_timeout_sec", 0.2)
        self.declare_parameter("max_vel", 0.3)
        self.declare_parameter("deadband", 0.03)
        self.declare_parameter("enabled", True)

        self._bridge = CvBridge()
        self._pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._server_url = self.get_parameter("server_url").value.rstrip("/")
        self._command = self.get_parameter("command").value
        self._tick = 0
        self._last_command = self._command

        self.create_subscription(Image, "/camera/image_raw", self._image_cb, 1)
        self.get_logger().info(
            f"Remote VLA client ready — sending frames to {self._server_url}"
        )

    def _image_cb(self, msg: Image) -> None:
        if not self.get_parameter("enabled").value:
            return

        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"cv_bridge conversion failed: {exc}")
            return

        quality = int(self.get_parameter("jpeg_quality").value)
        ok, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if not ok:
            self.get_logger().warn("jpeg encode failed")
            return

        command = self.get_parameter("command").value
        if command != self._last_command:
            self._last_command = command

        url = f"{self._server_url}?command={quote(command)}"
        timeout = float(self.get_parameter("request_timeout_sec").value)
        request = Request(
            url,
            data=encoded.tobytes(),
            method="POST",
            headers={"Content-Type": "image/jpeg"},
        )

        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            self.get_logger().warn(f"remote VLA request failed: {exc}")
            return

        max_vel = float(self.get_parameter("max_vel").value)
        deadband = float(self.get_parameter("deadband").value)

        pan_vel = float(np.clip(payload["pan_vel"], -max_vel, max_vel))
        tilt_vel = float(np.clip(payload["tilt_vel"], -max_vel, max_vel))
        if abs(pan_vel) < deadband:
            pan_vel = 0.0
        if abs(tilt_vel) < deadband:
            tilt_vel = 0.0

        twist = Twist()
        twist.angular.z = pan_vel
        twist.angular.y = tilt_vel
        self._pub.publish(twist)

        self._tick += 1
        if self._tick <= 50 or self._tick % 50 == 0:
            self.get_logger().info(
                "[t%d] pan=%+.3f tilt=%+.3f remote_total=%.1fms infer=%.1fms command=%r"
                % (
                    self._tick,
                    pan_vel,
                    tilt_vel,
                    float(payload.get("total_latency_ms", -1.0)),
                    float(payload.get("inference_latency_ms", -1.0)),
                    payload.get("command", command),
                )
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RemoteVLAClientNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
