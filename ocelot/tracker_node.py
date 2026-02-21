#!/usr/bin/env python3
"""Tracker node — Haar cascade face tracking, publishes velocity commands to /cmd_vel.

Error convention (before any sign flip):
  face right of center  → positive angular.z  → pan right
  face below center     → positive angular.y  → tilt down

If the camera moves away from your face, negate kp_pan or kp_tilt in params.
"""

import glob
import os
import sys

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image, RegionOfInterest


def _find_cascade() -> str:
    name = 'haarcascade_frontalface_default.xml'
    # pip opencv-python / opencv-python-headless
    if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
        p = os.path.join(cv2.data.haarcascades, name)
        if os.path.exists(p):
            return p
    # Search sys.path entries (catches pip installs that shadow the apt .so)
    for d in sys.path:
        p = os.path.join(d, 'cv2', 'data', name)
        if os.path.exists(p):
            return p
    # Fall back to the project .venv (bind-mounted in Docker at /ws/src/ocelot)
    repo_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    matches = glob.glob(os.path.join(repo_root, '.venv', '**', name), recursive=True)
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f'{name} not found. Run: pip3 install opencv-python-headless'
    )


_CASCADE_PATH = _find_cascade()


class TrackerNode(Node):
    def __init__(self):
        super().__init__('tracker_node')

        self.declare_parameter('enabled', False)
        self.declare_parameter('kp_pan', 0.3)    # velocity per normalized-error unit
        self.declare_parameter('kp_tilt', 0.3)
        self.declare_parameter('deadband', 30)   # pixels
        self.declare_parameter('max_velocity', 2.0)
        self.declare_parameter('min_neighbors', 3)
        self.declare_parameter('min_face_size', 40)

        self._bridge = CvBridge()
        self._cascade = cv2.CascadeClassifier(_CASCADE_PATH)
        if self._cascade.empty():
            self.get_logger().error(f'Failed to load Haar cascade from {_CASCADE_PATH}')

        self.create_subscription(Image, '/camera/image_raw', self._image_cb, 10)
        self._pub     = self.create_publisher(Twist,           '/cmd_vel',           10)
        self._roi_pub = self.create_publisher(RegionOfInterest, '/tracking/face_roi', 10)

        self.get_logger().info('Tracker node started')

    def _image_cb(self, msg: Image):
        if not self.get_parameter('enabled').value:
            self._pub.publish(Twist())
            self._roi_pub.publish(RegionOfInterest())
            return

        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        min_neighbors = self.get_parameter('min_neighbors').value
        min_face_size = self.get_parameter('min_face_size').value
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=min_neighbors,
            minSize=(min_face_size, min_face_size),
        )

        if len(faces) == 0:
            self._pub.publish(Twist())
            self._roi_pub.publish(RegionOfInterest())
            return

        # Track the largest detected face
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        cx = x + fw // 2
        cy = y + fh // 2

        error_x = cx - w // 2   # pixels; positive = face is right of center
        error_y = cy - h // 2   # pixels; positive = face is below center

        kp_pan = self.get_parameter('kp_pan').value
        kp_tilt = self.get_parameter('kp_tilt').value
        deadband = self.get_parameter('deadband').value
        max_vel = self.get_parameter('max_velocity').value

        # Normalize to [-1, 1] so kp is independent of resolution
        norm_x = error_x / (w / 2)
        norm_y = error_y / (h / 2)

        twist = Twist()
        if abs(error_x) > deadband:
            twist.angular.z = float(np.clip(kp_pan * norm_x, -max_vel, max_vel))
        if abs(error_y) > deadband:
            twist.angular.y = float(np.clip(kp_tilt * norm_y, -max_vel, max_vel))

        roi = RegionOfInterest()
        roi.x_offset = int(x)
        roi.y_offset = int(y)
        roi.width    = int(fw)
        roi.height   = int(fh)
        self._roi_pub.publish(roi)

        self._pub.publish(twist)
        self.get_logger().debug(
            f'face=({cx},{cy}) err=({error_x:+d},{error_y:+d}) '
            f'vel=({twist.angular.z:+.2f},{twist.angular.y:+.2f})'
        )


def main(args=None):
    rclpy.init(args=args)
    node = TrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
