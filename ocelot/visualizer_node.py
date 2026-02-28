#!/usr/bin/env python3
"""Visualizer node — annotates /camera/image_raw and publishes /camera/image_annotated.

Subscribes to tracker_node's outputs (/tracking/face_roi, /cmd_vel) so it shows
exactly the same detection data the tracker is acting on — no duplicate Haar cascade.

Draws:
  - Faint deadband circle at image center
  - Center crosshair
  - Face bounding box + center dot (from /tracking/face_roi)
  - Error vector arrow from image center to face center
  - Pixel error text near bounding box
  - cmd_vel (pan/tilt) as text overlay
"""

import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image, RegionOfInterest
from std_msgs.msg import String
import cv2

# Colours (BGR)
_GREEN  = (0, 220, 80)
_ORANGE = (0, 180, 255)
_YELLOW = (80, 230, 255)
_GREY   = (180, 180, 180)
_WHITE  = (255, 255, 255)


class VisualizerNode(Node):
    def __init__(self):
        super().__init__('visualizer_node')

        self.declare_parameter('deadband', 30)

        self._bridge       = CvBridge()
        self._latest_twist = Twist()
        self._latest_roi   = RegionOfInterest()  # width==0 → no face
        self._latest_cmd   = ''

        self.create_subscription(Image,           '/camera/image_raw',    self._image_cb, 10)
        self.create_subscription(Twist,           '/cmd_vel',             self._cmd_vel_cb, 10)
        self.create_subscription(RegionOfInterest, '/tracking/face_roi',  self._roi_cb, 10)
        self.create_subscription(String,          '/episode/cmd',         self._cmd_cb, 1)
        self._pub = self.create_publisher(Image, '/camera/image_annotated', 10)

        self.get_logger().info('Visualizer node started → /camera/image_annotated')

    def _cmd_vel_cb(self, msg: Twist):
        self._latest_twist = msg

    def _roi_cb(self, msg: RegionOfInterest):
        self._latest_roi = msg

    def _cmd_cb(self, msg: String):
        self._latest_cmd = msg.data

    def _image_cb(self, msg: Image):
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        deadband = self.get_parameter('deadband').value

        # --- deadband circle (faint) ---
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), deadband, _WHITE, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # --- center crosshair ---
        arm = 14
        cv2.line(frame, (cx - arm, cy), (cx + arm, cy), _GREY, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - arm), (cx, cy + arm), _GREY, 1, cv2.LINE_AA)

        # --- face annotations (only when tracker has a detection) ---
        roi = self._latest_roi
        if roi.width > 0:
            x, y = int(roi.x_offset), int(roi.y_offset)
            fw, fh = int(roi.width), int(roi.height)
            fx, fy = x + fw // 2, y + fh // 2
            ex, ey = fx - cx, fy - cy

            # bounding box
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), _GREEN, 2, cv2.LINE_AA)

            # face center dot
            cv2.circle(frame, (fx, fy), 5, _GREEN, -1, cv2.LINE_AA)

            # error vector arrow
            cv2.arrowedLine(frame, (cx, cy), (fx, fy), _ORANGE, 2,
                            cv2.LINE_AA, tipLength=0.2)

            # pixel error label near bounding box
            cv2.putText(frame, f'err ({ex:+d}, {ey:+d})px',
                        (x, max(y - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, _GREEN, 1, cv2.LINE_AA)

        # --- cmd_vel overlay ---
        twist = self._latest_twist
        cv2.putText(frame, f'pan  {twist.angular.z:+.2f}',
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _YELLOW, 1, cv2.LINE_AA)
        cv2.putText(frame, f'tilt {twist.angular.y:+.2f}',
                    (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _YELLOW, 1, cv2.LINE_AA)

        # --- episode command overlay (bottom-left) ---
        if self._latest_cmd:
            cv2.putText(frame, self._latest_cmd,
                        (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1, cv2.LINE_AA)

        out = self._bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        out.header = msg.header
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = VisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
