#!/usr/bin/env python3
"""Tracker node (placeholder) — subscribes to image, publishes zero velocity.

Week 3 will replace this with real Haar cascade face tracking logic.
For now it just proves the full topic chain compiles and runs end-to-end.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


class TrackerNode(Node):
    def __init__(self):
        super().__init__('tracker_node')

        self.declare_parameter('enabled', False)

        self.create_subscription(Image, '/camera/image_raw', self._image_cb, 10)
        self._pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(0.1, self._publish_zero)

        self.get_logger().info('Tracker node started (placeholder — publishing zero velocity)')

    def _image_cb(self, msg: Image):
        # Placeholder: image received but not processed yet
        pass

    def _publish_zero(self):
        self._pub.publish(Twist())


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
