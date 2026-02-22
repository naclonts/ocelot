#!/usr/bin/env python3
"""Adapter: /cmd_vel (Twist) → /joint_group_velocity_controller/commands.

tracker_node publishes geometry_msgs/Twist on /cmd_vel.
The ros2_control JointGroupVelocityController in Gazebo expects
std_msgs/Float64MultiArray on /joint_group_velocity_controller/commands.

This adapter translates between the two so the existing tracker_node drives
the simulated joints without any modification.

Sign convention (direct mapping — maintains parity with real hardware):
  commands[0] = twist.angular.z   → pan_joint  (axis Z, CCW positive)
  commands[1] = twist.angular.y   → tilt_joint (axis Y, nose-down positive)

With kp_pan = -0.8 (tracker_params.yaml):
  face RIGHT → angular.z < 0 → pan_joint vel < 0 → camera rotates CW
             → camera turns RIGHT → face centres ✓

Joint order matches controllers.yaml (pan_joint first, tilt_joint second).
"""

from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

import rclpy
from rclpy.node import Node


class CmdVelAdapter(Node):
    def __init__(self) -> None:
        super().__init__('cmd_vel_adapter')
        self._pub = self.create_publisher(
            Float64MultiArray,
            '/joint_group_velocity_controller/commands',
            10,
        )
        self.create_subscription(Twist, '/cmd_vel', self._cb, 10)
        self.get_logger().info('cmd_vel_adapter started — /cmd_vel → controller commands')

    def _cb(self, msg: Twist) -> None:
        out = Float64MultiArray()
        # Direct pass-through: angular.z → pan, angular.y → tilt.
        # No sign flip here; sign convention is handled by tracker kp_pan sign.
        out.data = [msg.angular.z, msg.angular.y]
        self._pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CmdVelAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
