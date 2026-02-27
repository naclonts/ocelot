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

Servo inertia simulation
------------------------
Real servos ramp up to commanded velocity over ~100–400 ms rather than
responding instantly (as ideal Gazebo joints do).  Setting ``inertia_tau``
(seconds) applies a first-order low-pass filter to the velocity commands,
matching that ramp-up behaviour in sim so training data reflects real
hardware dynamics.

  v_out[t] = alpha * v_cmd[t] + (1 - alpha) * v_out[t-1]
  alpha     = dt / (dt + tau)         (dt = 1 / control_hz)

tau = 0.0  → pass-through (default, legacy behaviour)
tau = 0.3  → ~0.7 s to reach 90% of a step command at 10 Hz
"""

from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

import rclpy
from rclpy.node import Node

_CONTROL_HZ = 10.0   # must match URDF camera update rate and oracle publish rate


class CmdVelAdapter(Node):
    def __init__(self) -> None:
        super().__init__('cmd_vel_adapter')

        self.declare_parameter('inertia_tau', 0.3)

        tau = self.get_parameter('inertia_tau').value
        dt  = 1.0 / _CONTROL_HZ
        self._alpha     = dt / (dt + tau) if tau > 0.0 else 1.0
        self._pan_vel   = 0.0
        self._tilt_vel  = 0.0

        self._pub = self.create_publisher(
            Float64MultiArray,
            '/joint_group_velocity_controller/commands',
            10,
        )
        self.create_subscription(Twist, '/cmd_vel', self._cb, 10)
        self.get_logger().info(
            f'cmd_vel_adapter started — tau={tau:.2f}s  alpha={self._alpha:.3f}'
        )

    def _cb(self, msg: Twist) -> None:
        a = self._alpha
        self._pan_vel  = a * msg.angular.z + (1.0 - a) * self._pan_vel
        self._tilt_vel = a * msg.angular.y + (1.0 - a) * self._tilt_vel

        out = Float64MultiArray()
        out.data = [self._pan_vel, self._tilt_vel]
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
