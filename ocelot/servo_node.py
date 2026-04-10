#!/usr/bin/env python3
"""Servo node — subscribes to /cmd_vel and drives pan/tilt servos via PCA9685."""

import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from adafruit_servokit import ServoKit

PAN_CH = 0
TILT_CH = 1
SERVO_MIN_PULSE = 500
SERVO_MAX_PULSE = 2500


class ServoNode(Node):
    def __init__(self):
        super().__init__('servo_node')

        self.declare_parameter('velocity_scale', 30.0)
        self.declare_parameter('pan_center', 90)
        self.declare_parameter('tilt_center', 180)
        self.declare_parameter('pan_limits', [0, 180])
        self.declare_parameter('tilt_limits', [90, 180])
        self.declare_parameter('command_timeout_sec', 0.25)

        self._vel_scale = self.get_parameter('velocity_scale').value
        pan_limits = self.get_parameter('pan_limits').value
        tilt_limits = self.get_parameter('tilt_limits').value
        self._pan_min, self._pan_max = pan_limits[0], pan_limits[1]
        self._tilt_min, self._tilt_max = tilt_limits[0], tilt_limits[1]
        self._command_timeout_sec = float(
            self.get_parameter('command_timeout_sec').value
        )

        self._pan_pos = float(self.get_parameter('pan_center').value)
        self._tilt_pos = float(self.get_parameter('tilt_center').value)
        self._pan_vel = 0.0
        self._tilt_vel = 0.0
        self._i2c_errors = 0
        self._last_cmd_time = time.monotonic()
        self._timed_out = False

        self._kit = ServoKit(channels=16)
        self._kit.servo[PAN_CH].set_pulse_width_range(SERVO_MIN_PULSE, SERVO_MAX_PULSE)
        self._kit.servo[TILT_CH].set_pulse_width_range(SERVO_MIN_PULSE, SERVO_MAX_PULSE)
        self._kit.servo[PAN_CH].angle = self._pan_pos
        self._kit.servo[TILT_CH].angle = self._tilt_pos
        self.get_logger().info(
            f'Servos ready. Pan center={self._pan_pos} Tilt center={self._tilt_pos}'
        )

        self.create_subscription(Twist, '/cmd_vel', self._cmd_cb, 10)
        self.create_timer(1.0 / 30.0, self._integrate)

    def _cmd_cb(self, msg: Twist):
        self._pan_vel = msg.angular.z
        self._tilt_vel = msg.angular.y
        self._last_cmd_time = time.monotonic()
        self._timed_out = False

    def _integrate(self):
        dt = 1.0 / 30.0
        if self._command_timeout_sec > 0.0:
            age = time.monotonic() - self._last_cmd_time
            if age > self._command_timeout_sec:
                if not self._timed_out and (self._pan_vel != 0.0 or self._tilt_vel != 0.0):
                    self.get_logger().warn(
                        f'/cmd_vel stale for {age:.3f}s; zeroing servo velocities'
                    )
                self._pan_vel = 0.0
                self._tilt_vel = 0.0
                self._timed_out = True

        self._pan_pos += self._pan_vel * self._vel_scale * dt
        self._tilt_pos += self._tilt_vel * self._vel_scale * dt

        self._pan_pos = max(self._pan_min, min(self._pan_max, self._pan_pos))
        self._tilt_pos = max(self._tilt_min, min(self._tilt_max, self._tilt_pos))

        try:
            self._kit.servo[PAN_CH].angle = self._pan_pos
            self._kit.servo[TILT_CH].angle = self._tilt_pos
            self._i2c_errors = 0
        except OSError as e:
            self._i2c_errors += 1
            self.get_logger().warn(
                f'I2C error #{self._i2c_errors}: {e} — attempting reinit'
            )
            if self._i2c_errors >= 5:
                self.get_logger().error('5 consecutive I2C errors — giving up')
                raise
            try:
                self._kit = ServoKit(channels=16)
                self._kit.servo[PAN_CH].set_pulse_width_range(SERVO_MIN_PULSE, SERVO_MAX_PULSE)
                self._kit.servo[TILT_CH].set_pulse_width_range(SERVO_MIN_PULSE, SERVO_MAX_PULSE)
            except OSError:
                pass  # will retry next tick

    def _center_and_stop(self):
        self._kit.servo[PAN_CH].angle = self.get_parameter('pan_center').value
        self._kit.servo[TILT_CH].angle = self.get_parameter('tilt_center').value

    def destroy_node(self):
        self._center_and_stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ServoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
