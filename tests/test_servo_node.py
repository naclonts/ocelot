"""Unit tests for stale-command handling in ocelot/servo_node.py."""

from __future__ import annotations

import sys
import types


class _FakeLogger:
    def __init__(self) -> None:
        self.warn_messages: list[str] = []

    def info(self, _msg: str) -> None:
        return

    def warn(self, msg: str) -> None:
        self.warn_messages.append(msg)

    def error(self, _msg: str) -> None:
        return


class _FakeBaseNode:
    def __init__(self, _name: str) -> None:
        self._params = {}
        self._logger = _FakeLogger()

    def declare_parameter(self, name: str, default):
        self._params[name] = default

    def get_parameter(self, name: str):
        return types.SimpleNamespace(value=self._params[name])

    def create_subscription(self, *_args, **_kwargs):
        return None

    def create_timer(self, *_args, **_kwargs):
        return None

    def get_logger(self):
        return self._logger

    def destroy_node(self):
        return None


class _FakeServo:
    def __init__(self) -> None:
        self.angle = None

    def set_pulse_width_range(self, _low: int, _high: int) -> None:
        return None


class _FakeServoKit:
    def __init__(self, channels: int) -> None:
        assert channels == 16
        self.servo = [_FakeServo() for _ in range(16)]


sys.modules["rclpy"] = types.SimpleNamespace(
    init=lambda *a, **k: None,
    shutdown=lambda: None,
)
sys.modules["rclpy.node"] = types.SimpleNamespace(Node=_FakeBaseNode)
sys.modules["geometry_msgs"] = types.SimpleNamespace()
sys.modules["geometry_msgs.msg"] = types.SimpleNamespace(Twist=object)
sys.modules["adafruit_servokit"] = types.SimpleNamespace(ServoKit=_FakeServoKit)

from ocelot import servo_node  # noqa: E402


def test_stale_command_zeroes_velocity(monkeypatch):
    now = {"value": 100.0}
    monkeypatch.setattr(servo_node.time, "monotonic", lambda: now["value"])

    node = servo_node.ServoNode()
    node._pan_vel = 0.2
    node._tilt_vel = -0.1
    node._last_cmd_time = 99.0

    node._integrate()

    assert node._pan_vel == 0.0
    assert node._tilt_vel == 0.0
    assert node._timed_out is True
    assert node.get_logger().warn_messages


def test_fresh_command_keeps_velocity(monkeypatch):
    now = {"value": 100.0}
    monkeypatch.setattr(servo_node.time, "monotonic", lambda: now["value"])

    node = servo_node.ServoNode()
    node._pan_vel = 0.2
    node._tilt_vel = -0.1
    node._last_cmd_time = 99.9

    node._integrate()

    assert node._pan_vel == 0.2
    assert node._tilt_vel == -0.1
    assert node._timed_out is False
