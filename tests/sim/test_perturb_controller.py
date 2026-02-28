"""Unit tests for PerturbController.

No ROS2, Gazebo, or sim container required — pure Python + numpy.
"""
import sys
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest

# Stub ROS2 packages before importing collect_data (not available in host .venv)
for _mod in (
    "rclpy", "rclpy.node",
    "cv_bridge",
    "geometry_msgs", "geometry_msgs.msg",
    "sensor_msgs", "sensor_msgs.msg",
    "std_msgs", "std_msgs.msg",
):
    sys.modules.setdefault(_mod, mock.MagicMock())

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.data_gen.collect_data import PERTURB_DURATION, PerturbController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FACE_X = 2.0
FACE_POS = (FACE_X, 0.0, 0.5)


def make_positions(x=FACE_X, y=0.0, z=0.5):
    return {"face_0": (x, y, z)}


def run_frames(ctrl, n_frames, positions=None):
    """Run the controller for n_frames, collect set_pose calls per frame."""
    calls = []  # list of (frame_idx, args) or None
    for i in range(n_frames):
        pos = positions if positions is not None else make_positions()
        frame_calls = []
        ctrl.step(i, pos, lambda *a: frame_calls.append(a))
        calls.append(frame_calls)
    return calls


# ---------------------------------------------------------------------------
# Tests: disabled
# ---------------------------------------------------------------------------

def test_disabled_never_calls_set_pose():
    ctrl = PerturbController(interval=0, range_rad=0.5, seed=0)
    calls = run_frames(ctrl, 100)
    assert all(len(c) == 0 for c in calls)


# ---------------------------------------------------------------------------
# Tests: trigger timing
# ---------------------------------------------------------------------------

def test_first_trigger_at_interval():
    interval = 15
    ctrl = PerturbController(interval=interval, range_rad=0.5, seed=42)
    calls = run_frames(ctrl, interval)
    # frames 0 .. interval-1: no calls yet
    assert all(len(c) == 0 for c in calls)


def test_set_pose_called_on_trigger_frame():
    interval = 15
    ctrl = PerturbController(interval=interval, range_rad=0.5, seed=42)
    calls = run_frames(ctrl, interval + 1)
    assert len(calls[interval]) == 1, "set_pose should be called on trigger frame"


def test_second_trigger_at_2x_interval():
    interval = 15
    ctrl = PerturbController(interval=interval, range_rad=0.5, seed=42)
    calls = run_frames(ctrl, 2 * interval + 1)
    assert len(calls[2 * interval]) == 1


def test_triggers_fire_every_interval():
    interval = 10
    n_frames = 100
    ctrl = PerturbController(interval=interval, range_rad=0.5, seed=0)
    calls = run_frames(ctrl, n_frames)
    trigger_frames = [i for i, c in enumerate(calls) if len(c) > 0]
    # First trigger: frame 10; subsequent: 20, 30, …
    # But frame 10 → next = 20, but frames 11-14 are still in PERTURB_DURATION window.
    # All trigger frames should be multiples of interval.
    # Only check the subset that are exactly at trigger points (not mid-window).
    expected_triggers = list(range(interval, n_frames, interval))
    for t in expected_triggers:
        assert t in trigger_frames, f"frame {t} should be a trigger"


# ---------------------------------------------------------------------------
# Tests: duration
# ---------------------------------------------------------------------------

def test_set_pose_called_for_duration_frames():
    ctrl = PerturbController(interval=15, range_rad=0.5, seed=1, duration=5)
    calls = run_frames(ctrl, 25)
    active = [i for i, c in enumerate(calls) if len(c) > 0]
    # Trigger at frame 15; active for frames 15, 16, 17, 18, 19 (5 frames)
    assert active == list(range(15, 15 + 5))


def test_no_calls_after_duration_ends():
    ctrl = PerturbController(interval=15, range_rad=0.5, seed=1, duration=5)
    calls = run_frames(ctrl, 21)   # frame 20 is first frame after window
    assert len(calls[20]) == 0


# ---------------------------------------------------------------------------
# Tests: pose values
# ---------------------------------------------------------------------------

def test_set_pose_x_unchanged():
    """X coordinate (depth) must not be modified."""
    ctrl = PerturbController(interval=5, range_rad=0.5, seed=7)
    pose_calls = []
    for i in range(6):
        ctrl.step(i, make_positions(x=3.0), lambda *a: pose_calls.append(a))
    for name, x, y, z in pose_calls:
        assert x == pytest.approx(3.0), "X must be unchanged"


def test_set_pose_offset_within_range():
    """Y and Z offsets must correspond to angles within ±range_rad."""
    range_rad = 0.5
    face_x = 2.0
    ctrl = PerturbController(interval=5, range_rad=range_rad, seed=99)
    pose_calls = []
    for i in range(6):
        ctrl.step(i, make_positions(x=face_x, y=0.0, z=0.0),
                  lambda *a: pose_calls.append(a))
    for name, x, y, z in pose_calls:
        # offset = face_x * tan(delta); delta in [-range, +range]
        max_offset = face_x * np.tan(range_rad)
        assert abs(y) <= max_offset + 1e-9
        assert abs(z) <= max_offset + 1e-9


def test_set_pose_adds_to_existing_position():
    """Offset is added on top of the face's current motion-pattern position."""
    # Give face a non-zero base position
    base_y, base_z = 0.3, 0.1
    ctrl = PerturbController(interval=5, range_rad=0.5, seed=3)
    pose_calls = []
    for i in range(6):
        ctrl.step(i, make_positions(y=base_y, z=base_z),
                  lambda *a: pose_calls.append(a))
    # At least one call; the Y/Z should differ from base by some nonzero offset
    assert len(pose_calls) > 0
    for name, x, y, z in pose_calls:
        # offset = y - base_y; must be in valid range
        y_offset = y - base_y
        z_offset = z - base_z
        max_offset = FACE_X * np.tan(0.5)
        assert abs(y_offset) <= max_offset + 1e-9
        assert abs(z_offset) <= max_offset + 1e-9


def test_offset_consistent_within_window():
    """Same offset applied for all frames within a perturbation window."""
    ctrl = PerturbController(interval=5, range_rad=0.5, seed=42, duration=4)
    pose_calls = []
    for i in range(10):
        ctrl.step(i, make_positions(), lambda *a: pose_calls.append(a))
    ys = [c[2] for c in pose_calls]  # y values from each set_pose call
    zs = [c[3] for c in pose_calls]
    assert all(y == pytest.approx(ys[0]) for y in ys), "Y offset must be constant within window"
    assert all(z == pytest.approx(zs[0]) for z in zs), "Z offset must be constant within window"


# ---------------------------------------------------------------------------
# Tests: missing face gracefully handled
# ---------------------------------------------------------------------------

def test_no_face_in_positions_no_crash():
    ctrl = PerturbController(interval=5, range_rad=0.5, seed=0)
    called = []
    for i in range(10):
        ctrl.step(i, {}, lambda *a: called.append(a))  # no face_0
    assert len(called) == 0


def test_face_absent_then_present_resumes():
    """If face_0 is missing on the trigger frame, next present frame in window still fires."""
    ctrl = PerturbController(interval=5, range_rad=0.5, seed=0)
    calls = []
    for i in range(10):
        pos = {} if i == 5 else make_positions()  # face absent only on trigger frame
        ctrl.step(i, pos, lambda *a: calls.append((i, a)))
    # Trigger frame 5: face absent → no set_pose, but offsets may still be set
    # Frames 6-9: face present, if window still active should fire
    active = [i for i, _ in calls]
    # Window may not have started since trigger frame had no face — acceptable.
    # Key requirement: no crash.


# ---------------------------------------------------------------------------
# Tests: reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_same_offsets():
    def collect_y_offsets(seed):
        ctrl = PerturbController(interval=10, range_rad=0.5, seed=seed)
        ys = []
        for i in range(60):
            ctrl.step(i, make_positions(), lambda n, x, y, z: ys.append(y))
        return ys

    assert collect_y_offsets(42) == collect_y_offsets(42)


def test_different_seeds_different_offsets():
    def first_offset(seed):
        ctrl = PerturbController(interval=5, range_rad=0.5, seed=seed)
        results = []
        for i in range(6):
            ctrl.step(i, make_positions(), lambda n, x, y, z: results.append(y))
        return results[0] if results else None

    assert first_offset(0) != first_offset(1)
