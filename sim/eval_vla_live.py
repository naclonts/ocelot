#!/usr/bin/env python3
"""sim/eval_vla_live.py — live VLA evaluation harness.

Runs N reproducible scenarios from the training distribution against a
live VLA node and measures FK angular error.

Usage (via Makefile):
    make sim-vla-eval VLA_ONNX=runs/v0.1/best.onnx [SCENARIO_SEED=0] [N_SCENARIOS=5]

Usage (manual, inside sim container with scenario_world + VLA already running):
    python3 /ws/src/ocelot/sim/eval_vla_live.py --seed 0 --n-scenarios 5
"""

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from sim.scenario_generator.gazebo_bridge import GazeboBridge
from sim.scenario_generator.episode_runner import EpisodeRunner
from sim.scenario_generator.scenario import ScenarioGenerator

# ---------------------------------------------------------------------------
# FK constants — must match urdf/pan_tilt.urdf and oracle_node.py
# ---------------------------------------------------------------------------

PAN_Z = 0.03   # base_link  → pan_joint, Z offset
TILT_Z = 0.03  # pan_link   → tilt_joint, Z offset
CAM_X = 0.02   # tilt_link  → camera_link, X offset
CAM_Z = 0.01   # tilt_link  → camera_link, Z offset

# ---------------------------------------------------------------------------
# Evaluation parameters
# ---------------------------------------------------------------------------

PASS_THRESHOLD_DEG = 10.0  # mean angular error threshold for PASS
EVAL_DT = 0.1              # step interval seconds (10 Hz — matches camera)
WARMUP_S = 4.0             # seconds to run motion before measuring
EVAL_S = 10.0              # seconds to measure


# ---------------------------------------------------------------------------
# ROS2 node — latest-value face pose + joint state buffers
# ---------------------------------------------------------------------------

class EvalNode(Node):
    """Minimal ROS2 node that buffers the latest face pose and joint angles."""

    def __init__(self) -> None:
        super().__init__("eval_vla_live")
        self._lock = threading.Lock()
        self._face_pos: np.ndarray | None = None
        self._pan_angle: float = 0.0
        self._tilt_angle: float = 0.0
        self._face_received = threading.Event()
        self._joints_received = threading.Event()

        self._cmd_pub = self.create_publisher(String, '/episode/cmd', 1)

        self.create_subscription(Pose, "/model/face_0/pose", self._face_cb, 10)
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)

        # Background spin thread — same pattern as CollectNode in collect_data.py
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

    def _spin(self) -> None:
        rclpy.spin(self)

    def _face_cb(self, msg: Pose) -> None:
        with self._lock:
            self._face_pos = np.array([
                msg.position.x,
                msg.position.y,
                msg.position.z,
            ])
        self._face_received.set()

    def _joint_cb(self, msg: JointState) -> None:
        with self._lock:
            for i, name in enumerate(msg.name):
                if name == "pan_joint":
                    self._pan_angle = msg.position[i]
                elif name == "tilt_joint":
                    self._tilt_angle = msg.position[i]
        self._joints_received.set()

    def publish_cmd(self, text: str) -> None:
        """Publish the scenario language label to /episode/cmd."""
        msg = String()
        msg.data = text
        self._cmd_pub.publish(msg)

    def wait_ready(self, timeout: float = 60.0) -> bool:
        """Block until /joint_states received (confirms ros2_control + VLA are up).

        Does NOT wait for /model/face_0/pose — in scenario_world the face entity
        is spawned dynamically per episode, so no pose is published until after
        the first runner.setup() call.
        """
        return self._joints_received.wait(timeout=timeout)

    def wait_face_pose(self, timeout: float = 10.0) -> bool:
        """Block until at least one /model/face_0/pose message is received."""
        return self._face_received.wait(timeout=timeout)

    def get_face_pos(self) -> "np.ndarray | None":
        with self._lock:
            return self._face_pos.copy() if self._face_pos is not None else None

    def get_joint_angles(self) -> tuple[float, float]:
        with self._lock:
            return self._pan_angle, self._tilt_angle


# ---------------------------------------------------------------------------
# FK angular error computation
# ---------------------------------------------------------------------------

def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _fk_angular_error(
    face_pos: np.ndarray,
    pan: float,
    tilt: float,
) -> tuple[float, float]:
    """Return (pan_err_rad, tilt_err_rad) using analytic FK.

    Identical FK to oracle_node.py.
    """
    R_pan = _rz(pan)
    R_tilt = _ry(tilt)
    R_cam = R_pan @ R_tilt

    P_tilt = np.array([0.0, 0.0, PAN_Z + TILT_Z])
    P_cam = P_tilt + R_cam @ np.array([CAM_X, 0.0, CAM_Z])
    d_cam = R_cam.T @ (face_pos - P_cam)

    if d_cam[0] <= 0.0:
        # Face behind camera — return worst-case 180°
        return math.pi, math.pi

    pan_err = math.atan2(d_cam[1], d_cam[0])
    tilt_err = math.atan2(d_cam[2], d_cam[0])
    return pan_err, tilt_err


# ---------------------------------------------------------------------------
# Per-scenario evaluation
# ---------------------------------------------------------------------------

def eval_scenario(node: EvalNode, runner: EpisodeRunner, config) -> dict:
    """Run one scenario and return error statistics.

    Returns a dict with keys: mean, max, n, pass.
    """
    runner.setup(config)

    # Wait for face_0 pose to start publishing (it's spawned dynamically above).
    # In scenario_world the entity doesn't exist until setup_episode() creates it.
    node._face_received.clear()
    if not node.wait_face_pose(timeout=10.0):
        print("  WARNING: /model/face_0/pose not received after setup — skipping scenario")
        runner.teardown()
        return {"mean": 999.0, "max": 999.0, "n": 0, "pass": False}

    # Warmup — drive motion but don't measure (let VLA converge)
    t0 = time.monotonic()
    while time.monotonic() - t0 < WARMUP_S:
        runner.step(time.monotonic() - t0)
        time.sleep(EVAL_DT)

    # Measurement loop
    errors: list[float] = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < EVAL_S:
        t = time.monotonic() - t0
        runner.step(t)
        face_pos = node.get_face_pos()
        pan, tilt = node.get_joint_angles()
        if face_pos is not None:
            pan_err, tilt_err = _fk_angular_error(face_pos, pan, tilt)
            total_deg = math.degrees(math.sqrt(pan_err**2 + tilt_err**2))
            errors.append(total_deg)
        time.sleep(EVAL_DT)

    runner.teardown()
    time.sleep(1.0)  # let Gazebo finish despawning before next setup

    if not errors:
        return {"mean": 999.0, "max": 999.0, "n": 0, "pass": False}
    mean_err = float(np.mean(errors))
    return {
        "mean": mean_err,
        "max": float(np.max(errors)),
        "n": len(errors),
        "pass": mean_err < PASS_THRESHOLD_DEG,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate a live VLA node against N training-distribution scenarios."
    )
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed for scenario sampling")
    p.add_argument("--n-scenarios", type=int, default=5, help="Number of scenarios to evaluate")
    p.add_argument(
        "--faces-dir",
        default=str(_root / "sim" / "scenario_generator"),
        help="Dir containing face_descriptions*.json (ScenarioGenerator computes assets/faces from here)",
    )
    p.add_argument(
        "--bg-dir",
        default=str(_root / "sim" / "assets" / "backgrounds"),
        help="Path to background PNG directory",
    )
    args = p.parse_args()

    rclpy.init()
    node = EvalNode()

    print("Waiting for Gazebo + VLA to publish /joint_states …")
    if not node.wait_ready(timeout=90.0):
        print("ERROR: timed out waiting for /joint_states")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)
    print("Ready.\n")

    bridge = GazeboBridge(world="scenario_world")
    runner = EpisodeRunner(bridge)
    generator = ScenarioGenerator(
        faces_dir=Path(args.faces_dir),
        backgrounds_dir=Path(args.bg_dir),
    )

    results = []
    for i in range(args.n_scenarios):
        seed = args.seed + i
        config = generator.sample(seed=seed)
        motion = config.faces[0].motion if config.faces else "unknown"
        print(
            f"Scenario {i + 1}/{args.n_scenarios}  "
            f"seed={seed}  motion={motion}  label={config.label_key}"
        )
        node.publish_cmd(f"seed={seed} | {config.language_label}")
        result = eval_scenario(node, runner, config)
        result.update({"seed": seed, "motion": motion, "label": config.label_key})
        results.append(result)
        status = "PASS" if result["pass"] else "FAIL"
        print(f"  mean={result['mean']:.1f}°  max={result['max']:.1f}°  n={result['n']}  [{status}]")

    # Summary table
    print("\n--- Summary ---")
    print(f"{'#':>3}  {'seed':>6}  {'motion':<16}  {'label':<14}  {'mean°':>6}  {'max°':>6}  pass")
    for i, r in enumerate(results):
        tick = "Y" if r["pass"] else "N"
        print(
            f"{i + 1:>3}  {r['seed']:>6}  {r['motion']:<16}  {r['label']:<14}  "
            f"{r['mean']:>6.1f}  {r['max']:>6.1f}  {tick}"
        )
    pass_rate = sum(r["pass"] for r in results) / len(results) * 100
    mean_all = float(np.mean([r["mean"] for r in results]))
    print(
        f"\nOverall: mean={mean_all:.1f}°  pass_rate={pass_rate:.0f}%  "
        f"(threshold={PASS_THRESHOLD_DEG}°)"
    )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
