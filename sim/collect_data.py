#!/usr/bin/env python3
"""sim/collect_data.py — Phase 2 Step 7 data collection pipeline.

Drives sequential simulation episodes, captures synchronized (frame, label,
velocity) tuples from the running oracle, and writes compressed HDF5 episode
files that become the VLA training dataset.

Usage (inside the sim container, with sim_launch.py already running):

    python3 /ws/src/ocelot/sim/collect_data.py \\
        --n_episodes 100 \\
        --output /ws/src/ocelot/dataset \\
        --base_seed 0 \\
        [--base_ep 0]

Prerequisites:
    ros2 launch ocelot sim_launch.py world:=scenario_world use_oracle:=true headless:=true
"""

import argparse
import json
import logging
import random
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Add project root to sys.path so "sim.scenario_generator.*" imports work.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import h5py
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CAMERA_HZ          = 10      # Hz — must match URDF <update_rate>10</update_rate>
EPISODE_SECS       = 10.0    # seconds of recorded data per episode
WARMUP_SECS        = 4.0     # seconds before recording starts (oracle convergence)
IMG_SIZE           = (224, 224)   # (width, height) for cv2.resize
FRAMES_PER_EPISODE = int(EPISODE_SECS * CAMERA_HZ)   # 100


# ---------------------------------------------------------------------------
# ROS2 node — latest-value buffers + event-driven image sync
# ---------------------------------------------------------------------------

class CollectNode(Node):
    """Minimal ROS2 node with latest-value buffers for image and cmd_vel.

    Camera frames drive the collection loop.  /cmd_vel and /joint_states are
    sampled at the moment each frame is captured (latest-value semantics —
    no message_filters synchronisation needed).
    """

    def __init__(self):
        super().__init__("collect_data")
        self._cv_bridge = CvBridge()
        self._lock = threading.Lock()

        self._latest_image: Image | None = None
        self._latest_cmd_vel: Twist | None = None

        # Signals that a new /camera/image_raw has arrived.  The collection
        # loop clears this before waiting so it always blocks for the *next*
        # frame rather than immediately returning a stale one.
        self._image_event = threading.Event()

        self.create_subscription(Image,      "/camera/image_raw", self._on_image,        10)
        self.create_subscription(JointState, "/joint_states",     self._on_joint_states, 10)
        self.create_subscription(Twist,      "/cmd_vel",          self._on_cmd_vel,      10)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_image(self, msg: Image) -> None:
        with self._lock:
            self._latest_image = msg
        self._image_event.set()

    def _on_joint_states(self, _msg: JointState) -> None:
        pass  # reserved for future diagnostics; not used in the capture loop

    def _on_cmd_vel(self, msg: Twist) -> None:
        with self._lock:
            self._latest_cmd_vel = msg

    # ------------------------------------------------------------------
    # Public helpers called from the collection loop
    # ------------------------------------------------------------------

    def wait_for_new_image(self, timeout: float = 5.0) -> bool:
        """Block until a new /camera/image_raw message arrives.

        Returns True if an image arrived within timeout seconds, False otherwise.
        """
        self._image_event.clear()
        return self._image_event.wait(timeout=timeout)

    def get_latest_frame_rgb(self) -> "np.ndarray | None":
        """Return the latest camera frame as an (H, W, 3) uint8 RGB array."""
        with self._lock:
            msg = self._latest_image
        if msg is None:
            return None
        return self._cv_bridge.imgmsg_to_cv2(msg, "rgb8")

    def get_pan_tilt_vel(self) -> tuple[float, float]:
        """Return (pan_vel, tilt_vel) from the latest /cmd_vel.

        pan_vel  = angular.z  (oracle output, rad/s)
        tilt_vel = angular.y  (oracle output, rad/s)
        Returns (0.0, 0.0) if no /cmd_vel has been received yet.
        """
        with self._lock:
            msg = self._latest_cmd_vel
        if msg is None:
            return 0.0, 0.0
        return float(msg.angular.z), float(msg.angular.y)


# ---------------------------------------------------------------------------
# Camera augmentation
# ---------------------------------------------------------------------------

def _apply_augmentation(
    frame: np.ndarray,
    config,
    aug_rng: np.random.Generator,
) -> np.ndarray:
    """Apply per-episode domain randomization to a uint8 RGB frame.

    Adds Gaussian noise (sigma from config.camera_noise_sigma) and a
    brightness offset (config.camera_brightness_offset pixels).  Uses a
    per-episode numpy.random.Generator seeded from config.seed so the
    result is fully reproducible without touching global RNG state.

    Returns a uint8 array clamped to [0, 255].
    """
    f = frame.astype(np.float32)
    if config.camera_noise_sigma > 0:
        f += aug_rng.normal(0.0, config.camera_noise_sigma * 255.0, f.shape)
    f += config.camera_brightness_offset
    return np.clip(f, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# HDF5 writing
# ---------------------------------------------------------------------------

def _write_hdf5(
    ep_idx: int,
    config,
    frames: "list[np.ndarray]",
    pan_vels: "list[float]",
    tilt_vels: "list[float]",
    out_dir: Path,
) -> Path:
    """Write one episode to an HDF5 file.

    Schema:
        frames    (N, 224, 224, 3) uint8  — augmented RGB frames
        pan_vel   (N,)             float32 — /cmd_vel angular.z per frame
        tilt_vel  (N,)             float32 — /cmd_vel angular.y per frame
        cmd       scalar str               — config.language_label
        label_key scalar str               — config.label_key
        metadata  scalar str               — JSON: config.to_dict()
    """
    episodes_dir = out_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    path = episodes_dir / f"ep_{ep_idx:06d}.h5"

    with h5py.File(path, "w") as f:
        f.create_dataset(
            "frames",
            data=np.stack(frames),
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset("pan_vel",  data=np.array(pan_vels,  dtype=np.float32))
        f.create_dataset("tilt_vel", data=np.array(tilt_vels, dtype=np.float32))
        f["cmd"]       = config.language_label
        f["label_key"] = config.label_key
        f["metadata"]  = json.dumps(config.to_dict())

    return path


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

def _write_splits(
    episode_ids: "list[int]",
    configs: list,
    out_dir: Path,
) -> None:
    """Write train.txt / val.txt / test.txt with an 80 / 10 / 10 split.

    Split is performed at scenario level: episodes from the same scenario_id
    are never spread across train and test, preventing label leakage.
    """
    groups: "dict[str, list[int]]" = defaultdict(list)
    for ep_id, cfg in zip(episode_ids, configs):
        groups[cfg.scenario_id].append(ep_id)

    scenario_ids = list(groups.keys())
    rng = random.Random(42)
    rng.shuffle(scenario_ids)

    n         = len(scenario_ids)
    train_ids = scenario_ids[:int(0.8 * n)]
    val_ids   = scenario_ids[int(0.8 * n):int(0.9 * n)]
    test_ids  = scenario_ids[int(0.9 * n):]

    for split_name, sids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        eps = [ep for sid in sids for ep in groups[sid]]
        (out_dir / f"{split_name}.txt").write_text(
            "\n".join(f"{e:06d}" for e in sorted(eps))
        )


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def run_collection(node: CollectNode, args) -> None:
    """Outer episode loop: sample → setup → warmup → record → teardown → write."""
    from sim.scenario_generator.gazebo_bridge import GazeboBridge
    from sim.scenario_generator.episode_runner import EpisodeRunner
    from sim.scenario_generator.scenario import ScenarioGenerator

    sim_dir         = Path(__file__).parent
    faces_dir       = sim_dir / "scenario_generator"
    backgrounds_dir = sim_dir / "assets" / "backgrounds"

    generator = ScenarioGenerator(faces_dir, backgrounds_dir)
    bridge    = GazeboBridge(world="scenario_world")
    runner    = EpisodeRunner(bridge)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_ids:  "list[int]"  = []
    configs:      list         = []
    label_counts: "dict[str, int]" = {}

    for ep_idx in range(args.base_ep, args.base_ep + args.n_episodes):
        seed   = args.base_seed + ep_idx
        config = generator.sample(seed)

        log.info("ep %06d: setup  seed=%d  label=%r", ep_idx, seed, config.label_key)

        # ── Setup ──────────────────────────────────────────────────────────
        # runner.setup() spawns Gazebo entities AND sets the oracle label key.
        try:
            runner.setup(config)
        except Exception as exc:
            log.warning("ep %06d: setup failed: %s — skipping", ep_idx, exc)
            continue

        # Per-episode numpy RNG for augmentation; seeded independently of the
        # scenario-generator stream so augmentation is reproducible.
        aug_rng = np.random.default_rng(config.seed)

        # ── Warmup ─────────────────────────────────────────────────────────
        # Drive motion patterns during warmup so the oracle starts from a
        # moving-face state rather than a static-spawn position.
        warmup_ok    = True
        warmup_start = time.monotonic()
        warmup_frame = 0

        while (time.monotonic() - warmup_start) < WARMUP_SECS:
            if not node.wait_for_new_image(timeout=5.0):
                log.error(
                    "ep %06d: no camera frame during warmup (timeout) — skipping episode",
                    ep_idx,
                )
                warmup_ok = False
                break
            runner.step(warmup_frame / CAMERA_HZ)
            warmup_frame += 1

        if not warmup_ok:
            runner.teardown()
            continue

        # ── Record ─────────────────────────────────────────────────────────
        frames:    "list[np.ndarray]" = []
        pan_vels:  "list[float]"      = []
        tilt_vels: "list[float]"      = []
        record_ok = True

        for frame_idx in range(FRAMES_PER_EPISODE):
            if not node.wait_for_new_image(timeout=5.0):
                log.warning("ep %06d: frame %d: image timeout", ep_idx, frame_idx)
                record_ok = False
                break

            raw_frame            = node.get_latest_frame_rgb()
            pan_vel, tilt_vel    = node.get_pan_tilt_vel()

            # Advance Gazebo entity positions to this simulation time.
            runner.step(WARMUP_SECS + frame_idx / CAMERA_HZ)

            if raw_frame is None:
                log.warning("ep %06d: frame %d: no image data", ep_idx, frame_idx)
                record_ok = False
                break

            frame = cv2.resize(raw_frame, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
            frame = _apply_augmentation(frame, config, aug_rng)

            frames.append(frame)
            pan_vels.append(pan_vel)
            tilt_vels.append(tilt_vel)

        runner.teardown()

        if not record_ok or len(frames) != FRAMES_PER_EPISODE:
            log.warning(
                "ep %06d: incomplete (%d/%d frames) — skipping write",
                ep_idx, len(frames), FRAMES_PER_EPISODE,
            )
            continue

        _write_hdf5(ep_idx, config, frames, pan_vels, tilt_vels, out_dir)
        episode_ids.append(ep_idx)
        configs.append(config)
        label_counts[config.label_key] = label_counts.get(config.label_key, 0) + 1
        log.info("ep %06d: %d frames  label=%r", ep_idx, len(frames), config.label_key)

    # ── Dataset-level files ────────────────────────────────────────────────
    if episode_ids:
        _write_splits(episode_ids, configs, out_dir)

    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).parent),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        git_hash = "unknown"

    metadata = {
        "schema_version":     "1.0",
        "collection_date":    datetime.now().isoformat(),
        "n_episodes":         len(episode_ids),
        "camera_hz":          CAMERA_HZ,
        "episode_secs":       EPISODE_SECS,
        "frames_per_episode": FRAMES_PER_EPISODE,
        "image_shape":        [224, 224, 3],
        "label_counts":       label_counts,
        "seed_range":         [args.base_seed, args.base_seed + args.n_episodes - 1],
        "git_hash":           git_hash,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    n_ep = len(episode_ids)
    log.info("Collection complete: %d episodes, %d frames", n_ep, n_ep * FRAMES_PER_EPISODE)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 Step 7 — data collection pipeline."
    )
    parser.add_argument(
        "--n_episodes", type=int, required=True,
        help="Number of episodes to collect.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for dataset files.",
    )
    parser.add_argument(
        "--base_seed", type=int, default=0,
        help="Seed offset: episode i uses seed base_seed + i.  Default: 0.",
    )
    parser.add_argument(
        "--base_ep", type=int, default=0,
        help="Episode index offset (for resuming or sharding).  Default: 0.",
    )
    args = parser.parse_args()

    rclpy.init()
    node = CollectNode()

    # Spin rclpy in a background daemon thread so callbacks fire while the
    # main thread runs the synchronous collection loop.
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        run_collection(node, args)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
