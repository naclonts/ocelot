# Phase 2 Step 7 — Data Collection Pipeline

**Goal**: Implement `sim/collect_data.py`, the main orchestrator that drives sequential simulation
episodes, captures synchronized (frame, label, velocity) tuples from the running oracle, and writes
compressed HDF5 episode files that become the VLA training dataset.

---

## Prerequisites

All of the following must be working before Step 7 begins:

- **Step 5 (Oracle)**: `oracle_node` achieves < 5 px mean tracking error. The physics crash
  (DART `dLDLTRemove` assertion) must be resolved. The fix — `max_velocity=1.0 rad/s` in
  `oracle_node.py` — was applied; confirm it holds across 10-minute runs before proceeding.
- **Step 6 (Scenario generator + episode runner)**: `pytest tests/sim/ -v` passes all 18 tests.
  `run_one_episode.py --duration 5` completes cleanly for seeds 0–9 (entity leak check).
- **Assets**: 110 face PNGs in `sim/assets/faces/`, 23 backgrounds in `sim/assets/backgrounds/`.

---

## Architecture

Two processes run concurrently, both inside the sim container:

```
Process 1: ros2 launch ocelot sim_launch.py world:=scenario_world use_oracle:=true headless:=true
    ├── Gazebo Harmonic (scenario_world — minimal, empty on startup)
    ├── robot_state_publisher + gz_ros2_control + joint_state_broadcaster
    ├── joint_group_velocity_controller (velocity commands on [pan_joint, tilt_joint])
    ├── ros_gz_bridge (/clock, /camera/image_raw, /model/face_0/pose)
    ├── oracle_node (reads /model/face_0/pose + /joint_states, publishes /cmd_vel)
    └── cmd_vel_adapter (/cmd_vel → /joint_group_velocity_controller/commands)

Process 2: python3 sim/collect_data.py --n_episodes 100 --output dataset/
    ├── rclpy.Node (subscribes to /camera/image_raw, /joint_states, /cmd_vel, /clock)
    ├── ScenarioGenerator (deterministic scenario configs from seeds)
    ├── EpisodeRunner + GazeboBridge (spawn/animate/despawn entities per episode)
    └── h5py (write one .h5 file per episode)
```

`collect_data.py` is the **driver**. It owns the episode lifecycle and writes all output.
The sim stack is a long-running server — start it once, then run collect_data.py against it.

---

## Step 7A — sim_launch.py: disable `move_face` for scenario_world

`move_face.py` oscillates the static `face_billboard` model in `tracker_world`. In
`scenario_world` there is no pre-placed face; entities are spawned per-episode by
`EpisodeRunner`. Launching `move_face.py` in this mode produces harmless-but-noisy
service-call errors in the log.

**Change**: In `launch/sim_launch.py`, make `move_face` conditional on the world name.

```python
# Only oscillate the face billboard in tracker_world (Step 4/5 testing).
# In scenario_world, collect_data.py drives all entity motion via EpisodeRunner.
if world_name == 'tracker_world':
    actions.append(TimerAction(
        period=15.0,
        actions=[ExecuteProcess(
            cmd=['python3', '/ws/src/ocelot/sim/move_face.py'],
            output='screen',
        )],
    ))
```

No other launch file changes are needed. The existing launch args cover data collection:
```bash
ros2 launch ocelot sim_launch.py world:=scenario_world use_oracle:=true headless:=true
```

---

## Step 7B — `sim/collect_data.py`

### File overview

```
sim/collect_data.py
├── CollectNode(rclpy.Node)          # ROS subscriptions + latest-value buffers
│   ├── _on_image(msg)               # stores latest sensor_msgs/Image
│   ├── _on_joint_states(msg)        # stores latest sensor_msgs/JointState
│   └── _on_cmd_vel(msg)             # stores latest geometry_msgs/Twist
├── run_collection(args)             # outer loop: episodes
│   ├── ScenarioGenerator.sample()
│   ├── EpisodeRunner.setup()
│   ├── _warmup()                    # spin until oracle converges or timeout
│   ├── _record_episode()            # inner loop: camera-driven frame capture
│   ├── EpisodeRunner.teardown()
│   └── _write_hdf5()
└── main()                           # argparse + rclpy init + run_collection
```

### Timing: camera-driven loop

The camera publishes at **10 Hz** (URDF `<update_rate>10</update_rate>`). The collection loop
is driven by camera frame arrival:

- Oracle publishes `/cmd_vel` at 20 Hz.
- Joint states publish at 50 Hz (controller rate).
- On each new camera frame, `collect_data.py` samples the latest values from the other two
  topics (simple latest-value buffers — no message_filters needed).

Frame count per episode: `EPISODE_SECS × CAMERA_HZ = 10 × 10 = 100 frames`.

The `runner.step(t)` call inside the recording loop advances motion patterns to `t = frame_idx /
CAMERA_HZ`. This is independent of wall-clock time: if Gazebo runs at 0.5× real-time (common
under software rendering), the motion patterns slow to match.

### Episode flow (pseudocode)

```python
CAMERA_HZ    = 10      # Hz — must match URDF update_rate
EPISODE_SECS = 10.0    # seconds of recorded data per episode
WARMUP_SECS  = 4.0     # seconds before recording starts (oracle convergence)
IMG_SIZE     = (224, 224)

for ep_idx in range(base_ep, base_ep + n_episodes):
    seed   = base_seed + ep_idx
    config = generator.sample(seed)

    runner.setup(config)              # spawns entities in Gazebo
    _set_oracle_label_key(config)     # already done inside runner.setup()

    # ── Warmup ────────────────────────────────────────────────────────────
    # Wait for entities to load and oracle to converge. Drive motion forward
    # during warmup so the oracle starts from a moving-face state, not from
    # a static-spawn position.
    t_warmup_start = node.get_clock().now()
    warmup_frame = 0
    while elapsed_sec(t_warmup_start) < WARMUP_SECS:
        wait_for_new_image()          # blocks until /camera/image_raw fires
        runner.step(warmup_frame / CAMERA_HZ)
        warmup_frame += 1

    # ── Record ────────────────────────────────────────────────────────────
    frames, pan_vels, tilt_vels = [], [], []

    for frame_idx in range(int(EPISODE_SECS * CAMERA_HZ)):
        wait_for_new_image()
        img = current_image           # latest sensor_msgs/Image (640×480 RGB8)

        pan_vel  = latest_cmd_vel.angular.z
        tilt_vel = latest_cmd_vel.angular.y

        # Advance Gazebo entity positions to this simulation time.
        t_episode = frame_idx / CAMERA_HZ
        runner.step(WARMUP_SECS + t_episode)

        # Convert and augment frame.
        frame = bridge.imgmsg_to_cv2(img, "rgb8")              # (480,640,3) uint8
        frame = cv2.resize(frame, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        frame = _apply_augmentation(frame, config)             # noise + brightness

        frames.append(frame)
        pan_vels.append(pan_vel)
        tilt_vels.append(tilt_vel)

    runner.teardown()
    _write_hdf5(ep_idx, config, frames, pan_vels, tilt_vels, out_dir)
    log.info(f"ep {ep_idx:06d}: {len(frames)} frames  label={config.label_key!r}")
```

### Camera augmentation (applied in software)

`ScenarioConfig` carries `camera_noise_sigma` and `camera_brightness_offset`. Apply these to
the captured frame **before** writing to HDF5 — this is the mechanism for domain randomization
at the pixel level (Gazebo render → software augment → stored).

```python
def _apply_augmentation(frame: np.ndarray, config: ScenarioConfig) -> np.ndarray:
    """Apply noise and brightness offset. Returns uint8 frame."""
    f = frame.astype(np.float32)
    if config.camera_noise_sigma > 0:
        f += np.random.normal(0, config.camera_noise_sigma * 255, f.shape)
    f += config.camera_brightness_offset
    return np.clip(f, 0, 255).astype(np.uint8)
```

Note: this must use a **separate** `numpy.random` instance seeded from `config.seed` (not the
global `random` module), so augmentation is reproducible per episode.

### HDF5 schema (per episode)

File: `dataset/episodes/ep_NNNNNN.h5`

| Dataset   | Shape        | dtype   | Description                             |
|-----------|--------------|---------|----------------------------------------|
| `frames`  | (N,224,224,3)| uint8   | RGB frames, augmented                  |
| `pan_vel` | (N,)         | float32 | `/cmd_vel` angular.z (oracle output)  |
| `tilt_vel`| (N,)         | float32 | `/cmd_vel` angular.y (oracle output)  |
| `cmd`     | scalar       | str     | `config.language_label`                |
| `label_key`| scalar      | str     | `config.label_key` (for filtering)    |
| `metadata`| scalar       | str     | JSON: `config.to_dict()`               |

Create with compression:
```python
with h5py.File(path, "w") as f:
    f.create_dataset("frames",   data=np.stack(frames),   compression="gzip", compression_opts=4)
    f.create_dataset("pan_vel",  data=np.array(pan_vels, dtype=np.float32))
    f.create_dataset("tilt_vel", data=np.array(tilt_vels, dtype=np.float32))
    f["cmd"]       = config.language_label
    f["label_key"] = config.label_key
    f["metadata"]  = json.dumps(config.to_dict())
```

### Dataset-level files

Written once after the collection loop completes:

**`dataset/metadata.json`**:
```json
{
  "schema_version": "1.0",
  "collection_date": "...",
  "n_episodes": 10000,
  "camera_hz": 10,
  "episode_secs": 10,
  "frames_per_episode": 100,
  "image_shape": [224, 224, 3],
  "label_counts": {"track": 6000, "multi_attr": 1200, ...},
  "seed_range": [0, 9999],
  "git_hash": "..."
}
```

**`train.txt` / `val.txt` / `test.txt`**: 80/10/10 split, assigned at **scenario** level (by
`scenario_id` hash, not episode index). Episodes from the same scenario config never span
train and test. Write episode IDs (zero-padded 6-digit strings), one per line.

```python
def _write_splits(episode_ids: list[int], configs: list[ScenarioConfig], out_dir: Path):
    # Group by scenario_id to prevent leakage.
    from collections import defaultdict
    groups = defaultdict(list)
    for ep_id, cfg in zip(episode_ids, configs):
        groups[cfg.scenario_id].append(ep_id)
    scenario_ids = list(groups.keys())
    rng = random.Random(42)
    rng.shuffle(scenario_ids)
    n = len(scenario_ids)
    train_ids = scenario_ids[:int(0.8 * n)]
    val_ids   = scenario_ids[int(0.8 * n):int(0.9 * n)]
    test_ids  = scenario_ids[int(0.9 * n):]
    for split_name, sids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        eps = [ep for sid in sids for ep in groups[sid]]
        (out_dir / f"{split_name}.txt").write_text("\n".join(f"{e:06d}" for e in sorted(eps)))
```

### CLI

```bash
python3 sim/collect_data.py \
    --n_episodes 100 \
    --output /ws/src/ocelot/dataset \
    --base_seed 0 \
    [--base_ep 0]          # episode index offset (for resuming or sharding)
```

For parallel sharding (Step 9), run N processes each with a non-overlapping seed+ep range:
```bash
# Shard 0: episodes 0..9999, seeds 0..9999
python3 sim/collect_data.py --n_episodes 10000 --base_seed 0    --base_ep 0     --output dataset/shard_0 &
# Shard 1: episodes 0..9999, seeds 10000..19999
python3 sim/collect_data.py --n_episodes 10000 --base_seed 10000 --base_ep 10000 --output dataset/shard_1 &
```

---

## Step 7C — `sim/check_dataset.py`

Minimal quality check script. Run against a small batch (e.g. 50 episodes) before scaling.

```
Checks:
  [OK] 50 episode files found
  [OK] All files readable; min/max frame counts: 100/100
  Label distribution:
      track        31  (62.0%)
      multi_attr    8  (16.0%)
      multi_left    5  (10.0%)
      multi_right   4   (8.0%)
      multi_closest 2   (4.0%)
  pan_vel  — mean: -0.012  std: 0.312  min: -0.987  max: 0.991
  tilt_vel — mean:  0.008  std: 0.201  min: -0.623  max: 0.607
  [WARN] multi_closest at 4.0% — below 5% threshold
  [OK] No duplicate scenario_ids
  [OK] All frames shape (224, 224, 3) uint8
```

Success criteria for this check:
- No corrupt/missing files
- All episodes have exactly `EPISODE_SECS × CAMERA_HZ` frames
- `pan_vel` std > 0.1 (oracle is actually commanding motion, not idling)
- No label at 0% (all label types represented)

---

## Implementation Order

1. **Modify `sim_launch.py`** (7A) — 5 min, trivial change, validate with a headless run.
2. **Write `sim/collect_data.py`** (7B) — main work.
   - Start with a minimal version: one hardcoded episode, print frame count, no HDF5.
   - Add HDF5 writing once the capture loop is confirmed working.
   - Add augmentation.
   - Add metadata.json + splits.
   - Add argparse.
3. **Write `sim/check_dataset.py`** (7C) — write after first 10 test episodes are captured.

---

## Validation Workflow

### Step 1: Confirm sim stack is stable

```bash
# Terminal 1 — start sim stack
docker compose -f deploy/docker/docker-compose.sim.yml run --rm --name ocelot-sim sim bash -c "
  source /opt/ros/jazzy/setup.bash && cd /ws &&
  colcon build --symlink-install --packages-select ocelot --event-handlers console_direct- &&
  source /ws/install/setup.bash &&
  ros2 launch ocelot sim_launch.py world:=scenario_world use_oracle:=true headless:=true
"
```

Wait 20s for Gazebo + controllers to initialize (watch for `Spawned [ocelot]` and
`Successfully loaded controller joint_group_velocity_controller`).

### Step 2: Run 5-episode smoke test

```bash
# Terminal 2
docker exec -e ROS_DOMAIN_ID=1 ocelot-sim bash -c "
  source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash &&
  python3 /ws/src/ocelot/sim/collect_data.py --n_episodes 5 --output /tmp/dataset_test --base_seed 42
"
```

Expected output:
```
ep 000000: 100 frames  label='track'
ep 000001: 100 frames  label='multi_attr'
ep 000002: 100 frames  label='track'
ep 000003: 100 frames  label='multi_left'
ep 000004: 100 frames  label='track'
Collection complete: 5 episodes, 500 frames
```

### Step 3: Inspect episode file

```python
# Inside container with h5py
import h5py, json
with h5py.File("/tmp/dataset_test/episodes/ep_000000.h5") as f:
    print(dict(f))
    print("frames:", f["frames"].shape, f["frames"].dtype)
    print("pan_vel:", f["pan_vel"][:5])
    print("cmd:", f["cmd"][()])
    cfg = json.loads(f["metadata"][()])
    print("label_key:", cfg["label_key"])
```

Expected: `frames (100, 224, 224, 3) uint8`, `pan_vel` has nonzero values, `cmd` matches `label_key`.

### Step 4: Run quality check

```bash
python3 /ws/src/ocelot/sim/check_dataset.py --dataset /tmp/dataset_test
```

### Step 5: 100-episode run

Before scaling to 10k, collect 100 episodes, check the dataset, spot-check 10 random frames
visually (dump to PNG and inspect with an image viewer):

```bash
python3 /ws/src/ocelot/sim/collect_data.py --n_episodes 100 --output /tmp/dataset_100 --base_seed 0
python3 /ws/src/ocelot/sim/check_dataset.py --dataset /tmp/dataset_100 --sample_frames 10
```

---

## Known Issues and Mitigations

| Issue | Mitigation |
|---|---|
| Gazebo physics crash (DART) under high velocity commands | `max_velocity=1.0 rad/s` in oracle_node.py matches URDF limit. If crash recurs, add a rate-limiter in cmd_vel_adapter. |
| Camera frame drops (Gazebo software-render backlog) | Detect frame gaps via header.stamp and log a warning. Don't skip — use the most recent frame. |
| Oracle hasn't converged by end of warmup | Increase `WARMUP_SECS` from 4.0 to 6.0. Check velocity std in check_dataset — low std means oracle is still centering during recording. |
| `episode_runner.py` entity spawn timeout | `GazeboBridge.setup_episode()` returns False on timeout. Detect this in `run_collection()` and skip to the next episode (log a warning, don't crash). |
| Warmup frame `wait_for_new_image()` blocks indefinitely | Add a 5-second timeout; if no camera frame arrives, restart the sim stack (the bridge may have died). |
| ROS domain mismatch | All `docker exec` commands must pass `-e ROS_DOMAIN_ID=1`. collect_data.py reads `ROS_DOMAIN_ID` from environment — set it in the shell before running. |

---

## Success Gate

Step 7 is complete when:

- [ ] `sim/collect_data.py` collects 100 episodes without crashing or hanging
- [ ] HDF5 files are well-formed: correct shapes, nonzero pan/tilt variance, readable cmd strings
- [ ] `check_dataset.py` passes on the 100-episode sample with no FAIL flags
- [ ] Label distribution has no label at 0% (all label types appear in 100 episodes; may need more episodes for rare multi-face types)
- [ ] Train/val/test split files are written with correct scenario-level grouping

---

## File Summary

| File | Status | Notes |
|---|---|---|
| `launch/sim_launch.py` | **modify** | Condition `move_face` on `world_name == 'tracker_world'` |
| `sim/collect_data.py` | **create** | Main orchestrator — ~300 lines |
| `sim/check_dataset.py` | **create** | Quality check CLI — ~150 lines |

No other files need modification. The existing `EpisodeRunner`, `GazeboBridge`,
`ScenarioGenerator`, and `oracle_node` are used as-is.
