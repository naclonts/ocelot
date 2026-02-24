# Phase 2 — Simulation and Synthetic Data Engine

**Goal**: Build a Gazebo Harmonic simulation environment, implement a privileged oracle policy, and generate a large versioned synthetic dataset for VLA training — with zero human labeling effort.

**When to run**: Phase 1 is complete when the classical tracker reliably follows a face on real hardware and rosbag recording works. Phase 2 starts there.

**Where to run**: Simulation and data generation run on a **development machine** (Debian Bookworm x86_64, Docker-based). ROS 2 Jazzy and Gazebo Harmonic run inside a container — no native ROS packages on the host are required. The Pi remains the deployment target only.

---

## Step 0 — Repository Housekeeping: Move Dockerfiles

Before adding sim infrastructure, move the deployment files out of the root to reduce clutter.

```bash
mkdir -p deploy/docker
git mv Dockerfile deploy/docker/Dockerfile.robot
git mv docker-compose.yml deploy/docker/docker-compose.yml
git commit -m "chore: move Dockerfiles to deploy/docker/"
```

Update any path references:
- `docker-compose.yml`: the `build.context` should be `../../` (project root) and `dockerfile: deploy/docker/Dockerfile.robot`
- All `docker compose` commands work from the repo root: `docker compose -f deploy/docker/docker-compose.yml ...`
  - Or add a root-level `docker-compose.yml` that includes / extends it for convenience

When Phase 2 adds a sim image, it lands as `deploy/docker/Dockerfile.sim` — no further restructuring needed.

**Validation**: `docker compose -f deploy/docker/docker-compose.yml build` succeeds, `docker compose -f deploy/docker/docker-compose.yml up` launches all nodes.

---

## Step 1 — Sim Container Setup

The dev machine is Debian Bookworm (x86_64) — no native ROS 2 Jazzy packages exist for Debian.
The entire sim stack runs inside a Docker container based on `osrf/ros:jazzy-simulation`, which
ships Gazebo Harmonic pre-installed. Gazebo's GUI is forwarded to the host via X11.

### 1a — Dockerfile.sim

`deploy/docker/Dockerfile.sim`:
- Base: `osrf/ros:jazzy-simulation` (Ubuntu 24.04 + ROS 2 Jazzy + Gazebo Harmonic)
- Add: `ros-jazzy-ros-gz`, `ros-jazzy-gz-ros2-control`, `ros-jazzy-ros2-control`,
  `ros-jazzy-ros2-controllers`, `ros-jazzy-joint-state-publisher`,
  `ros-jazzy-robot-state-publisher`, `ros-jazzy-rqt`, `ros-jazzy-rviz2`
- Add: `python3-pip`, then `pip install dvc h5py`
- Workspace at `/ws/src/ocelot` (bind-mounted from host)
- Source ROS + workspace overlay in `/etc/bash.bashrc`

### 1b — docker-compose.sim.yml

`deploy/docker/docker-compose.sim.yml` — separate from the Pi stack, runs on the dev machine:

```yaml
services:
  sim:
    build:
      context: ../../
      dockerfile: deploy/docker/Dockerfile.sim
    environment:
      - DISPLAY=${DISPLAY}
      - ROS_DOMAIN_ID=1          # separate domain from real-robot stack
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix  # X11 socket for GUI forwarding
      - ../../:/ws/src/ocelot           # live source mount
      - sim_build:/ws/build
      - sim_install:/ws/install
    devices:
      - /dev/dri:/dev/dri              # GPU/DRI passthrough for hardware rendering
    network_mode: host
    stdin_open: true
    tty: true

volumes:
  sim_build:
  sim_install:
```

No VNC, no sidecar — X11 forwarding is the right approach on Linux.
`ROS_DOMAIN_ID=1` keeps sim topics isolated from the real-robot stack.

### 1c — Host prerequisite (one-time, Debian host)

```bash
# Create a wildcard X auth cookie so the container's root user can authenticate
# with the host X server. The sed converts hostname-specific entries to FamilyWild
# (0xffff) so the cookie works regardless of the container's hostname.
touch /tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f /tmp/.docker.xauth nmerge -

# DVC on the host for dataset management (no ROS needed)
pip install dvc h5py
```

### Validation

```bash
# Build the sim image
docker compose -f deploy/docker/docker-compose.sim.yml build

# One-time: create the wildcard X auth cookie (re-run if session changes)
touch /tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f /tmp/.docker.xauth nmerge -

# Launch an interactive shell and start Gazebo — GUI window should appear on host
docker compose -f deploy/docker/docker-compose.sim.yml run --rm sim bash
# inside container:
source /opt/ros/jazzy/setup.bash
# Launch Gazebo server + clock bridge (gz_sim.launch.py alone does not bridge /clock)
ros2 launch ros_gz_sim gz_sim.launch.py gz_args:='-r empty.sdf' &
ros2 run ros_gz_bridge parameter_bridge /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock &
sleep 5
ros2 topic list | grep /clock
```

**Success gate**: Gazebo window appears on the host display, `ros2 topic echo /clock` produces
output inside the container, no errors.

---

## Step 2 — Extend URDF for Simulation

The existing `urdf/pan_tilt.urdf` has geometry but lacks the physics and hardware interface elements that Gazebo and ros2_control require. Add:

1. **`<inertial>` blocks** to each non-fixed link — Gazebo ignores links without inertia.
2. **`<collision>` geometry** — copy the existing `<visual>` geometry boxes; exact shapes don't matter for this sim.
3. **`<ros2_control>` hardware interface block** — declares the two joints as velocity-commanded, using `gz_ros2_control/GazeboSimSystem` as the hardware plugin.
4. **Camera sensor** on `camera_link` — Gazebo Harmonic `<sensor type="camera">` plugin; bridge to ROS `/camera/image_raw`.

Approximate structure for the ros2_control block:
```xml
<ros2_control name="ocelot_gz" type="system">
  <hardware>
    <plugin>gz_ros2_control/GazeboSimSystem</plugin>
  </hardware>
  <joint name="pan_joint">
    <command_interface name="velocity"/>
    <state_interface name="position"/>
    <state_interface name="velocity"/>
  </joint>
  <joint name="tilt_joint">
    <command_interface name="velocity"/>
    <state_interface name="position"/>
    <state_interface name="velocity"/>
  </joint>
</ros2_control>
```

**Files to create/modify**:
- `urdf/pan_tilt.urdf` — extend in-place (keep backward compatible, sim-only tags are ignored by real hardware stack)

**Validation**:
```bash
# Parse check
check_urdf urdf/pan_tilt.urdf

# Loads in RViz without errors
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:="$(cat urdf/pan_tilt.urdf)"
```

**Success gate**: `check_urdf` passes, RViz shows the two-joint chain with camera.

---

## Step 3 — Simulation World and Launch Infrastructure

Create a `sim/` directory for all simulation artifacts.

### 3a — Gazebo World File

`sim/worlds/tracker_world.sdf` — the simulation scene:
- Ground plane + sky
- One directional sun light (direction and intensity configurable at runtime)
- Placeholder for face billboard models (spawned/despawned programmatically)
- Ambient light at moderate intensity

### 3b — Face Billboard Model

`sim/models/face_billboard/` — a flat textured plane (0.2 × 0.2 m) that displays a face texture. Create several texture variants (frontal, 3/4, left profile) to force the model to generalize. Free sources: Chicago Face Database, VGGFace2 sample images.

Model structure:
```
sim/models/face_billboard/
  model.config
  model.sdf          # flat plane with material/texture
  materials/textures/
    face_01.png
    face_02.png
    ...
```

### 3c — Simulation Launch File

`launch/sim_launch.py` — launches:
1. Gazebo Harmonic with `tracker_world.sdf`
2. `robot_state_publisher` with the extended URDF
3. `gz_ros2_control` (spawns the hardware interface)
4. `joint_state_broadcaster` controller
5. `velocity_controllers/JointGroupVelocityController` on `[pan_joint, tilt_joint]`
6. `ros_gz_bridge` nodes: clock, camera image, joint states, model pose

```bash
# Quick start
ros2 launch ocelot sim_launch.py

# Verify topics exist
ros2 topic list | grep -E "/camera/image_raw|/joint_states|/cmd_vel"
```

**Success gate**: `/camera/image_raw` publishes at ~15 Hz in simulation, sending velocity commands to `/cmd_vel` moves the joints visibly in Gazebo.

---

## Step 4 — Sim-Real Parity Check

Run the existing `tracker_node` in simulation without modifying any parameters. The goal is to confirm sim and real hardware exhibit qualitatively identical behavior before generating training data on top of this sim.

```bash
# Launch sim, then enable the tracker
ros2 launch ocelot sim_launch.py
ros2 param set /tracker_node enabled true
```

Manually compare:
- Does the camera follow the face billboard when it moves?
- Does the velocity profile (magnitude, smoothness) look similar to real hardware behavior?
- Is the face detection rate comparable (Haar cascade on simulated face textures may be lower — acceptable)?

**Known difference to tolerate**: Haar cascade detection rate on rendered faces will be lower than on real faces. This is fine — Phase 3 replaces Haar cascade with the trained VLA model. What matters is that `cmd_vel` has the right sign and magnitude when detection fires.

**Success gate**: Tracker follows a slowly-moving face billboard. Gains don't need re-tuning. If sign is reversed, document it (camera pose in sim vs real may differ).

---

## Step 5 — Oracle Policy Node

The oracle is the core of Phase 2. It has **privileged access** to the simulator — it reads the ground-truth 3D position of the face billboard directly from Gazebo without running any face detector.

**File**: `ocelot/oracle_node.py`

**Inputs**:
- `/joint_states` — current pan and tilt angles
- `/model/face_0/pose` (via ros_gz_bridge) — face billboard 3D position in world frame

**Algorithm**:
1. Transform face world position → camera frame using TF (pan + tilt joint transforms)
2. Compute azimuth angle to face (pan error) and elevation angle (tilt error)
3. `pan_vel = kp * pan_error_rad`, `tilt_vel = kp * tilt_error_rad`
4. Clamp to `max_velocity`, apply deadband
5. Publish to `/cmd_vel`

This is exact inverse kinematics for a 2-DOF serial mechanism — closed-form, no iteration needed.

**Validation**:
- Static face at various positions: oracle achieves < 2 px steady-state error in < 2 seconds
- Moving face (linear drift): oracle tracks with low lag

**Success gate**: Oracle achieves < 5 px mean tracking error on a randomly moving face, measured over 60-second runs. This is the teacher performance ceiling that the VLA student will approximate.

---

## Step 6 — Scenario Generator and Domain Randomization

**Dir**: `sim/scenario_generator/` — a Python class that generates randomized scenario configurations and applies them to a running Gazebo instance via service calls.

### Randomization parameters

| Category | Parameters | Range |
|---|---|---|
| Faces | count | 1–3 |
| Faces | texture per face | uniform sample from texture library, labeled ("man wearing pirate hat", "woman with long hair") |
| Faces | initial position | x: [1.0–3.0 m], y: [−1.0–1.0 m], z: [0.5–1.5 m] |
| Motion | pattern | static, linear_drift, sinusoidal, random_walk |
| Motion | speed | [0.05–0.5 m/s] |
| Lighting | sun direction | azimuth [0–360°], elevation [15–75°] |
| Lighting | ambient intensity | [0.2–0.8] |
| Background | wall color | random RGB in [0.3–0.9] each channel |
| Background | wall texture | uniform sample from background textures library |
| Camera | Gaussian noise σ | [0.0–0.015] |
| Camera | brightness offset | [−20–+20] pixel value |

### Language label generation

Labels are generated **deterministically** from scenario parameters — no human annotation:

| Condition | Label |
|---|---|
| 1 face, centered, any speed | `"track the face"` |
| 1 face, commanded slow | `"follow slowly"` |
| 1 face, face left of center | `"track the face on the left"` |
| 1 face, face right of center | `"track the face on the right"` |
| 2+ faces, target is leftmost | `"follow the person on the left"` |
| 2+ faces, target is rightmost | `"follow the person on the right"` |
| 2+ faces, target is wearing hat and other isn't | `"follow the person in the hat"` |
| 2+ faces, target is largest | `"track the closest person"` |

Start with 4–6 label types. Add more in Phase 3 once the model trains.

### Implementation notes

- Use `gz service -s /world/default/set_pose` to teleport face billboards
- Use Gazebo's `gz topic -t /world/default/light_config` to adjust lighting (or edit SDF and reload)
- Spawn/despawn additional faces via `gz service -s /world/default/create` and `destroy`
- Each scenario is a dataclass — serializable to JSON for reproducibility

---

## Step 7 — Data Collection Pipeline

**File**: `sim/collect_data.py` — the main collection orchestrator.

### Collection loop

```
for each scenario in scenario_stream:
    1. Apply scenario config to Gazebo (pose, lighting, texture, noise)
    2. Reset joints to center
    3. Run oracle for T seconds (e.g. 10 s at 15 Hz = 150 frames per episode)
    4. Record synchronized tuples: (frame_rgb, language_cmd, pan_vel, tilt_vel)
    5. Write episode to HDF5
```

### Output format

```
dataset/
  episodes/
    ep_000000.h5      # keys: frames (N,224,224,3) uint8
                      #       cmd (string)
                      #       pan_vel (N,) float32
                      #       tilt_vel (N,) float32
                      #       metadata (JSON string: scenario config)
    ep_000001.h5
    ...
  metadata.json       # dataset-level stats, schema version, collection date
  train.txt           # episode IDs for training split
  val.txt
  test.txt
```

**Why HDF5**: Random-access reads for DataLoader shuffling, compressed storage, easy inspection with `h5py` or `HDFView`.

**Frame resolution**: 224×224 RGB — matches DINOv2 input directly, keeps file sizes small.

### Initial collection target: 10k episodes

Validate the pipeline is correct before scaling. Spot-check 50 episodes manually.

---

## Step 8 — Dataset Versioning with DVC

```bash
# One-time setup
dvc init
dvc remote add -d local_store /data/ocelot_datasets   # local path for now

# Track dataset
dvc add dataset/
git add dataset.dvc .gitignore
git commit -m "add phase2 synthetic dataset v0.1"
dvc push
```

Add a `Makefile` or `scripts/collect.sh` that runs the full collection pipeline from scratch:
```bash
make dataset       # runs collect_data.py, then dvc add + dvc push
make dataset-check # runs quality checks and prints summary stats
```

Every dataset version is reproducible: `git checkout <hash> && dvc pull`.

---

## Step 9 — Scale to Production Volume

Target: **50,000–100,000 episodes**. At 10 s × 15 Hz = 150 frames per episode, this is 7.5M–15M frames.

### Headless rendering

All of the following runs inside the sim container (no host X server needed for batch runs):

```bash
# Gazebo Harmonic headless — pass GZ_HEADLESS or the launch arg
docker compose -f deploy/docker/docker-compose.sim.yml run --rm sim \
  bash -c "source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash && \
           ros2 launch ocelot sim_launch.py headless:=true"
# or set env var inside container: GZ_HEADLESS=1
```

For overnight batch runs, omit the `DISPLAY` env var and `/tmp/.X11-unix` volume — the container
works fully headless without them.

### Parallelization

Run N independent collection processes, each with its own Gazebo instance on a different `GZ_PARTITION` or port. Merge shards afterward.

```bash
# Example: 4 parallel collectors, each collecting 25k episodes
for i in 0 1 2 3; do
  GZ_PARTITION=$i python3 sim/collect_data.py --shard $i --n_episodes 25000 &
done
wait
python3 sim/merge_shards.py
```

A single modern desktop can produce ~50k episodes overnight (8 hours).

### Quality checks

`sim/check_dataset.py` — run before finalizing:
- **Velocity distribution**: histogram of pan_vel and tilt_vel — should be roughly uniform across [−max_vel, max_vel], not all zeros
- **Face position coverage**: 2D histogram of face center pixel positions — should cover the full image, not just center
- **Label balance**: count per label type — flag if any label is < 5% of dataset
- **Frame diversity**: sample 100 random frames per label type, visually confirm variation in lighting, background, face texture
- **No duplicate scenarios**: assert episode metadata hashes are unique

### Train/val/test split

Split at **scenario level**, not frame level. Episodes from the same scenario config must not appear in both train and test.

```python
# 80 / 10 / 10 split
# Group by scenario_id (hash of scenario config minus random seed)
# Assign groups to splits, then write episode IDs to train.txt / val.txt / test.txt
```

---

## Step 10 — Validation Gate and Documentation

### Unit Tests

Write `pytest` tests alongside each new component — not retroactively at the end. The data pipeline is pure Python (no ROS, no hardware), so tests run on any machine including CI.

Minimum coverage:
- `test_scenario_generator.py` — label generation is deterministic from params; assert correct label for known inputs; assert randomization stays within declared bounds
- `test_oracle_ik.py` — given known face position + joint angles, assert expected pan/tilt velocities (pure math)
- `test_dataset_checks.py` — feed `check_dataset.py` a known-bad synthetic dataset; assert each check flags the right failure

Place under `tests/sim/`. A bug in `scenario_generator.py` can silently corrupt 50k episodes — this is where testing pays off.

---

Before calling Phase 2 complete:

- [ ] Oracle achieves < 5 px mean tracking error across 10 diverse scenarios
- [ ] Dataset contains ≥ 50k episodes
- [ ] All 4–6 label types are represented at ≥ 5% each
- [ ] Face positions cover at least 80% of image area (measured by 2D histogram)
- [ ] Velocity histograms show non-trivial variance (std > 0.3)
- [ ] 50 random spot-checked frames look visually diverse
- [ ] `dvc pull && python sim/collect_data.py --dry-run` reproduces dataset stats
- [ ] README updated with dataset section (stats, label distribution, sample frames)

---

## Key Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Haar cascade fails on rendered faces (sim-real face appearance gap) | Oracle doesn't use Haar at all — this only affects the parity check in Step 4 |
| Gazebo rendering too slow for 100k episodes | Headless mode + parallel instances + overnight batch runs |
| Oracle tracking error too high (TF transform misconfigured) | Validate with static face at known position; check TF frame orientation |
| Label distribution imbalanced | Explicitly oversample underrepresented labels in scenario generator |
| DINOv2 features don't transfer from rendered→real faces (sim-to-real gap) | Aggressive domain randomization in Step 6 is the primary defense; LoRA fine-tune in Phase 4 if needed |

---

## Directory Layout After Phase 2

```
ocelot/
├── ocelot/
│   ├── oracle_node.py          # NEW — privileged oracle policy
│   └── ... (Phase 1 nodes unchanged)
├── sim/
│   ├── scenario_generator.py   # NEW — procedural scene config
│   ├── collect_data.py         # NEW — collection orchestrator
│   ├── check_dataset.py        # NEW — quality checks
│   ├── merge_shards.py         # NEW — shard merger
│   ├── worlds/
│   │   └── tracker_world.sdf   # NEW — Gazebo world
│   └── models/
│       └── face_billboard/     # NEW — face plane models + textures
├── launch/
│   ├── tracker_launch.py       # unchanged
│   └── sim_launch.py           # NEW — simulation launch
├── deploy/
│   └── docker/
│       ├── Dockerfile.robot        # MOVED from root (was Dockerfile)
│       ├── Dockerfile.sim          # NEW — Gazebo sim image (dev machine)
│       ├── docker-compose.yml      # MOVED from root (Pi robot stack)
│       └── docker-compose.sim.yml  # NEW — sim stack with X11 forwarding
├── dataset/                    # DVC-tracked, gitignored
│   ├── episodes/
│   ├── metadata.json
│   ├── train.txt / val.txt / test.txt
│   └── dataset.dvc             # git-tracked DVC pointer
├── tests/
│   └── sim/
│       ├── test_scenario_generator.py  # NEW
│       ├── test_oracle_ik.py           # NEW
│       └── test_dataset_checks.py      # NEW
└── urdf/
    └── pan_tilt.urdf           # extended with inertial, collision, ros2_control
```

---

## Phase 3 Handoff Criteria

Phase 3 (VLA model training) can begin when:
1. Oracle achieves < 5 px mean tracking error in sim
2. Dataset has ≥ 50k episodes with clean train/val/test splits
3. DVC tracks the dataset and `dvc pull` works from a clean checkout
4. `check_dataset.py` passes all quality gates
