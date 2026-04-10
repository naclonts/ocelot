# Ocelot

Pan-tilt face tracking robot (Raspberry Pi 5), plus a simulated environment for training. Classical CV baseline with Haar cascade for now; VLA model coming soon.

## Hardware
| Component | Detail |
|---|---|
| Compute | Raspberry Pi 5 |
| Camera | Pi Camera V2 (CSI) |
| Servo driver | PCA9685 at I2C 0x40 |
| Servos | SG90 — pan ch0 (0–180°), tilt ch1 (90–180°) |
| OS | Pi OS Bookworm (Python 3.11) |

## Quickstart

### First time

```bash
# 1. Build Docker image
docker compose build

# 2. Launch
docker compose up
```

### Launch Options

```bash
docker compose up                          # VLA model (INT8) — default
VISUALIZE=true docker compose up           # VLA + annotated stream
USE_HAAR=true docker compose up            # classical Haar cascade tracker
VLA_COMMAND="look at the person" docker compose up
USE_REMOTE_VLA=true REMOTE_VLA_URL=http://<workstation-ip>:8765/infer docker compose up
ROS_LOCALHOST_ONLY=1 docker compose up    # optional: keep ROS graph private to the Pi

# Change the active language command at runtime
ros2 param set /vla_node command "look at the person"
ros2 param set /remote_vla_client_node command "look at the person"

# Remote mode only: increase HTTP timeout if the first uncached command stalls
ros2 param set /remote_vla_client_node request_timeout_sec 1.0
```

`docker compose up` loads `models/active.onnx` — a symlink to the currently active INT8 model.
By default the Pi now exposes its ROS 2 graph on the LAN (`ROS_LOCALHOST_ONLY=0`), so a laptop
on the same network can inspect topics like `/camera/image_raw`. Set `ROS_LOCALHOST_ONLY=1` if
you want to isolate the ROS graph to the Pi again.

### Offboard inference and cross-machine ROS over wired LAN

By default the Pi exposes its ROS 2 graph on the LAN (`ROS_LOCALHOST_ONLY=0`), so a laptop on
the same network can inspect topics like `/camera/image_raw`. Set `ROS_LOCALHOST_ONLY=1` if you
want to isolate the ROS graph to the Pi again.

To inspect the Pi's ROS graph from your laptop:

```bash
# On the Pi
docker compose up

# On your laptop
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
ros2 topic list
ros2 topic echo /cmd_vel
```

To run the camera + servo stack on the Pi and the ONNX model on your workstation:

```bash
# On your workstation
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
python -m ocelot.remote_vla_server \
  --checkpoint models/active.onnx \
  --token-cache models/active_tokens.json \
  --host 0.0.0.0 \
  --port 8765

# On the Pi
USE_REMOTE_VLA=true \
REMOTE_VLA_URL=http://<workstation-ip>:8765/infer \
docker compose up
```

The Pi keeps `camera_node` and `servo_node` local, JPEG-encodes each frame, posts it to the
workstation, and applies the returned `(pan_vel, tilt_vel)` commands locally. `servo_node`
zeros velocities if `/cmd_vel` goes stale for 250 ms, so a lost workstation or cable does not
leave the head slewing on the last command.

### Switching models

To deploy a new trained model to the Pi, run `make use-model` from the project root. It
quantizes `best.onnx` to INT8 (skipped if `best_int8.onnx` already exists), then updates
the `models/active.onnx` symlink:

```bash
make use-model RUN=runs/v0.1.1-single-face
docker compose up
```

Quantization runs inside the robot container (~30 s, once per checkpoint). Always deploy INT8 — FP32 models are too slow for real-time tracking on Pi 5 CPU.

To override the model at launch time without changing the symlink:

```bash
VLA_CHECKPOINT=/ws/src/ocelot/runs/v0.1.1-single-face/best_int8.onnx docker compose up
```

Editing Python source files does **not** require a rebuild (symlinks are live). Rebuilding
is only needed when `setup.py` entry points change.

### View streams

| Stream | URL |
|---|---|
| Raw | `http://<pi-ip>:8080/stream?topic=/camera/image_raw` |
| Annotated | `http://<pi-ip>:8080/stream?topic=/camera/image_annotated` |

### Sim (dev machine)

```bash
make sim-build   # build the sim image (once, or after Dockerfile changes)
make sim         # headless — no GUI, fast, works on any machine
make sim-gui     # Gazebo GUI — software rendering (no GPU required)
make sim-gpu     # Gazebo GUI — GPU accelerated (requires NVIDIA runtime)
make sim-vla VLA_ONNX=runs/v0.1/best.onnx   # run trained VLA model in sim (GPU, requires NVIDIA runtime)
make sim-vla-eval  # eval VLA against N training-distribution scenarios (see below)
make sim-xauth   # one-time X11 auth setup (re-run if display session changes)
make sim-shell   # interactive shell in a fresh sim container
```

The colcon build is fast on repeat runs — named volumes (`sim_build`, `sim_install`) cache artifacts between container invocations.

After ~15 seconds the sim is fully up: the face billboard starts oscillating in both pan (Y) and tilt (Z), and the tracker follows it automatically. No manual steps needed.

Verify tracking is working from a second shell in the container:
```bash
ros2 topic echo /joint_states --field position   # pan/tilt positions should change
```

#### Episode runner (scenario generator)

The episode runner generates randomized scenarios — face textures, background, lighting, motion
patterns, language labels — and drives them in a live Gazebo session. Use it to smoke-test the
scenario generator before running full data collection.

**Prerequisites** (assets must exist before running):

```bash
# Face description JSONs (git-tracked — present after clone if committed)
ls sim/scenario_generator/face_descriptions*.json

# Face PNGs in sim/assets/faces/ and background PNGs in sim/assets/backgrounds/
# Pull from DVC if available:
dvc pull
# Or regenerate locally (backgrounds take seconds; faces require an AI image API):
make backgrounds                          # generates 6 plain-color PNGs; no API required
# make faces                             # generates descriptions + calls image API
```

**Run a single episode inside the sim container:**

```bash
make sim-shell   # open an interactive shell in a fresh sim container

# Inside the container — build, start sim in background, then run one episode
colcon build --symlink-install --packages-select ocelot --event-handlers console_direct-
source /ws/install/setup.bash
ros2 launch ocelot sim_launch.py world:=scenario_world headless:=true use_oracle:=true &
sleep 15   # wait for Gazebo + ros2_control to finish starting

python3 /ws/src/ocelot/sim/scenario_generator/run_one_episode.py --seed 42 --duration 10
```

Exit code 0 means the episode completed without error. The script prints the full scenario config,
face positions every second, and final positions at teardown.

**Run 10 sequential episodes (entity leak check):**

```bash
for i in $(seq 0 9); do
    python3 /ws/src/ocelot/sim/scenario_generator/run_one_episode.py --seed $i --duration 5
done
# After all episodes: verify no leaked entities
gz model --list   # should show only: ground_plane, ocelot
gz light --list   # should be empty
```

#### Data collection

`collect_parallel.sh` generates episodes, runs them, and saves output. Output always goes to
`/ws/src/ocelot/sim/dataset` (bind-mounted to `sim/dataset/` on the host).

```bash
bash sim/data_gen/collect_parallel.sh --shards 7 --episodes 700
```

The script auto-detects the next unused shard index from the output directory, so re-running
never overwrites existing data. Override with `--start-shard N` if needed.

After collection, verify a shard:

```bash
docker exec -e ROS_DOMAIN_ID=1 ocelot-sim-0 \
  python3 /ws/src/ocelot/sim/data_gen/check_dataset.py --dataset /ws/src/ocelot/sim/dataset/shard_0
```

Then merge all shards into one dataset **on the host** (not inside a container — merge only needs h5py from `.venv`):

```bash
source .venv/bin/activate
python3 sim/data_gen/merge_shards.py \
    --parent sim/dataset \
    --output sim/dataset/merged
```

`collect_parallel.sh` runs this automatically at the end of a full run. If containers were killed early, run it manually. The merger auto-discovers all `shard_N/` directories, deduplicates episode IDs across shards, regenerates train/val/test splits, and writes `sim/dataset/merged/`.

For zero-velocity supervision, `collect_data.py` can also sample `no_face` and `centered` episodes:

```bash
python3 sim/data_gen/collect_data.py \
    --n_episodes 1000 \
    --output sim/dataset \
    --no_face_rate 0.10 \
    --centered_rate 0.05
```

#### Training

Install training dependencies:

```bash
pip install -r requirements-train.txt
```

Pull dataset using DVC (>75 GB data) with `dvc pull`.

##### Sweep

Hyperparameter sweep over `lr` × `n_fusion_layers`:

```bash
SWEEP=sweep-v0.1   # increment to avoid overwriting previous sweep checkpoints
for lr in 1e-4 3e-4 1e-3; do
  for layers in 1 2 4; do
    python3 train/train.py \
        --dataset_dir sim/dataset/ \
        --output_dir  runs/$SWEEP/lr${lr}_l${layers}/ \
        --epochs 3 \
        --lr $lr \
        --n_fusion_layers $layers \
        --batch_size 64 \
        --max_episodes 1500 \
        --amp \
        --experiment ocelot-sweep
  done
done
```

Each combo writes a separate checkpoint under `runs/$SWEEP/` and a separate MLflow run under the `ocelot-sweep` experiment. View results sorted by `val_loss`:

```bash
mlflow ui    # http://localhost:5000 → experiment "ocelot-sweep"
```

##### Full train

Full training run with AMP (use best `lr`/`layers`/`bs` from sweep):

```bash
python3 train/train.py \
    --dataset_dir sim/dataset/ \
    --output_dir  runs/v0.1.0/ \
    --epochs 20 \
    --batch_size 64 \
    --num_workers 12 \
    --amp \
    --confidence_weight 1.0 \
    --experiment ocelot-v0.1.0
```

To enable training-only domain randomization, add `--domain_randomization` and
tune the per-transform probabilities and strengths in `train/train.py`.

##### Track-only train

Train on single-face tracking episodes only (`label_key=track`), filtering out
multi-face/attribute commands. Useful when the deployment only needs face tracking:

```bash
python3 train/train.py \
    --dataset_dir sim/dataset/ \
    --output_dir  runs/v0.2-track-only/ \
    --epochs 20 \
    --batch_size 64 \
    --num_workers 12 \
    --amp \
    --label_keys track \
    --experiment ocelot-v0.2-track-only
```

Inspect metrics:

```bash
mlflow ui    # open http://localhost:5000
```

`val_mse_<label_key>` columns show per-label breakdown (e.g. `basic_track`, `multi_left`).
A good model reaches RMSE < 0.015 rad/s per axis (< 10% of the typical oracle signal).

#### Evaluate a checkpoint

```bash
source .venv/bin/activate

# Text report (RMSE, Pearson r, sign agreement, per-label breakdown):
python3 train/eval.py \
    --checkpoint runs/v0.0-smoke/best.pt \
    --dataset_dir sim/dataset/

# With scatter plot + 4 episode time-series overlays:
python3 train/eval.py \
    --checkpoint runs/v0.0-smoke/best.pt \
    --dataset_dir sim/dataset/ \
    --plot --episodes 4
# → runs/v0.0-smoke/scatter.png, runs/v0.0-smoke/episodes.png
```

#### VLA sim validation

Run the trained model inside Gazebo in closed-loop: the model sees each live camera frame and its output drives the pan-tilt joints. The face billboard oscillates automatically so there is always something to track.

**Step 1 — Export to ONNX** (host, one-time per checkpoint):

```bash
source .venv/bin/activate
python3 train/export_onnx.py \
    --checkpoint runs/v0.0-smoke/best.pt \
    --output     runs/v0.0-smoke/best.onnx \
    --verify
# → best.onnx + best_tokens.json alongside the checkpoint
```

**Step 2 — Rebuild sim image** (needed once after the Dockerfile changed to `onnxruntime-gpu`):

```bash
make sim-build
```

**Step 3 — Launch sim with VLA node (CPU)**:

```bash
docker compose -f deploy/docker/docker-compose.sim.yml run --rm sim bash -c "
  source /opt/ros/jazzy/setup.bash && cd /ws &&
  colcon build --symlink-install --packages-select ocelot --event-handlers console_direct- &&
  source /ws/install/setup.bash &&
  ros2 launch ocelot sim_launch.py use_vla:=true headless:=true
"
```

**Step 3 (GPU) — Launch sim with VLA node on NVIDIA GPU** (requires NVIDIA container runtime):

```bash
# Run VLA in sim
make sim-vla VLA_ONNX=runs/sweep-v0.0.2-1500-ep/lr1e-4_l2/best.onnx

# Evaluate against N reproducible scenarios (optional: override seed and count)
make sim-vla-eval VLA_ONNX=runs/sweep-v0.0.2-1500-ep/lr1e-4_l2/best.onnx SCENARIO_SEED=0 N_SCENARIOS=5
```

The `vla_node` logs which ONNX provider it is using on startup:
```
ONNX session ready (provider: CUDAExecutionProvider)   # GPU
ONNX session ready (provider: CPUExecutionProvider)    # CPU fallback
```

The default checkpoint path inside the container is `/ws/src/ocelot/runs/sweep-v0.0.2-1500-ep/lr1e-4_l2/best.onnx`
(the `runs/` directory is bind-mounted from the host). Override checkpoint or command:

```bash
ros2 launch ocelot sim_launch.py use_vla:=true headless:=true \
    vla_checkpoint:=/ws/src/ocelot/runs/v0.1/best.onnx \
    vla_command:="track the face"
```

**Monitor** from a second shell in the same container:

```bash
# Joint positions should change as the face oscillates
ros2 topic echo /joint_states --field position

# VLA velocity commands
ros2 topic echo /cmd_vel
```

The `vla_node` logs `pan=+0.xxx  tilt=+0.xxx rad/s` per frame. If the joints track
the face motion, behavioral cloning is working. If output is near-zero or static,
the model needs more training data or epochs.

#### VLA live evaluation (training-distribution scenarios)

`sim-vla-eval` tests the VLA against N reproducible scenarios drawn from the same
distribution used for data collection — varied face textures, backgrounds, lighting,
motion patterns, and distractors. It measures FK angular error while the model drives
the robot and prints a pass/fail table.

```bash
# 5 scenarios from seed 0 (default)
make sim-vla-eval VLA_ONNX=runs/v0.1/best.onnx

# More scenarios, different seed range
make sim-vla-eval VLA_ONNX=runs/v0.1/best.onnx SCENARIO_SEED=50 N_SCENARIOS=10
```

The script waits up to 90 s for Gazebo and the VLA node to publish before starting —
no manual sleep needed. Each scenario runs a 4 s warmup (VLA convergence) then a
10 s measurement window. Output:

```
Scenario 1/5  seed=0  motion=sinusoidal  label=basic_track
  mean=3.2°  max=8.7°  n=100  [PASS]
...
--- Summary ---
  #    seed  motion            label           mean°   max°  pass
  1       0  sinusoidal        basic_track       3.2     8.7  Y
  ...
Overall: mean=4.1°  pass_rate=80%  (threshold=10.0°)
```

Pass threshold: mean FK angular error < 10°. A well-trained model should achieve
< 5° mean and > 80% pass rate.

---

## Rosbag

Bags are stored in `./bags/` (bind-mounted to `/ws/bags/` in the container).

### Record

With the stack running (`docker compose up`), open a second terminal and record:

```bash
docker compose exec ocelot bash -i -c "
  ros2 bag record \
    --storage mcap \
    --compression-mode file \
    --compression-format zstd \
    -o /ws/bags/my_session \
    /camera/image_raw /cmd_vel
"
```

Ctrl+C to stop recording cleanly. The bag lands in `./bags/my_session/` on the host.

### Playback

Stop the main stack first (avoids topic conflicts with camera_node), then:

```bash
docker compose run --rm ocelot bash -i -c "
  ros2 run web_video_server web_video_server &
  ros2 run ocelot visualizer_node &
  ros2 bag play /ws/bags/my_session --loop
"
```

| Stream | URL |
|---|---|
| Raw | `http://<pi-ip>:8080/stream?topic=/camera/image_raw` |
| Annotated | `http://<pi-ip>:8080/stream?topic=/camera/image_annotated` |

The annotated stream shows face bounding box, center crosshair, error vector, deadband circle, and cmd_vel values.

### Inspect

```bash
docker compose exec ocelot bash -i -c "ros2 bag info /ws/bags/my_session"
```

## Architecture

Classical tracker:
```
camera_node ──/camera/image_raw──▶ tracker_node ──/cmd_vel──▶ servo_node
 (picamera2)   sensor_msgs/Image    (Haar cascade)  Twist       (PCA9685)
                      │
                      └──▶ visualizer_node ──/camera/image_annotated──▶ web_video_server
                              (optional)
```

VLA mode (default):
```
camera_node ──/camera/image_raw──▶ vla_node ──/cmd_vel──▶ servo_node
 (picamera2)   sensor_msgs/Image    (ONNX INT8)  Twist       (PCA9685)
```

`tracker_node` and `vla_node` are mutually exclusive — only one runs at a time (`USE_HAAR=true` selects Haar).

### Nodes

**`camera_node`** — Captures 640×480 RGB frames from Pi Camera V2. Because `libcamera`'s Python bindings are compiled for Python 3.11 (Pi OS Bookworm) and the ROS container uses Python 3.12, capture runs in a `python3.11` subprocess (`capture_worker.py`) communicating frames to the node via a length-prefixed pipe.

**`servo_node`** — Subscribes to `/cmd_vel` (`geometry_msgs/Twist`). Integrates `angular.z` (pan) and `angular.y` (tilt) velocity at 30 Hz into servo positions via `adafruit-circuitpython-servokit`. Centers on shutdown.

**`tracker_node`** — Subscribes to `/camera/image_raw`, runs Haar cascade face detection, publishes velocity commands to `/cmd_vel` and bounding box to `/tracking/face_roi`. Key params: `kp_pan`, `kp_tilt`, `deadband`, `min_neighbors`, `min_face_size`. Enabled only when `USE_HAAR=true`.

**`vla_node`** — Subscribes to `/camera/image_raw`, runs the trained ONNX model (DINOv2-small + CLIP text encoder + action head), publishes `/cmd_vel`. Default mode. Key params: `checkpoint`, `token_cache`, `command`, `max_vel`, `max_accel`.

**`visualizer_node`** — Subscribes to `/camera/image_raw`, `/tracking/face_roi`, and `/cmd_vel`; publishes annotated frames to `/camera/image_annotated`. Optional — launch with `visualize:=true`.


## Validate

```bash
# I2C — should show 0x40
i2cdetect -y 1

# Manual servo via ROS topic
ros2 topic pub --once /cmd_vel geometry_msgs/Twist \
  "{angular: {z: 1.0, y: 0.0}}"

# Confirm publish rates
ros2 topic list
ros2 topic hz /camera/image_raw    # expect ~15 Hz
ros2 topic echo /cmd_vel --no-arr
```


## Project Structure

```
ocelot/
├── ocelot/
│   ├── camera_node.py       # ROS node (py3.12), spawns capture_worker
│   ├── capture_worker.py    # picamera2 capture (py3.11 subprocess)
│   ├── servo_node.py        # PCA9685 servo control
│   ├── tracker_node.py      # Haar cascade proportional controller
│   ├── oracle_node.py       # Privileged ground-truth FK tracker (sim only)
│   ├── oracle_validator.py  # Pixel-error measurement for oracle validation
│   ├── vla_node.py          # ONNX inference node for sim validation (Phase 3)
│   └── visualizer_node.py   # Annotated image publisher (optional)
├── launch/tracker_launch.py
├── config/tracker_params.yaml
├── urdf/pan_tilt.urdf
├── bags/                    # rosbag recordings (gitignored)
├── scripts/                 # bare-metal validation (no ROS needed)
│   ├── test_servos.py
│   └── test_tracking_manual.py
├── deploy/docker/
│   ├── Dockerfile.robot     # robot deployment image (Pi 5)
│   └── docker-compose.yml   # compose config (relative to deploy/docker/)
├── docker-compose.yml       # convenience wrapper — includes deploy/docker/
├── package.xml              # ament_python
├── setup.py
└── setup.cfg
```

## Troubleshooting

### `haarcascade_frontalface_default.xml not found`
The apt `python3-opencv` package does not bundle cascade data files. `opencv-data` must also be installed — it provides the cascade XMLs at `/usr/share/opencv4/haarcascades/`. This is already in `Dockerfile.sim`. If you see this error after a rebuild, check that both `python3-opencv` and `opencv-data` are present in the apt install section.

### `ImportError: libturbojpeg.so.0: cannot open shared object file`
simplejpeg (required by picamera2's JPEG encoder) needs `libturbojpeg` from the host. Check that `deploy/docker/docker-compose.yml` bind-mounts `/usr/lib/aarch64-linux-gnu/libturbojpeg.so.0` from the host.

### `ModuleNotFoundError: No module named 'v4l2'`
picamera2 imports `v4l2` for sensor mode enumeration. The file lives at `/usr/lib/python3/dist-packages/v4l2.py` on the host and must be bind-mounted into the container. Check `deploy/docker/docker-compose.yml`.

### Stale or incompatible `.venv`
If the `.venv` was created outside the container (host Pi OS Python 3.11 has a different ABI from deadsnakes), or if numpy/simplejpeg compatibility breaks after a Pi OS update, delete and recreate it inside the container:
```bash
docker compose run --rm ocelot bash -i -c "
  rm -rf /ws/src/ocelot/.venv &&
  python3.11 -m venv --without-pip /ws/src/ocelot/.venv &&
  /ws/src/ocelot/.venv/bin/python3.11 -c 'import urllib.request; exec(urllib.request.urlopen(\"https://bootstrap.pypa.io/get-pip.py\").read())' &&
  /ws/src/ocelot/.venv/bin/pip install -r /ws/src/ocelot/requirements-worker.txt
"
```
Then `docker compose up` as normal — the venv will be picked up on the next run.

### `pip dependency resolver` warning about `pyyaml` / `launch-ros`
Harmless. pip can see ROS packages in the environment and warns about missing deps for them. The worker venv doesn't need them — ignore it.

### Annotated stream blank / `visualizer_node` missing from `ros2 node list`
If `VISUALIZE=true docker compose up` starts only 4 nodes (no `visualizer_node`), the colcon install directory is stale. Run a rebuild inside the container then restart:
```bash
docker compose run --rm ocelot bash -i -c "cd /ws && colcon build --packages-select ocelot --symlink-install"
VISUALIZE=true docker compose up
```
This is needed whenever a new entry point is added to `setup.py`.

### Sim (Gazebo) — `docker-compose.sim.yml`

#### Gazebo window appears but freezes / not responding
Root cause: Gazebo transport tries multicast peer discovery on all interfaces when `GZ_IP` is unset. The GUI event loop blocks waiting for the server handshake — the window frame appears (Qt init succeeds) but hangs before the scene loads.

Fix (already in `docker-compose.sim.yml`):
```yaml
environment:
  - GZ_IP=127.0.0.1
```
This binds Gazebo transport to loopback only, so server↔GUI discovery resolves instantly.

#### Gazebo window is black / empty world
Root cause: Docker's default `/dev/shm` is 64 MB — too small for Gazebo's OGRE renderer, which transfers render buffers between server and GUI via shared memory.

Fix (already in `docker-compose.sim.yml`):
```yaml
shm_size: '2g'
ipc: host
environment:
  - QT_X11_NO_MITSHM=1
```

#### X11 auth: container (root) refused by X server

Run `sudo make sim-xauth` once (re-run if the display session changes). The compose file mounts `/tmp/.docker.xauth` and sets `XAUTHORITY=/tmp/.docker.xauth`.

#### `MESA: error: ZINK: vkCreateInstance failed` / software rendering
The `jazzy-simulation` base image doesn't include Vulkan ICDs, so OGRE logs this and falls back to software OpenGL (llvmpipe). This is expected and harmless when running without the GPU overlay — the sim works but renders on CPU.

To switch to GPU-accelerated rendering (NVIDIA), use the GPU compose overlay as described in the [Sim section](#sim-dev-machine) above.

---

## Phase Roadmap
| Phase | Weeks | Goal |
|---|---|---|
| 1 | 1–4 | Classical face tracker (Haar cascade) — **complete** |
| 2 | 5–8 | Gazebo sim + synthetic data engine — **complete** |
| 3 | 9–13 | VLA model (DINOv2 + CLIP + action head) — **complete** |
| 4 | 14–18 | Edge deployment + MLOps loop — **current** |
| 5 | 19–20 | Polish + portfolio |
