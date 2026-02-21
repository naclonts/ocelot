# Ocelot

Pan-tilt face tracking robot (Raspberry Pi 5). Classical CV baseline with Haar cascade for now; VLA model coming soon.

## Hardware
| Component | Detail |
|---|---|
| Compute | Raspberry Pi 5 |
| Camera | Pi Camera V2 (CSI) |
| Servo driver | PCA9685 at I2C 0x40 |
| Servos | SG90 — pan ch0 (0–180°), tilt ch1 (90–180°) |
| OS | Pi OS Bookworm (Python 3.11) |

## Architecture

```
camera_node ──/camera/image_raw──▶ tracker_node ──/cmd_vel──▶ servo_node
 (picamera2)   sensor_msgs/Image    (Haar cascade)  Twist       (PCA9685)
```

Three ROS 2 Jazzy nodes in a single `ament_python` package, running in Docker.

### Nodes

**`camera_node`** — Captures 640×480 RGB frames from Pi Camera V2. Because `libcamera`'s Python bindings are compiled for Python 3.11 (Pi OS Bookworm) and the ROS container uses Python 3.12, capture runs in a `python3.11` subprocess (`capture_worker.py`) communicating frames to the node via a length-prefixed pipe.

**`servo_node`** — Subscribes to `/cmd_vel` (`geometry_msgs/Twist`). Integrates `angular.z` (pan) and `angular.y` (tilt) velocity at 30 Hz into servo positions via `adafruit-circuitpython-servokit`. Centers on shutdown.

**`tracker_node`** — Subscribes to `/camera/image_raw`, runs Haar cascade face detection, publishes proportional velocity commands to `/cmd_vel`. Disabled by default; enable with `ros2 param set /tracker_node enabled true`. Key params: `kp_pan`, `kp_tilt`, `deadband`, `min_neighbors`, `min_face_size`.

## Quickstart

### First time

```bash
# 1. Build Docker image
docker compose build

# 2. Create the Python 3.11 venv and build the ROS package (once, or after setup.py changes)
docker compose run --rm ocelot bash -c "
  python3.11 -m venv /ws/src/ocelot/.venv &&
  /ws/src/ocelot/.venv/bin/pip install -r /ws/src/ocelot/requirements-worker.txt &&
  source /opt/ros/jazzy/setup.bash &&
  cd /ws &&
  colcon build --packages-select ocelot --symlink-install
"
```

> **Why the venv?** `capture_worker.py` runs under the container's `python3.11` (deadsnakes)
> so that picamera2/libcamera — compiled for Pi OS Bookworm's Python 3.11 — work correctly.
> picamera2 itself is bind-mounted from the host; the venv provides its Python deps
> (numpy, Pillow, jsonschema) that aren't in the Docker image.
>
> **Why `--symlink-install`?** The camera node uses `__file__` to locate the `.venv`.
> Without symlinks, that path resolves to the install dir instead of the source tree.

### Launch

```bash
docker compose up
```

Editing Python source files does **not** require a rebuild (symlinks are live). Rebuilding
is only needed when `setup.py` entry points change.

### View camera stream

```
http://<pi-ip>:8080/stream?topic=/camera/image_raw
```

### Enable face tracking

```bash
ros2 param set /tracker_node enabled true
```

## Rosbag

Record `/camera/image_raw` + `/cmd_vel` with zstd compression (bags land in `./bags/`):

```bash
RECORD=true docker compose up
```

Playback or inspect (inside container):

```bash
ros2 bag play /ws/bags/<bag-dir>
ros2 bag info /ws/bags/<bag-dir>
```

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
│   └── tracker_node.py      # Haar cascade proportional controller
├── launch/tracker_launch.py
├── config/tracker_params.yaml
├── urdf/pan_tilt.urdf
├── bags/                    # rosbag recordings (gitignored)
├── scripts/                 # bare-metal validation (no ROS needed)
│   ├── test_servos.py
│   └── test_tracking_manual.py
├── Dockerfile
├── docker-compose.yml
├── package.xml              # ament_python
├── setup.py
└── setup.cfg
```

## Phase Roadmap
| Phase | Weeks | Goal |
|---|---|---|
| 1 | 1–4 | Classical face tracker (Haar cascade) — **current** |
| 2 | 5–8 | Gazebo sim + synthetic data engine |
| 3 | 9–13 | VLA model (DINOv2 + CLIP + action head) |
| 4 | 14–18 | Edge deployment + MLOps loop |
| 5 | 19–20 | Polish + portfolio |
