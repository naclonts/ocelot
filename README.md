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

## Quickstart

### First time

```bash
# 1. Build Docker image
docker compose build

# 2. Create the Python 3.11 venv and build the ROS package (once, or after setup.py changes)
docker compose run --rm ocelot bash -i -c "
  python3.11 -m venv --without-pip /ws/src/ocelot/.venv &&
  /ws/src/ocelot/.venv/bin/python3.11 -c \"import urllib.request; exec(urllib.request.urlopen('https://bootstrap.pypa.io/get-pip.py').read())\" &&
  /ws/src/ocelot/.venv/bin/pip install -r /ws/src/ocelot/requirements-worker.txt &&
  cd /ws && colcon build --packages-select ocelot --symlink-install
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
docker compose up                          # tracker only
VISUALIZE=true docker compose up           # tracker + annotated stream
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
# One-time: build the sim image
docker compose -f deploy/docker/docker-compose.sim.yml build

# Run headless (server only — no GUI, fast, validated path)
docker compose -f deploy/docker/docker-compose.sim.yml run --rm sim bash -c \
  "source /opt/ros/jazzy/setup.bash && cd /ws && \
   colcon build --symlink-install --packages-select ocelot --event-handlers console_direct- && \
   source /ws/install/setup.bash && \
   ros2 launch ocelot sim_launch.py headless:=true"

# Run with Gazebo GUI (requires one-time xauth setup below)
docker compose -f deploy/docker/docker-compose.sim.yml run --rm sim bash -c \
  "source /opt/ros/jazzy/setup.bash && cd /ws && \
   colcon build --symlink-install --packages-select ocelot --event-handlers console_direct- && \
   source /ws/install/setup.bash && \
   ros2 launch ocelot sim_launch.py headless:=false"
```

The colcon build is fast on repeat runs — named volumes (`sim_build`, `sim_install`) cache artifacts between container invocations.

**One-time xauth setup** (only needed for GUI mode; re-run if the display session changes):
```bash
touch /tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f /tmp/.docker.xauth nmerge -
```

**Interactive shell** (for manual exploration):
```bash
docker compose -f deploy/docker/docker-compose.sim.yml run --rm sim bash
```

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

```
camera_node ──/camera/image_raw──▶ tracker_node ──/cmd_vel──▶ servo_node
 (picamera2)   sensor_msgs/Image    (Haar cascade)  Twist       (PCA9685)
                      │
                      └──▶ visualizer_node ──/camera/image_annotated──▶ web_video_server
                              (optional)
```

Three ROS 2 Jazzy nodes in a single `ament_python` package, running in Docker.

### Nodes

**`camera_node`** — Captures 640×480 RGB frames from Pi Camera V2. Because `libcamera`'s Python bindings are compiled for Python 3.11 (Pi OS Bookworm) and the ROS container uses Python 3.12, capture runs in a `python3.11` subprocess (`capture_worker.py`) communicating frames to the node via a length-prefixed pipe.

**`servo_node`** — Subscribes to `/cmd_vel` (`geometry_msgs/Twist`). Integrates `angular.z` (pan) and `angular.y` (tilt) velocity at 30 Hz into servo positions via `adafruit-circuitpython-servokit`. Centers on shutdown.

**`tracker_node`** — Subscribes to `/camera/image_raw`, runs Haar cascade face detection, publishes velocity commands to `/cmd_vel` and bounding box to `/tracking/face_roi`. Key params: `kp_pan`, `kp_tilt`, `deadband`, `min_neighbors`, `min_face_size`.

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
The apt `python3-opencv` package does not bundle cascade data files. The Dockerfile must install `opencv-python-headless` via pip instead. If you see this after a rebuild, check that `python3-opencv` is absent from the apt section and `opencv-python-headless` is in the pip section.

### `ImportError: libturbojpeg.so.0: cannot open shared object file`
simplejpeg (required by picamera2's JPEG encoder) needs `libturbojpeg` from the host. Check that `deploy/docker/docker-compose.yml` bind-mounts `/usr/lib/aarch64-linux-gnu/libturbojpeg.so.0` from the host.

### `ModuleNotFoundError: No module named 'v4l2'`
picamera2 imports `v4l2` for sensor mode enumeration. The file lives at `/usr/lib/python3/dist-packages/v4l2.py` on the host and must be bind-mounted into the container. Check `deploy/docker/docker-compose.yml`.

### `ensurepip` fails when creating the Python 3.11 venv
deadsnakes Python 3.11 on Ubuntu 24.04 (Noble) does not bundle the pip wheel used by `ensurepip`. Always create the venv with `--without-pip` and bootstrap pip via `get-pip.py` as shown in the build step above.

### Stale `.venv` from host Python
If the `.venv` was created by the host's Pi OS Python 3.11 (outside the container), delete it and recreate inside the container:
```bash
rm -rf .venv
# then re-run step 2 of First time setup
```

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
The container runs as root; `xhost +si:localuser:nathan` won't help. Use a wildcard xauth cookie instead. One-time host setup (re-run if the display session changes):
```bash
touch /tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f /tmp/.docker.xauth nmerge -
```
The compose file mounts `/tmp/.docker.xauth` and sets `XAUTHORITY=/tmp/.docker.xauth`.

#### `MESA: error: ZINK: vkCreateInstance failed` / software rendering
The `jazzy-simulation` base image doesn't include Vulkan ICDs, so OGRE falls back to software OpenGL (llvmpipe). The sim works but renders on CPU. To enable GPU acceleration, add to `Dockerfile.sim`:
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*
```

---

## Phase Roadmap
| Phase | Weeks | Goal |
|---|---|---|
| 1 | 1–4 | Classical face tracker (Haar cascade) — **complete** |
| 2 | 5–8 | Gazebo sim + synthetic data engine — **current** |
| 3 | 9–13 | VLA model (DINOv2 + CLIP + action head) |
| 4 | 14–18 | Edge deployment + MLOps loop |
| 5 | 19–20 | Polish + portfolio |
