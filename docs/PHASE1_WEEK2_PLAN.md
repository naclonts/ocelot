# Phase 1, Week 2: ROS 2 Foundation

## Starting Point

Session 1 (hardware validation) is complete:
- PCA9685 servo driver working at I2C 0x40 via `adafruit-circuitpython-servokit`
- Pan (ch0, 0-180°) and tilt (ch1, 90-180°) ranges confirmed
- Pi Camera V2 capturing 640x480 frames via picamera2
- Manual tracking script validates full camera+servo pipeline

**Not yet done:** No ROS 2, no Docker, no URDF, no package structure beyond empty `ocelot/__init__.py`. `setup.py` still lists `pantilthat` as a dependency (stale).

## Cleanup First

Before building new things, fix what's stale:

1. **Update `setup.py`** — replace `pantilthat` with `adafruit-circuitpython-servokit` and add `smbus2`
2. **Add a `requirements.txt`** — pin working versions from the current venv

## Goal

By end of week: ROS 2 Jazzy running in Docker on the Pi 5, three ROS 2 nodes communicating over topics, and a URDF describing the pan-tilt mechanism. You can `ros2 topic pub` a velocity command and watch the servos move.

---

## Session A: Docker + ROS 2 Environment (~1-2 hrs)

### A1. Dockerfile

Base image: `ros:jazzy-ros-base`. Install on top:
- `python3-opencv`, `python3-smbus2`, `ros-jazzy-cv-bridge`
- `adafruit-circuitpython-servokit` via pip (not in apt)
- `picamera2` via pip (or `python3-picamera2` from apt if available in the container)

### A2. docker-compose.yml

```yaml
services:
  ocelot:
    build: .
    privileged: true
    network_mode: host
    volumes:
      - ./:/ws/src/ocelot
      - /dev:/dev
    environment:
      - ROS_DOMAIN_ID=0
    devices:
      - /dev/i2c-1:/dev/i2c-1
      - /dev/video0:/dev/video0
```

Key decisions:
- `privileged: true` for I2C and camera access (simplest on Pi 5)
- `network_mode: host` for DDS discovery without config
- Mount source for live editing

### A3. Validate inside container

- `i2cdetect -y 1` shows 0x40
- Quick Python snippet moves a servo
- `libcamera-hello` or picamera2 test captures a frame

**If camera fails in Docker:** May need `--device /dev/media0`, `/dev/media1`, etc. Pi Camera V2 uses libcamera which exposes multiple device nodes. Debug here, not later.

---

## Session B: ROS 2 Package + Three Nodes (~2-3 hrs)

### B1. Package structure

```
ocelot/
├── ocelot/
│   ├── __init__.py
│   ├── camera_node.py
│   ├── servo_node.py
│   └── tracker_node.py       # placeholder — publishes zero velocity
├── launch/
│   └── tracker_launch.py
├── config/
│   └── tracker_params.yaml
├── urdf/
│   └── pan_tilt.urdf
├── package.xml
├── setup.py
├── setup.cfg
├── Dockerfile
├── docker-compose.yml
└── scripts/                   # existing validation scripts (keep)
```

Use `ament_python` package format.

### B2. camera_node.py

- Uses `picamera2` to capture 640x480 RGB888 frames
- Publishes to `/camera/image_raw` as `sensor_msgs/Image` via `cv_bridge`
- Parameter: `fps` (default 15), `resolution` (default [640, 480])
- Timer-driven at configured FPS

### B3. servo_node.py

- Subscribes to `/cmd_vel` (`geometry_msgs/Twist`)
- `angular.z` = pan velocity, `angular.y` = tilt velocity
- Integrates velocity into position at 30 Hz timer
- Clamps: pan 0-180°, tilt 90-180°
- Uses `adafruit-circuitpython-servokit` with PCA9685
- Parameter: `velocity_scale` (default 30 deg/s per unit velocity)
- On shutdown: centers pan to 90°, tilt to 180° (forward)
- Pulse range: 500-2500 microseconds

### B4. tracker_node.py (placeholder)

- Subscribes to `/camera/image_raw` (does nothing with it yet)
- Publishes zero `Twist` to `/cmd_vel` at 10 Hz
- This just proves the full topic chain works end-to-end
- Real face tracking logic comes in week 3

### B5. Launch file + params

`tracker_launch.py` launches all three nodes with parameters from `tracker_params.yaml`:

```yaml
camera_node:
  ros__parameters:
    fps: 15
    resolution: [640, 480]

servo_node:
  ros__parameters:
    velocity_scale: 30.0
    pan_center: 90
    tilt_center: 180
    pan_limits: [0, 180]
    tilt_limits: [90, 180]

tracker_node:
  ros__parameters:
    enabled: false   # placeholder, no tracking yet
```

---

## Session C: URDF + Validation (~1 hr)

### C1. pan_tilt.urdf

Describe the mechanism as:
- `base_link` — fixed mount point
- `pan_link` — connected to base via revolute joint, axis Z, limits 0-180° (centered at 90°)
- `tilt_link` — connected to pan via revolute joint, axis Y, limits 90-180° (centered at 180°)
- `camera_link` — fixed to tilt_link

This URDF isn't used for control yet — it's needed later for Gazebo in Phase 2. Building it now while the physical hardware is fresh in your mind is the right time.

### C2. End-to-end validation checklist

Run inside Docker:

1. `ros2 launch ocelot tracker_launch.py` — all three nodes start clean
2. `ros2 topic list` — shows `/camera/image_raw` and `/cmd_vel`
3. `ros2 topic hz /camera/image_raw` — confirms ~15 Hz publish rate
4. `ros2 topic echo /cmd_vel` — shows zero velocities from placeholder tracker
5. Manual servo test:
   ```
   ros2 topic pub --once /cmd_vel geometry_msgs/Twist \
     "{angular: {y: 0.0, z: 1.0}}"
   ```
   Servo should pan. Ctrl+C, it stops.
6. `ros2 topic echo /camera/image_raw --no-arr` — confirms image messages flowing

If all six pass, week 2 is done.

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| picamera2 doesn't work inside Docker | Try mounting all `/dev/media*` and `/dev/video*` devices. Fallback: USB webcam with `cv2.VideoCapture(0)` |
| ROS 2 Jazzy ARM64 image is large / slow to build | Use multi-stage build. Cache apt and pip layers. Build once, iterate with mounted source |
| I2C permission denied in container | `privileged: true` should handle it. If not, add user to `i2c` group inside container |
| DDS discovery issues | `network_mode: host` avoids this. If still broken, set `ROS_LOCALHOST_ONLY=1` for single-machine |

## Time Estimate

| Session | Estimated |
|---|---|
| A: Docker + ROS 2 env | 1-2 hrs |
| B: Package + 3 nodes | 2-3 hrs |
| C: URDF + validation | 1 hr |
| **Total** | **4-6 hrs** |
