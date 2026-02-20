# Ocelot

Pan-tilt face tracking robot (Raspberry Pi 5). Classical CV baseline in Phase 1; VLA model in later phases.

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

**`tracker_node`** — Week 2 placeholder: subscribes to `/camera/image_raw`, publishes zero `Twist` at 10 Hz. Week 3 replaces this with Haar cascade proportional controller.

## Quickstart

```bash
# Build image
docker compose build

# Shell into container
docker compose run --rm ocelot bash

# Inside container: build package and launch
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ocelot --symlink-install
source install/setup.bash
ros2 launch ocelot tracker_launch.py
```

## Validate (A3 checklist)

```bash
# I2C — should show 0x40
i2cdetect -y 1

# Servo quick test
python3 -c "
from adafruit_servokit import ServoKit; import time
k = ServoKit(channels=16)
k.servo[0].set_pulse_width_range(500, 2500)
k.servo[0].angle = 45; time.sleep(0.5)
k.servo[0].angle = 135; time.sleep(0.5)
k.servo[0].angle = 90; print('OK')
"

# Manual servo via ROS topic
ros2 topic pub --once /cmd_vel geometry_msgs/Twist \
  "{angular: {z: 1.0, y: 0.0}}"

# Confirm topics
ros2 topic list
ros2 topic hz /camera/image_raw
ros2 topic echo /cmd_vel --no-arr
```

## Project Structure

```
ocelot/
├── ocelot/
│   ├── camera_node.py       # ROS node (py3.12), spawns capture_worker
│   ├── capture_worker.py    # picamera2 capture (py3.11 subprocess)
│   ├── servo_node.py        # PCA9685 servo control
│   └── tracker_node.py      # placeholder (week 3: Haar cascade)
├── launch/tracker_launch.py
├── config/tracker_params.yaml
├── urdf/pan_tilt.urdf
├── scripts/                 # bare-metal validation (no ROS needed)
│   ├── test_servos.py
│   └── test_tracking_manual.py
├── docs/
│   ├── DEV_PLAN.md
│   └── PHASE1_WEEK2_PLAN.md
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
