## Quick Hardware Assessment

You have a great setup. The **Pimoroni Pan-Tilt HAT** is your I2C servo driver (plugs onto the GPIO header, handles PWM), the **Adafruit Mini Pan-Tilt Kit** provides the mechanical bracket + SG90 servos, and the **Pi Camera V2** connects via CSI. Your existing `audio-visual-bot` repo already proves `pantilthat` + Haar cascade face tracking works — so the job is wrapping that in ROS 2 nodes.

**One thing to watch:** Pi 5 uses a new RP1 I/O chipset. The `pantilthat` library may need its latest Git version, or you may need to fall back to `adafruit-circuitpython-servokit`. Confirming this before anything else saves a lot of pain.

---

## Decision: ROS 2 via Docker (Recommended)

Your project plan calls for Dockerization in Week 4 anyway. Start with it on Day 1 instead — run **ROS 2 Jazzy** inside a Docker container on Raspberry Pi OS Bookworm. You pass hardware through with `privileged: true` and `network_mode: host`. This means you never have to retroactively containerize a working native setup.

---

## The Three Sessions

### Session 1: Validate Hardware on Bare Metal (~30 min)

Before touching ROS, confirm the hardware works on raw Pi OS:

1. Enable I2C via `raspi-config`, reboot
2. Run `i2cdetect -y 1` — look for address `0x15` (the Pimoroni HAT)
3. `pip install pantilthat`, run a servo sweep script (pan -45 to +45)
4. `sudo apt install python3-picamera2`, capture a test frame to JPEG
5. If `pantilthat` fails on Pi 5, install `adafruit-circuitpython-servokit` as fallback

If servos sweep and camera captures, your hardware is good. If either fails, debug here — not inside Docker.

### Session 2: Docker + Package Structure (~1-2 hrs)

Set up the ROS 2 Docker environment:

**docker-compose.yml** passes through I2C and camera devices, uses `network_mode: host` for DDS discovery, and mounts your `src/` directory for live editing. The Dockerfile extends `ros:jazzy-ros-base`, installs `python3-opencv`, `python3-smbus`, `python3-picamera2`, `ros-jazzy-cv-bridge`, and `pantilthat`.

Create the `mini_vla` ament_python package with this structure:

```
mini_vla/
├── mini_vla/
│   ├── camera_node.py
│   ├── servo_node.py
│   └── face_tracker_node.py
├── launch/
│   └── tracker_launch.py
├── config/
│   └── tracker_params.yaml
├── package.xml
└── setup.py
```

### Session 3: Implement Nodes + Tune (~2-3 hrs)

You need three nodes connected by two topics:

```
camera_node ──/camera/image_raw──→ face_tracker_node ──/cmd_vel──→ servo_node
           (sensor_msgs/Image)                     (geometry_msgs/Twist)
```

**camera_node** — Uses `picamera2` to capture frames, publishes via `cv_bridge` as `rgb8`. Configurable FPS (start at 15) and resolution (640×480). Fallback: if picamera2 is painful in Docker, swap to `cv2.VideoCapture(0)` with a USB webcam — gets you unblocked immediately.

**servo_node** — Subscribes to `/cmd_vel`, uses `angular.z` for pan velocity and `angular.y` for tilt velocity. Integrates velocity into position at 30Hz, clamps to servo limits, calls `pantilthat.pan()` / `pantilthat.tilt()`. Key parameter: `velocity_scale` (degrees per second per unit velocity — start at 30, tune from there). On shutdown, centers both servos.

**face_tracker_node** — Subscribes to `/camera/image_raw`, runs Haar cascade face detection, computes proportional velocity commands based on face-center-to-image-center error. Key parameters: `kp_pan`/`kp_tilt` (proportional gain, start 0.3), `deadband` (pixel threshold, start 30), `max_velocity` (clamp, start 2.0). Publishes zero velocity when no face detected.

The velocity convention is: face is right of center → negative `angular.z` → pan left toward face. You'll almost certainly need to flip a sign or two during tuning — that's normal.

---

## Tuning (What AI Can't Do For You)

Once launched with `ros2 launch mini_vla tracker_launch.py`:

1. **Direction check:** If camera moves *away* from your face, flip the sign on the relevant axis in `face_tracker_node`
2. **velocity_scale:** Barely moves → increase. Slams to limits → decrease
3. **kp gains:** Jumpy/oscillating → lower. Sluggish → raise
4. **deadband:** Jitters when face is centered → increase. Feels unresponsive → decrease
5. **FPS:** If tracking feels laggy, try 30, but watch CPU usage

---

## Pi 5 Gotchas to Expect

- **`pantilthat` Pi 5 compatibility**: The RP1 chipset changed I2C/GPIO drivers. If it fails, `adafruit-circuitpython-servokit` is the proven alternative — I can write that version of servo_node if needed
- **Camera in Docker**: Pi Camera V2 uses libcamera. Verify `libcamera-hello` works on the host first. In Docker you may need `--device /dev/media0` in addition to `/dev/video0`
- **cv_bridge encoding**: picamera2 outputs RGB888; OpenCV expects BGR. Publish as `rgb8`, request `bgr8` in the subscriber — cv_bridge handles the conversion
- **Servo jitter**: If servos buzz at idle, it's the continuous PWM signal. Call `pantilthat.servo_enable(channel, False)` when stationary

---

## How This Maps to Your 20-Week Plan

| Project Plan | What You're Doing |
|---|---|
| Week 1 (assembly + sweep) | Session 1 hardware validation |
| Week 2 (ROS 2 foundation) | Sessions 2-3: Docker, package, three nodes |
| Week 3 (face tracking + tuning) | Tuning checklist above |
| Week 4 (recording + Docker + GitHub) | Already Dockerized; add recorder node + rosbag2 logging |

You're essentially doing Weeks 1-2 of Phase 1. The URDF comes next (needed for Gazebo in Phase 2) — it describes your two revolute joints and can be validated with `rviz2` before you ever touch simulation.

Want me to write out the complete files (Dockerfile, docker-compose, all three nodes, launch file, setup.py) as a ready-to-clone package? Or tackle the `adafruit-servokit` fallback version of servo_node first?
