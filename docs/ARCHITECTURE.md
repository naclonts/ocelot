# Architecture

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

## Nodes

**`camera_node`** — Captures 640×480 RGB frames from Pi Camera V2. Because `libcamera`'s Python bindings are compiled for Python 3.11 (Pi OS Bookworm) and the ROS container uses Python 3.12, capture runs in a `python3.11` subprocess (`capture_worker.py`) communicating frames to the node via a length-prefixed pipe.

**`servo_node`** — Subscribes to `/cmd_vel` (`geometry_msgs/Twist`). Integrates `angular.z` (pan) and `angular.y` (tilt) velocity at 30 Hz into servo positions via `adafruit-circuitpython-servokit`. Centers on shutdown.

**`tracker_node`** — Subscribes to `/camera/image_raw`, runs Haar cascade face detection, publishes velocity commands to `/cmd_vel` and bounding box to `/tracking/face_roi`. Key params: `kp_pan`, `kp_tilt`, `deadband`, `min_neighbors`, `min_face_size`. Enabled only when `USE_HAAR=true`.

**`vla_node`** — Subscribes to `/camera/image_raw`, runs the trained ONNX model (DINOv2-small + CLIP text encoder + action head), publishes `/cmd_vel`. Default mode. Key params: `checkpoint`, `token_cache`, `command`, `max_vel`, `max_accel`.

**`visualizer_node`** — Subscribes to `/camera/image_raw`, `/tracking/face_roi`, and `/cmd_vel`; publishes annotated frames to `/camera/image_annotated`. Optional — launch with `visualize:=true`.
