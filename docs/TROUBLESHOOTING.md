# Troubleshooting

## Validation / bring-up checks

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

## `haarcascade_frontalface_default.xml not found`

The apt `python3-opencv` package does not bundle cascade data files. `opencv-data` must also be installed — it provides the cascade XMLs at `/usr/share/opencv4/haarcascades/`. This is already in `Dockerfile.sim`. If you see this error after a rebuild, check that both `python3-opencv` and `opencv-data` are present in the apt install section.

## `ImportError: libturbojpeg.so.0: cannot open shared object file`

simplejpeg (required by picamera2's JPEG encoder) needs `libturbojpeg` from the host. Check that `deploy/docker/docker-compose.yml` bind-mounts `/usr/lib/aarch64-linux-gnu/libturbojpeg.so.0` from the host.

## `ModuleNotFoundError: No module named 'v4l2'`

picamera2 imports `v4l2` for sensor mode enumeration. The file lives at `/usr/lib/python3/dist-packages/v4l2.py` on the host and must be bind-mounted into the container. Check `deploy/docker/docker-compose.yml`.

## Stale or incompatible `.venv`

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

## `pip dependency resolver` warning about `pyyaml` / `launch-ros`

Harmless. pip can see ROS packages in the environment and warns about missing deps for them. The worker venv doesn't need them — ignore it.

## Annotated stream blank / `visualizer_node` missing from `ros2 node list`

If `VISUALIZE=true docker compose up` starts only 4 nodes (no `visualizer_node`), the colcon install directory is stale. Run a rebuild inside the container then restart:

```bash
docker compose run --rm ocelot bash -i -c "cd /ws && colcon build --packages-select ocelot --symlink-install"
VISUALIZE=true docker compose up
```

This is needed whenever a new entry point is added to `setup.py`.

## Sim (Gazebo) — `docker-compose.sim.yml`

### Gazebo window appears but freezes / not responding

Root cause: Gazebo transport tries multicast peer discovery on all interfaces when `GZ_IP` is unset. The GUI event loop blocks waiting for the server handshake — the window frame appears (Qt init succeeds) but hangs before the scene loads.

Fix (already in `docker-compose.sim.yml`):

```yaml
environment:
  - GZ_IP=127.0.0.1
```

This binds Gazebo transport to loopback only, so server↔GUI discovery resolves instantly.

### Gazebo window is black / empty world

Root cause: Docker's default `/dev/shm` is 64 MB — too small for Gazebo's OGRE renderer, which transfers render buffers between server and GUI via shared memory.

Fix (already in `docker-compose.sim.yml`):

```yaml
shm_size: '2g'
ipc: host
environment:
  - QT_X11_NO_MITSHM=1
```

### X11 auth: container (root) refused by X server

Run `sudo make sim-xauth` once (re-run if the display session changes). The compose file mounts `/tmp/.docker.xauth` and sets `XAUTHORITY=/tmp/.docker.xauth`.

### `MESA: error: ZINK: vkCreateInstance failed` / software rendering

The `jazzy-simulation` base image doesn't include Vulkan ICDs, so OGRE logs this and falls back to software OpenGL (llvmpipe). This is expected and harmless when running without the GPU overlay — the sim works but renders on CPU.

To switch to GPU-accelerated rendering (NVIDIA), use the GPU compose overlay as described in the sim section of the main `README.md`.
