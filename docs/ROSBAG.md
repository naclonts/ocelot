# Rosbag

Bags are stored in `./bags/` (bind-mounted to `/ws/bags/` in the container).

## Record

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

## Playback

Stop the main stack first to avoid topic conflicts with `camera_node`, then:

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

The annotated stream shows face bounding box, center crosshair, error vector, deadband circle, and `cmd_vel` values.

## Inspect

```bash
docker compose exec ocelot bash -i -c "ros2 bag info /ws/bags/my_session"
```
