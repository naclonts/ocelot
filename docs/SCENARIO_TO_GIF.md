# Scenario -> Video -> GIF Runbook

This runbook shows how to capture an oracle-driven sim scenario and convert it
into a web-friendly GIF.

## 1) Record a scenario to MP4

From repo root:

```bash
docker compose -f deploy/docker/docker-compose.sim.yml run --rm sim bash -lc '
  source /opt/ros/jazzy/setup.bash && \
  cd /ws && \
  colcon build --symlink-install --packages-select ocelot --event-handlers console_direct- && \
  source /ws/install/setup.bash && \
  ros2 launch ocelot sim_launch.py world:=scenario_world use_oracle:=true headless:=true & \
  sleep 15 && \
  python3 /ws/src/ocelot/sim/preview_episode.py --seed 280 --out /ws/src/ocelot/sim/oracle-single-face-raw.mp4
'
```

Notes:
- Change `--seed` to select a different deterministic scenario.
- `preview_episode.py` records 10 seconds at 10 fps by default.
- If the process aborts during shutdown, the MP4 is often already written; verify before re-running.

## 2) Trim to desired clip length

First 5 seconds:

```bash
ffmpeg -y -i sim/oracle-single-face-raw.mp4 -ss 0 -t 5 -an -c:v libx264 -pix_fmt yuv420p sim/oracle-single-face-5s.mp4
```

Last 5 seconds:

```bash
ffmpeg -y -ss 5 -t 5 -i sim/oracle-single-face-raw.mp4 -an -c:v libx264 -pix_fmt yuv420p sim/oracle-single-face-5s.mp4
```

## 3) Convert MP4 -> optimized GIF

Recommended quality/size balance:

```bash
ffmpeg -y -i sim/oracle-single-face-5s.mp4 \
  -filter_complex "fps=8,scale=400:-1:flags=lanczos,split[s0][s1];[s0]palettegen=stats_mode=full[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4" \
  sim/oracle-single-face-5s.gif
```

Higher quality (larger file):

```bash
ffmpeg -y -i sim/oracle-single-face-5s.mp4 \
  -filter_complex "fps=10,scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen=stats_mode=full[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3" \
  sim/oracle-single-face-5s.gif
```

## 4) Verify output properties

```bash
ffprobe -v error -show_entries stream=width,height,r_frame_rate,duration -of default=noprint_wrappers=1 sim/oracle-single-face-5s.mp4
ffprobe -v error -show_entries stream=width,height,r_frame_rate,duration -of default=noprint_wrappers=1 sim/oracle-single-face-5s.gif
ls -lh sim/oracle-single-face-5s.mp4 sim/oracle-single-face-5s.gif
```

## Quick parameter guide

- Scenario choice: `--seed <int>`
- Clip window: `-ss <start_sec> -t <duration_sec>`
- GIF smoothness: `fps=8` or `fps=10`
- GIF size: `scale=400:-1` (smaller) or `scale=480:-1` (larger)
