"""Play back a single episode from the sim dataset.

Usage:
    python3 sim/play_episode.py 012360
    python3 sim/play_episode.py 12360 --dataset-dir sim/dataset --fps 10
    python3 sim/play_episode.py 000042 --save /tmp/ep.mp4
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np


def find_episode(ep_id: str, dataset_dir: Path) -> Path:
    """Search shard_* dirs for ep_{ep_id}.h5, return path or raise."""
    # Direct (flat) layout
    direct = dataset_dir / "episodes" / f"ep_{ep_id}.h5"
    if direct.exists():
        return direct
    # Sharded layout
    for h5 in sorted(dataset_dir.glob(f"shard_*/episodes/ep_{ep_id}.h5")):
        return h5
    raise FileNotFoundError(
        f"Episode ep_{ep_id}.h5 not found under {dataset_dir}"
    )


def play(ep_path: Path, fps: int, save: Path | None) -> None:
    with h5py.File(ep_path, "r") as f:
        frames   = f["frames"][:]          # (N, 224, 224, 3) uint8 RGB
        pan_vel  = f["pan_vel"][:]         # (N,) float32
        tilt_vel = f["tilt_vel"][:]        # (N,) float32
        cmd      = f["cmd"][()].decode() if isinstance(f["cmd"][()], bytes) else str(f["cmd"][()])
        meta_raw = f["metadata"][()] if "metadata" in f else None

    n_frames = frames.shape[0]
    scenario_id = ""
    if meta_raw is not None:
        try:
            d = json.loads(meta_raw.decode() if isinstance(meta_raw, bytes) else meta_raw)
            scenario_id = d.get("scenario_id", "")
        except Exception:
            pass

    print(f"Episode : {ep_path.stem}")
    print(f"Scenario: {scenario_id}")
    print(f"Command : {cmd!r}")
    print(f"Frames  : {n_frames} @ {fps} fps  ({n_frames/fps:.1f} s)")
    print()
    if save is None:
        print("Controls: SPACE=pause  q/ESC=quit  any other key=step")
        print()

    delay_ms = max(1, 1000 // fps)
    scale    = 3   # 224 → 672 px for comfortable viewing
    out_w, out_h = 224 * scale, 224 * scale + 40  # +40 for info bar

    writer = None
    if save is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save), fourcc, fps, (out_w, out_h))

    paused = False
    i = 0
    while i < n_frames:
        # Build display frame (RGB → BGR for OpenCV)
        bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (224 * scale, 224 * scale), interpolation=cv2.INTER_NEAREST)

        # Info bar
        bar = np.zeros((40, out_w, 3), dtype=np.uint8)
        pct = (i + 1) / n_frames
        cv2.rectangle(bar, (0, 36), (int(out_w * pct), 39), (80, 160, 80), -1)
        info = (f"frame {i+1:3d}/{n_frames}  "
                f"pan_vel={pan_vel[i]:+.3f}  tilt_vel={tilt_vel[i]:+.3f}  "
                f"{cmd}")
        cv2.putText(bar, info, (4, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)

        canvas = np.vstack([bgr, bar])

        if writer is not None:
            writer.write(canvas)
            i += 1
            continue

        cv2.imshow(ep_path.stem, canvas)
        key = cv2.waitKey(1 if paused else delay_ms) & 0xFF
        if key in (ord("q"), 27):   # q or ESC
            break
        elif key == ord(" "):
            paused = not paused
        elif paused and key != 0xFF:
            i += 1                  # single-step on any key while paused
        elif not paused:
            i += 1

    if writer is not None:
        writer.release()
        print(f"Saved → {save}")
    else:
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Play back a sim dataset episode.")
    parser.add_argument("episode_id", help="6-digit episode ID (e.g. 012360 or 12360)")
    parser.add_argument("--dataset-dir", default="sim/dataset", type=Path,
                        help="Dataset root (default: sim/dataset)")
    parser.add_argument("--fps", type=int, default=10,
                        help="Playback speed in frames/sec (default: 10)")
    parser.add_argument("--save", type=Path, default=None,
                        help="Save to MP4 instead of displaying (e.g. --save /tmp/ep.mp4)")
    args = parser.parse_args()

    ep_id = f"{int(args.episode_id):06d}"

    try:
        ep_path = find_episode(ep_id, args.dataset_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    play(ep_path, args.fps, args.save)


if __name__ == "__main__":
    main()
