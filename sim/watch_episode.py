"""Watch a dataset episode with live model predictions overlaid.

For each frame the ONNX model is run and both ground-truth and predicted
pan/tilt velocities are shown side by side.  Velocity bars give an intuitive
sense of agreement.

Usage:
    source .venv/bin/activate

    python3 sim/watch_episode.py 000042 \\
        --model runs/v0.0-smoke/best.onnx \\
        --token-cache runs/v0.0-smoke/best_tokens.json

    # Save to MP4 instead of displaying:
    python3 sim/watch_episode.py 000042 \\
        --model runs/v0.0-smoke/best.onnx \\
        --save /tmp/ep42_watch.mp4

    # Slow it down:
    python3 sim/watch_episode.py 000042 --model ... --fps 3

Controls (interactive mode):
    SPACE       pause / resume
    any key     step one frame (while paused)
    q / ESC     quit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# ImageNet normalisation — must match train/dataset.py
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_SCALE   = 3          # 224 → 672 px display
_IMG_PX  = 224 * _SCALE
_BAR_H   = 80         # info panel height in pixels
_MAX_VEL = 2.0        # velocity bar full-scale (rad/s)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _preprocess(frame_hwc: np.ndarray) -> np.ndarray:
    """uint8 (224,224,3) RGB → float32 (1,3,224,224) ImageNet-normalised."""
    f = frame_hwc.astype(np.float32) / 255.0
    f = (f - _MEAN) / _STD
    return np.ascontiguousarray(f.transpose(2, 0, 1)[np.newaxis])  # (1,3,224,224)


def _vel_bar(canvas: np.ndarray, x: int, y: int, w: int, h: int,
             gt: float, pred: float, label: str) -> None:
    """Draw a horizontal velocity bar for one axis.

    Centre line = zero velocity.  Green bar = GT, orange bar = prediction.
    """
    cx = x + w // 2
    scale = (w // 2) / _MAX_VEL

    # Background
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (40, 40, 40), -1)
    # Centre line
    cv2.line(canvas, (cx, y), (cx, y + h), (100, 100, 100), 1)

    def _bar(val: float, color: tuple, thick: int) -> None:
        px = int(np.clip(val * scale, -w // 2, w // 2))
        x0, x1 = (cx + px, cx) if px < 0 else (cx, cx + px)
        mid = y + h // 2
        cv2.rectangle(canvas, (x0, mid - thick), (x1, mid + thick), color, -1)

    _bar(gt,   (60, 180, 60),   6)   # green  — ground truth
    _bar(pred, (30, 140, 210),  3)   # blue   — prediction

    # Label + values
    txt = f"{label}  gt={gt:+.3f}  pred={pred:+.3f}"
    cv2.putText(canvas, txt, (x + 4, y + h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)


def _find_episode(ep_id: str, dataset_dir: Path) -> Path:
    direct = dataset_dir / "episodes" / f"ep_{ep_id}.h5"
    if direct.exists():
        return direct
    for h5 in sorted(dataset_dir.glob(f"shard_*/episodes/ep_{ep_id}.h5")):
        return h5
    raise FileNotFoundError(f"ep_{ep_id}.h5 not found under {dataset_dir}")


# ---------------------------------------------------------------------------
# Main watch loop
# ---------------------------------------------------------------------------

def watch(ep_path: Path, model_path: Path, token_cache_path: Path | None,
          fps: int, save: Path | None) -> None:
    import onnxruntime as ort

    # Load model
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(model_path), providers=providers)
    provider = sess.get_providers()[0]
    print(f"ONNX provider : {provider}")

    # Load episode
    with h5py.File(ep_path, "r") as f:
        frames   = f["frames"][:]       # (N,224,224,3) uint8 RGB
        pan_gt   = f["pan_vel"][:]      # (N,) float32
        tilt_gt  = f["tilt_vel"][:]     # (N,) float32
        cmd_raw  = f["cmd"][()]
    cmd = cmd_raw.decode() if isinstance(cmd_raw, bytes) else str(cmd_raw)
    n = len(frames)

    # Build token arrays
    if token_cache_path is not None and token_cache_path.exists():
        cache = json.loads(token_cache_path.read_text())
        # pick exact match or first entry
        key = cmd if cmd in cache else next(iter(cache))
        if key != cmd:
            print(f"Command {cmd!r} not in token cache; using {key!r}")
        entry = cache[key]
        input_ids      = np.array([entry["input_ids"]],      dtype=np.int64)
        attention_mask = np.array([entry["attention_mask"]], dtype=np.int64)
    else:
        from transformers import CLIPTokenizerFast
        from train.model import VLAModel
        tok = CLIPTokenizerFast.from_pretrained(VLAModel.CLIP_ID)
        enc = tok(cmd, return_tensors="np", padding="max_length",
                  truncation=True, max_length=77)
        input_ids      = enc["input_ids"].astype(np.int64)
        attention_mask = enc["attention_mask"].astype(np.int64)

    print(f"Episode       : {ep_path.stem}")
    print(f"Command       : {cmd!r}")
    print(f"Frames        : {n} @ {fps} fps  ({n/fps:.1f} s)")
    print()

    # Run all inference up front (fast enough; avoids per-frame latency)
    print("Running inference …", end="", flush=True)
    frames_nchw = np.stack([_preprocess(frames[i])[0] for i in range(n)])  # (N,3,224,224)
    ids_rep  = np.repeat(input_ids,      n, axis=0)
    mask_rep = np.repeat(attention_mask, n, axis=0)
    preds = sess.run(
        ["actions"],
        {"frames": frames_nchw, "input_ids": ids_rep, "attention_mask": mask_rep},
    )[0]  # (N, 2)
    pan_pred  = preds[:, 0]
    tilt_pred = preds[:, 1]
    print(" done.")

    mse = float(np.mean((preds - np.stack([pan_gt, tilt_gt], axis=1)) ** 2))
    print(f"Episode MSE   : {mse:.5f}")
    print()
    if save is None:
        print("Controls: SPACE=pause  q/ESC=quit  any other key=step")
        print()

    # Canvas layout
    out_w = _IMG_PX
    out_h = _IMG_PX + _BAR_H
    delay_ms = max(1, 1000 // fps)

    writer = None
    if save is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save), fourcc, fps, (out_w, out_h))

    paused = False
    i = 0
    while i < n:
        bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (_IMG_PX, _IMG_PX), interpolation=cv2.INTER_NEAREST)

        panel = np.zeros((_BAR_H, out_w, 3), dtype=np.uint8)

        # Progress bar along bottom edge of panel
        pct = (i + 1) / n
        cv2.rectangle(panel, (0, _BAR_H - 4), (int(out_w * pct), _BAR_H - 1),
                      (80, 80, 80), -1)

        # Frame counter + command
        header = f"frame {i+1:3d}/{n}   {cmd}"
        cv2.putText(panel, header, (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        # Velocity bars — each takes half the width
        half = out_w // 2
        bar_h = (_BAR_H - 22) // 2
        _vel_bar(panel,    0, 18,      half, bar_h, pan_gt[i],  pan_pred[i],  "pan ")
        _vel_bar(panel,    0, 18 + bar_h, half, bar_h, tilt_gt[i], tilt_pred[i], "tilt")

        # Legend (right side)
        cv2.rectangle(panel, (half + 8, 22), (half + 22, 30), (60, 180, 60), -1)
        cv2.putText(panel, "ground truth", (half + 26, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (half + 8, 36), (half + 22, 44), (30, 140, 210), -1)
        cv2.putText(panel, "model pred", (half + 26, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

        # Per-frame MSE badge
        frame_mse = float((pan_pred[i] - pan_gt[i])**2 + (tilt_pred[i] - tilt_gt[i])**2) / 2
        color = (60, 200, 60) if frame_mse < 0.05 else (30, 100, 220) if frame_mse < 0.20 else (40, 40, 200)
        cv2.putText(panel, f"MSE {frame_mse:.4f}", (half + 8, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        canvas = np.vstack([bgr, panel])

        if writer is not None:
            writer.write(canvas)
            i += 1
            continue

        cv2.imshow(ep_path.stem, canvas)
        key = cv2.waitKey(1 if paused else delay_ms) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
        elif paused and key != 0xFF:
            i += 1
        elif not paused:
            i += 1

    if writer is not None:
        writer.release()
        print(f"Saved → {save}")
    else:
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Watch a dataset episode with VLA predictions.")
    p.add_argument("episode_id", help="6-digit episode ID (e.g. 000042)")
    p.add_argument("--model", required=True, type=Path,
                   help="Path to ONNX model (e.g. runs/v0.0-smoke/best.onnx)")
    p.add_argument("--token-cache", type=Path, default=None,
                   help="Token cache JSON from export_onnx.py (avoids loading transformers)")
    p.add_argument("--dataset-dir", default="sim/dataset", type=Path)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--save", type=Path, default=None,
                   help="Save to MP4 instead of displaying")
    args = p.parse_args()

    ep_id = f"{int(args.episode_id):06d}"

    try:
        ep_path = _find_episode(ep_id, args.dataset_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    watch(ep_path, args.model, args.token_cache, args.fps, args.save)


if __name__ == "__main__":
    main()
