"""Experiment C — Rate limiter impact analysis.

Replays val-episode model predictions through the vla_node rate limiter
(max_accel=1.5) vs raw (no limiter), measuring:
  - MSE of rate-limited vs raw predictions against GT
  - Lag at sharp transitions (frames until output matches GT direction change)
  - Overshoot: frames where rate-limited output has opposite sign to GT

Usage:
    source .venv/bin/activate
    python3 train/experiment_c_ratelimiter.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import h5py
import numpy as np

from train.dataset import IMAGENET_MEAN, IMAGENET_STD, OcelotDataset


def _preprocess(frames_nhwc: np.ndarray) -> np.ndarray:
    f = frames_nhwc.astype(np.float32) / 255.0
    f = (f - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(f.transpose(0, 3, 1, 2))


def _apply_rate_limiter(preds: np.ndarray, max_accel: float, fps: float) -> np.ndarray:
    """Simulate vla_node rate limiter on a (T, 2) prediction sequence.

    Returns (T, 2) rate-limited output.
    """
    max_delta = max_accel / fps
    out = np.zeros_like(preds)
    prev = np.zeros(2)
    for t in range(len(preds)):
        clamped = np.clip(preds[t], prev - max_delta, prev + max_delta)
        out[t] = clamped
        prev = clamped
    return out


def _detect_transitions(gt: np.ndarray, threshold: float = 0.05) -> list[int]:
    """Find frames where GT velocity crosses zero (sign change) with magnitude > threshold."""
    transitions = []
    for t in range(1, len(gt)):
        if gt[t - 1] * gt[t] < 0 and (abs(gt[t - 1]) > threshold or abs(gt[t]) > threshold):
            transitions.append(t)
    return transitions


def _measure_lag_at_transition(gt: np.ndarray, output: np.ndarray, t_cross: int) -> int:
    """Count frames after a GT sign change until the output matches the new GT sign."""
    new_sign = np.sign(gt[t_cross])
    if new_sign == 0:
        return 0
    for dt in range(0, min(20, len(gt) - t_cross)):
        if np.sign(output[t_cross + dt]) == new_sign:
            return dt
    return 20  # cap


def main():
    import onnxruntime as ort

    model_path = _root / "runs" / "v0.1.0" / "best.onnx"
    dataset_dir = _root / "sim" / "dataset"
    token_cache_path = _root / "runs" / "v0.1.0" / "best_tokens.json"

    # Load model
    print(f"Loading {model_path} ...")
    sess = ort.InferenceSession(
        str(model_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    print(f"Provider: {sess.get_providers()[0]}")

    # Load token cache
    token_cache = json.loads(token_cache_path.read_text())

    # Find val episodes
    split_files = OcelotDataset._find_split_files(dataset_dir, "val")
    episode_paths: list[tuple[str, Path]] = []
    for sf in sorted(split_files):
        shard_dir = sf.parent
        ep_ids = [line.strip() for line in sf.read_text().splitlines() if line.strip()]
        for ep_id in ep_ids:
            h5_path = shard_dir / "episodes" / f"ep_{ep_id}.h5"
            if h5_path.exists():
                episode_paths.append((ep_id, h5_path))

    print(f"Found {len(episode_paths)} val episodes")

    # Subsample for speed — 200 episodes is plenty for statistics
    rng = np.random.default_rng(42)
    if len(episode_paths) > 200:
        idx = rng.choice(len(episode_paths), size=200, replace=False)
        episode_paths = [episode_paths[i] for i in sorted(idx)]
        print(f"Subsampled to {len(episode_paths)} episodes")

    MAX_VEL = 0.3
    MAX_ACCEL = 1.5
    FPS = 10.0

    # Accumulators
    raw_mse_sum, lim_mse_sum, n_frames_total = 0.0, 0.0, 0
    raw_lag_pan, lim_lag_pan = [], []
    raw_lag_tilt, lim_lag_tilt = [], []
    overshoot_raw_count, overshoot_lim_count = 0, 0
    overshoot_total = 0  # frames near transitions where overshoot is possible

    for ep_idx, (ep_id, h5_path) in enumerate(episode_paths):
        with h5py.File(h5_path, "r") as f:
            frames_nhwc = f["frames"][:]
            pan_gt = f["pan_vel"][:]
            tilt_gt = f["tilt_vel"][:]
            cmd_raw = f["cmd"][()]
        cmd = cmd_raw.decode("utf-8") if isinstance(cmd_raw, bytes) else str(cmd_raw)

        if cmd not in token_cache:
            continue

        tokens = token_cache[cmd]
        input_ids = np.array([tokens["input_ids"]], dtype=np.int64)
        attn_mask = np.array([tokens["attention_mask"]], dtype=np.int64)

        T = len(frames_nhwc)
        frames_nchw = _preprocess(frames_nhwc)

        # Run inference
        preds_list = []
        for start in range(0, T, 64):
            end = min(start + 64, T)
            fb = frames_nchw[start:end]
            B = len(fb)
            pred = sess.run(
                ["actions"],
                {
                    "frames": fb,
                    "input_ids": np.repeat(input_ids, B, 0),
                    "attention_mask": np.repeat(attn_mask, B, 0),
                },
            )[0]
            preds_list.append(pred)

        preds_raw = np.concatenate(preds_list, axis=0)  # (T, 2)
        gt = np.stack([pan_gt, tilt_gt], axis=1)  # (T, 2)

        # Clip to max_vel (both paths do this)
        preds_clipped = np.clip(preds_raw, -MAX_VEL, MAX_VEL)

        # Rate-limited path
        preds_limited = _apply_rate_limiter(preds_clipped, MAX_ACCEL, FPS)

        # MSE
        raw_mse = np.mean((preds_clipped - gt) ** 2)
        lim_mse = np.mean((preds_limited - gt) ** 2)
        raw_mse_sum += raw_mse * T
        lim_mse_sum += lim_mse * T
        n_frames_total += T

        # Transition analysis — per axis
        for axis, (raw_lags, lim_lags) in enumerate(
            [(raw_lag_pan, lim_lag_pan), (raw_lag_tilt, lim_lag_tilt)]
        ):
            gt_axis = gt[:, axis]
            transitions = _detect_transitions(gt_axis)
            for t_cross in transitions:
                raw_lags.append(
                    _measure_lag_at_transition(gt_axis, preds_clipped[:, axis], t_cross)
                )
                lim_lags.append(
                    _measure_lag_at_transition(gt_axis, preds_limited[:, axis], t_cross)
                )

                # Overshoot: in 5 frames after transition, does output have opposite sign to GT?
                window = min(5, T - t_cross)
                overshoot_total += window
                for dt in range(window):
                    gt_sign = np.sign(gt_axis[t_cross + dt])
                    if gt_sign != 0:
                        if np.sign(preds_clipped[t_cross + dt, axis]) == -gt_sign:
                            overshoot_raw_count += 1
                        if np.sign(preds_limited[t_cross + dt, axis]) == -gt_sign:
                            overshoot_lim_count += 1

        if (ep_idx + 1) % 50 == 0:
            print(f"  [{ep_idx + 1}/{len(episode_paths)}]")

    # Report
    print(f"\n{'=' * 60}")
    print(f"  Experiment C: Rate Limiter Impact (max_accel={MAX_ACCEL})")
    print(f"{'=' * 60}")
    print(f"  Episodes evaluated:  {len(episode_paths)}")
    print(f"  Total frames:        {n_frames_total}")
    print()
    print(f"  MSE (raw, no limiter):   {raw_mse_sum / n_frames_total:.6f}")
    print(f"  MSE (rate-limited):      {lim_mse_sum / n_frames_total:.6f}")
    print(f"  MSE increase from limiter: {(lim_mse_sum - raw_mse_sum) / raw_mse_sum * 100:+.1f}%")
    print()
    print("  Transition lag (frames until output matches new GT sign):")
    print(
        f"    Pan  — raw: mean={np.mean(raw_lag_pan):.2f}, "
        f"median={np.median(raw_lag_pan):.1f}  |  "
        f"limited: mean={np.mean(lim_lag_pan):.2f}, "
        f"median={np.median(lim_lag_pan):.1f}"
    )
    print(
        f"    Tilt — raw: mean={np.mean(raw_lag_tilt):.2f}, "
        f"median={np.median(raw_lag_tilt):.1f}  |  "
        f"limited: mean={np.mean(lim_lag_tilt):.2f}, "
        f"median={np.median(lim_lag_tilt):.1f}"
    )
    print()
    if overshoot_total > 0:
        print("  Overshoot (opposite-sign frames in 5-frame window after transitions):")
        print(
            f"    Raw:     {overshoot_raw_count}/{overshoot_total} "
            f"({100 * overshoot_raw_count / overshoot_total:.1f}%)"
        )
        print(
            f"    Limited: {overshoot_lim_count}/{overshoot_total} "
            f"({100 * overshoot_lim_count / overshoot_total:.1f}%)"
        )
    print(f"{'=' * 60}")

    # Save results as JSON for ticket
    results = {
        "experiment": "C",
        "description": "Rate limiter impact analysis",
        "params": {"max_accel": MAX_ACCEL, "max_vel": MAX_VEL, "fps": FPS},
        "n_episodes": len(episode_paths),
        "n_frames": n_frames_total,
        "mse_raw": raw_mse_sum / n_frames_total,
        "mse_limited": lim_mse_sum / n_frames_total,
        "mse_increase_pct": float((lim_mse_sum - raw_mse_sum) / raw_mse_sum * 100),
        "transition_lag": {
            "pan_raw_mean": float(np.mean(raw_lag_pan)),
            "pan_limited_mean": float(np.mean(lim_lag_pan)),
            "tilt_raw_mean": float(np.mean(raw_lag_tilt)),
            "tilt_limited_mean": float(np.mean(lim_lag_tilt)),
        },
        "overshoot": {
            "raw_count": overshoot_raw_count,
            "limited_count": overshoot_lim_count,
            "total_window_frames": overshoot_total,
            "raw_pct": float(100 * overshoot_raw_count / overshoot_total) if overshoot_total else 0,
            "limited_pct": (
                float(100 * overshoot_lim_count / overshoot_total) if overshoot_total else 0
            ),
        },
    }
    out_path = _root / "runs" / "v0.1.0" / "experiment_c_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
