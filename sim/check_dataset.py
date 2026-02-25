#!/usr/bin/env python3
"""sim/check_dataset.py — dataset quality check.

Reads a collected dataset directory and verifies integrity, label distribution,
velocity variance, frame shapes, and scenario-id uniqueness.

Usage:
    python3 sim/check_dataset.py --dataset /tmp/dataset_test
    python3 sim/check_dataset.py --dataset /tmp/dataset_100 --sample_frames 10

Exit codes:
    0  — all checks passed
    1  — one or more checks failed
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

FRAMES_PER_EPISODE  = 100
FRAME_SHAPE         = (224, 224, 3)   # (H, W, C)
PAN_VEL_STD_MIN     = 0.01            # oracle must be commanding real motion (not all-zero)
LABEL_PERCENT_WARN  = 5.0             # warn if any label is below this share


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode(val) -> str:
    """Safely decode an h5py scalar string (bytes or str)."""
    if hasattr(val, "decode"):
        return val.decode("utf-8")
    return str(val)


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------

def check_dataset(dataset_dir: Path, sample_frames: int = 0) -> bool:
    """Run all quality checks on dataset_dir.

    Returns True if all checks pass, False otherwise.
    """
    episodes_dir = dataset_dir / "episodes"
    if not episodes_dir.exists():
        print(f"[FAIL] episodes/ directory not found in {dataset_dir}")
        return False

    h5_files = sorted(episodes_dir.glob("ep_*.h5"))
    n = len(h5_files)
    if n == 0:
        print("[FAIL] No episode files found.")
        return False
    print(f"[OK]  {n} episode files found")

    # ── Per-episode reads ─────────────────────────────────────────────────
    all_pass    = True
    frame_counts: "list[int]"      = []
    label_keys:   "list[str]"      = []
    scenario_ids: "list[str]"      = []
    all_pan_vels: "list[float]"    = []
    all_tilt_vels: "list[float]"   = []
    sample_pool:  "list[tuple]"    = []  # (Path, n_frames) for visual sampling

    for p in h5_files:
        try:
            with h5py.File(p, "r") as f:
                frames_shape = f["frames"].shape
                pan_vel      = f["pan_vel"][:]
                tilt_vel     = f["tilt_vel"][:]
                label_key    = _decode(f["label_key"][()])
                meta_str     = _decode(f["metadata"][()])
                metadata     = json.loads(meta_str)
                scenario_id  = metadata.get("scenario_id", "")

            # Frame shape (H, W, C)
            if frames_shape[1:] != FRAME_SHAPE:
                print(
                    f"[FAIL] {p.name}: unexpected frame shape {frames_shape[1:]} "
                    f"(expected {FRAME_SHAPE})"
                )
                all_pass = False

            frame_counts.append(frames_shape[0])
            label_keys.append(label_key)
            scenario_ids.append(scenario_id)
            all_pan_vels.extend(pan_vel.tolist())
            all_tilt_vels.extend(tilt_vel.tolist())

            if sample_frames > 0:
                sample_pool.append((p, frames_shape[0]))

        except Exception as exc:
            print(f"[FAIL] {p.name}: could not read: {exc}")
            all_pass = False

    if not frame_counts:
        print("[FAIL] No readable episodes.")
        return False

    # ── Frame count ───────────────────────────────────────────────────────
    wrong = [c for c in frame_counts if c != FRAMES_PER_EPISODE]
    if wrong:
        print(
            f"[FAIL] {len(wrong)} episodes have wrong frame count "
            f"(expected {FRAMES_PER_EPISODE}): {wrong[:5]}"
        )
        all_pass = False
    else:
        print(
            f"[OK]  All files readable; "
            f"min/max frame counts: {min(frame_counts)}/{max(frame_counts)}"
        )

    # ── Label distribution ────────────────────────────────────────────────
    counts = Counter(label_keys)
    total  = len(label_keys)
    print("  Label distribution:")
    for key, cnt in sorted(counts.items()):
        pct  = 100.0 * cnt / total if total else 0.0
        flag = "[WARN]" if pct < LABEL_PERCENT_WARN else "      "
        print(f"      {flag}  {key:<22s} {cnt:>4d}  ({pct:.1f}%)")
        if pct < LABEL_PERCENT_WARN:
            print(f"             ^ below {LABEL_PERCENT_WARN:.0f}% threshold")

    # ── Velocity stats ────────────────────────────────────────────────────
    pan_arr  = np.array(all_pan_vels,  dtype=np.float32)
    til_arr  = np.array(all_tilt_vels, dtype=np.float32)
    print(
        f"  pan_vel  — mean: {pan_arr.mean():+.3f}  std: {pan_arr.std():.3f}  "
        f"min: {pan_arr.min():.3f}  max: {pan_arr.max():.3f}"
    )
    print(
        f"  tilt_vel — mean: {til_arr.mean():+.3f}  std: {til_arr.std():.3f}  "
        f"min: {til_arr.min():.3f}  max: {til_arr.max():.3f}"
    )
    if pan_arr.std() < PAN_VEL_STD_MIN:
        print(
            f"[FAIL] pan_vel std {pan_arr.std():.3f} < {PAN_VEL_STD_MIN} "
            "(oracle may be idling — check scene setup or warmup duration)"
        )
        all_pass = False
    else:
        print(f"[OK]  pan_vel std > {PAN_VEL_STD_MIN}")

    # ── Duplicate scenario_ids ────────────────────────────────────────────
    sid_counts = Counter(s for s in scenario_ids if s)
    dups       = {sid: cnt for sid, cnt in sid_counts.items() if cnt > 1}
    if dups:
        print(f"[WARN] {len(dups)} duplicate scenario_ids (expected only if seeds repeat)")
    else:
        print("[OK]  No duplicate scenario_ids")

    # ── Optional sample frames ────────────────────────────────────────────
    if sample_frames > 0 and sample_pool:
        import cv2
        out_dir = Path("/tmp/dataset_samples")
        out_dir.mkdir(exist_ok=True)
        rng     = random.Random(0)
        chosen  = rng.sample(sample_pool, min(sample_frames, len(sample_pool)))
        for i, (p, n_frames) in enumerate(chosen):
            fi = rng.randint(0, n_frames - 1)
            with h5py.File(p, "r") as f:
                frame = f["frames"][fi]          # (224, 224, 3) uint8 RGB
            bgr      = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out_path = out_dir / f"sample_{i:03d}_{p.stem}_f{fi:03d}.png"
            cv2.imwrite(str(out_path), bgr)
            print(f"  Saved sample: {out_path}")
        print(f"[OK]  {len(chosen)} sample frames saved to {out_dir}")

    # ── Summary ───────────────────────────────────────────────────────────
    if all_pass:
        print("\n[OK]  All checks passed.")
    else:
        print("\n[FAIL] Some checks failed — see above.")

    return all_pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset quality check."
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to the dataset directory (contains episodes/, metadata.json, etc.).",
    )
    parser.add_argument(
        "--sample_frames", type=int, default=0,
        help=(
            "Number of random frames to dump as PNG for visual inspection. "
            "Saved to /tmp/dataset_samples/.  Default: 0 (disabled)."
        ),
    )
    args = parser.parse_args()

    ok = check_dataset(Path(args.dataset), sample_frames=args.sample_frames)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
