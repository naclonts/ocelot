#!/usr/bin/env python3
"""sim/merge_shards.py — merge parallel collection shards into one dataset.

Each shard is a directory produced by sim/data_gen/collect_data.py --shard N, containing:
    shard_N/
        episodes/ep_NNNNNN.h5
        metadata.json
        train.txt / val.txt / test.txt

This script hard-links all episode files into a single merged directory,
regenerates the train/val/test splits at scenario level, and writes a merged
metadata.json.

Usage:

    # Auto-discover shard_N/ subdirs under a parent directory:
    python3 sim/data_gen/merge_shards.py \\
        --parent /data/dataset \\
        --output /data/dataset/merged

    # Explicit shard directories:
    python3 sim/data_gen/merge_shards.py \\
        --shards /data/dataset/shard_0 /data/dataset/shard_1 \\
        --output /data/dataset/merged

Hard-links are used where possible (same filesystem) to avoid duplicating
data.  If the output is on a different filesystem, files are copied instead.

Exit codes:
    0 — merge completed successfully
    1 — one or more errors occurred
"""

import argparse
import json
import logging
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import h5py

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode(val) -> str:
    """Safely decode an h5py scalar string (bytes or str)."""
    if hasattr(val, "decode"):
        return val.decode("utf-8")
    return str(val)


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link src → dst; fall back to copy if cross-filesystem."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _discover_shards(parent: Path) -> list[Path]:
    """Return sorted shard_N/ subdirectories under parent."""
    pattern = re.compile(r"^shard_(\d+)$")
    shards = sorted(
        (d for d in parent.iterdir() if d.is_dir() and pattern.match(d.name)),
        key=lambda d: int(pattern.match(d.name).group(1)),
    )
    return shards


# ---------------------------------------------------------------------------
# Split writer (mirrored from collect_data.py)
# ---------------------------------------------------------------------------

def _write_splits(
    episode_ids: list[int],
    scenario_ids: list[str],
    out_dir: Path,
) -> None:
    """Write train.txt / val.txt / test.txt with an 80/10/10 scenario-level split."""
    groups: dict[str, list[int]] = defaultdict(list)
    for ep_id, sid in zip(episode_ids, scenario_ids):
        groups[sid].append(ep_id)

    sid_list = list(groups.keys())
    rng = random.Random(42)
    rng.shuffle(sid_list)

    n         = len(sid_list)
    train_ids = sid_list[:int(0.8 * n)]
    val_ids   = sid_list[int(0.8 * n):int(0.9 * n)]
    test_ids  = sid_list[int(0.9 * n):]

    for split_name, sids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        eps = [ep for sid in sids for ep in groups[sid]]
        (out_dir / f"{split_name}.txt").write_text(
            "\n".join(f"{e:06d}" for e in sorted(eps))
        )
        log.info("  %s.txt: %d episodes", split_name, len(eps))


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_shards(shard_dirs: list[Path], out_dir: Path) -> bool:
    """Merge shard_dirs into out_dir.

    Returns True on success, False if any error occurred.
    """
    out_episodes = out_dir / "episodes"
    out_episodes.mkdir(parents=True, exist_ok=True)

    all_ok          = True
    all_ep_ids:     list[int]  = []
    all_scenario_ids: list[str] = []
    all_label_counts: dict[str, int] = {}
    shard_metadata:  list[dict] = []
    n_linked = 0

    for shard_dir in shard_dirs:
        log.info("Processing shard: %s", shard_dir)
        episodes_dir = shard_dir / "episodes"
        if not episodes_dir.exists():
            log.error("  No episodes/ dir in %s — skipping", shard_dir)
            all_ok = False
            continue

        h5_files = sorted(episodes_dir.glob("ep_*.h5"))
        if not h5_files:
            log.warning("  No episode files in %s — skipping", shard_dir)
            continue

        # Load shard-level metadata if present.
        meta_path = shard_dir / "metadata.json"
        if meta_path.exists():
            try:
                shard_meta = json.loads(meta_path.read_text())
                shard_metadata.append(shard_meta)
            except Exception as exc:
                log.warning("  Could not read metadata.json in %s: %s", shard_dir, exc)

        for h5_path in h5_files:
            # Parse episode index from filename.
            m = re.match(r"ep_(\d+)\.h5$", h5_path.name)
            if not m:
                log.warning("  Unexpected filename %s — skipping", h5_path.name)
                continue
            ep_id = int(m.group(1))

            dst = out_episodes / h5_path.name
            if dst.exists():
                log.warning("  Episode %s already exists in output — skipping duplicate", h5_path.name)
                all_ok = False
                continue

            # Read scenario_id and label_key from the episode file.
            try:
                with h5py.File(h5_path, "r") as f:
                    label_key   = _decode(f["label_key"][()])
                    meta_str    = _decode(f["metadata"][()])
                    meta        = json.loads(meta_str)
                    scenario_id = meta.get("scenario_id", f"ep_{ep_id}")
            except Exception as exc:
                log.error("  Could not read %s: %s — skipping", h5_path.name, exc)
                all_ok = False
                continue

            _link_or_copy(h5_path, dst)
            all_ep_ids.append(ep_id)
            all_scenario_ids.append(scenario_id)
            all_label_counts[label_key] = all_label_counts.get(label_key, 0) + 1
            n_linked += 1

        log.info("  %d episodes processed", len(h5_files))

    if not all_ep_ids:
        log.error("No episodes merged — nothing to write.")
        return False

    log.info("Merged %d episodes total", n_linked)

    # ── Train / val / test splits ─────────────────────────────────────────
    log.info("Writing train/val/test splits...")
    _write_splits(all_ep_ids, all_scenario_ids, out_dir)

    # ── Merged metadata.json ──────────────────────────────────────────────
    seed_ranges = [m.get("seed_range", [0, 0]) for m in shard_metadata if "seed_range" in m]
    merged_seed_range: list[int] | None = None
    if seed_ranges:
        merged_seed_range = [
            min(r[0] for r in seed_ranges),
            max(r[1] for r in seed_ranges),
        ]

    camera_hz          = shard_metadata[0].get("camera_hz",          10)   if shard_metadata else 10
    episode_secs       = shard_metadata[0].get("episode_secs",       10.0) if shard_metadata else 10.0
    frames_per_episode = shard_metadata[0].get("frames_per_episode", 100)  if shard_metadata else 100
    image_shape        = shard_metadata[0].get("image_shape",   [224,224,3]) if shard_metadata else [224,224,3]

    merged_meta = {
        "schema_version":     "1.0",
        "collection_date":    datetime.now().isoformat(),
        "n_episodes":         len(all_ep_ids),
        "camera_hz":          camera_hz,
        "episode_secs":       episode_secs,
        "frames_per_episode": frames_per_episode,
        "image_shape":        image_shape,
        "label_counts":       all_label_counts,
        "n_shards":           len(shard_dirs),
        "shard_dirs":         [str(d) for d in shard_dirs],
    }
    if merged_seed_range is not None:
        merged_meta["seed_range"] = merged_seed_range

    (out_dir / "metadata.json").write_text(json.dumps(merged_meta, indent=2))
    log.info("Wrote metadata.json (%d episodes, %d label types)", len(all_ep_ids), len(all_label_counts))

    if all_ok:
        log.info("Merge complete — output: %s", out_dir)
    else:
        log.warning("Merge finished with errors — check log above")

    return all_ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge parallel collection shards into one dataset.",
    )
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--shards", nargs="+", metavar="DIR",
        help="Explicit list of shard directories to merge.",
    )
    src_group.add_argument(
        "--parent", metavar="DIR",
        help="Parent directory containing shard_N/ subdirectories to auto-discover.",
    )
    parser.add_argument(
        "--output", required=True, metavar="DIR",
        help="Output directory for the merged dataset.",
    )
    args = parser.parse_args()

    if args.shards:
        shard_dirs = [Path(d) for d in args.shards]
    else:
        parent = Path(args.parent)
        shard_dirs = _discover_shards(parent)
        if not shard_dirs:
            log.error("No shard_N/ directories found under %s", parent)
            sys.exit(1)
        log.info("Discovered %d shards: %s", len(shard_dirs), [d.name for d in shard_dirs])

    out_dir = Path(args.output)
    ok = merge_shards(shard_dirs, out_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
