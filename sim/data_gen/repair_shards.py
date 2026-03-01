#!/usr/bin/env python3
"""sim/data_gen/repair_shards.py — fill in missing metadata.json and split txts.

For each shard_* directory that has episode H5 files but is missing any of:
    metadata.json, train.txt, val.txt, test.txt

…reads the H5 files to reconstruct label_counts, seed_range, and scenario
groupings, then writes the missing files.

Shards with no H5 files are skipped silently.

Usage (from project root, with .venv active):
    python3 sim/data_gen/repair_shards.py [--dataset sim/dataset] [--dry-run]
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import h5py

# Constants matching collect_data.py
CAMERA_HZ          = 10
EPISODE_SECS       = 10.0
FRAMES_PER_EPISODE = 100
IMAGE_SHAPE        = [224, 224, 3]
PERTURB_INTERVAL   = 30
PERTURB_RANGE      = 0.45
PERTURB_DURATION   = 8


def _write_splits(episode_ids: list[int], scenario_ids: list[str], out_dir: Path, dry_run: bool) -> None:
    """80/10/10 train/val/test split grouped by scenario_id (same logic as collect_data.py)."""
    groups: dict[str, list[int]] = defaultdict(list)
    for ep_id, sid in zip(episode_ids, scenario_ids):
        groups[sid].append(ep_id)

    all_scenario_ids = list(groups.keys())
    rng = random.Random(42)
    rng.shuffle(all_scenario_ids)

    n = len(all_scenario_ids)
    train_sids = all_scenario_ids[:int(0.8 * n)]
    val_sids   = all_scenario_ids[int(0.8 * n):int(0.9 * n)]
    test_sids  = all_scenario_ids[int(0.9 * n):]

    for split_name, sids in [("train", train_sids), ("val", val_sids), ("test", test_sids)]:
        path = out_dir / f"{split_name}.txt"
        if path.exists():
            continue
        eps = [ep for sid in sids for ep in groups[sid]]
        content = "\n".join(f"{e:06d}" for e in sorted(eps))
        print(f"  writing {path.name} ({len(eps)} episodes)")
        if not dry_run:
            path.write_text(content)


def repair_shard(shard_dir: Path, dry_run: bool) -> None:
    episodes_dir = shard_dir / "episodes"
    h5_files = sorted(episodes_dir.glob("ep_*.h5")) if episodes_dir.exists() else []

    if not h5_files:
        return  # nothing to do

    need_meta   = not (shard_dir / "metadata.json").exists()
    need_splits = any(not (shard_dir / f"{s}.txt").exists() for s in ("train", "val", "test"))

    if not need_meta and not need_splits:
        return

    print(f"{shard_dir.name}: {len(h5_files)} episodes — regenerating missing files")

    # Read all H5 files
    episode_ids:   list[int]   = []
    scenario_ids:  list[str]   = []
    label_counts:  dict[str, int] = {}
    seeds:         list[int]   = []

    for h5_path in h5_files:
        ep_id = int(h5_path.stem[3:])  # "ep_000800" → 800
        try:
            with h5py.File(h5_path, "r") as f:
                label_key   = f["label_key"][()].decode()
                meta        = json.loads(f["metadata"][()])
        except Exception as e:
            print(f"  WARNING: could not read {h5_path.name}: {e} — skipping")
            continue

        episode_ids.append(ep_id)
        scenario_ids.append(meta["scenario_id"])
        seeds.append(meta["seed"])
        label_counts[label_key] = label_counts.get(label_key, 0) + 1

    if not episode_ids:
        print(f"  WARNING: no readable H5 files — skipping")
        return

    if need_meta:
        metadata = {
            "schema_version":     "1.0",
            "collection_date":    datetime.fromtimestamp(h5_files[0].stat().st_mtime).isoformat(),
            "n_episodes":         len(episode_ids),
            "camera_hz":          CAMERA_HZ,
            "episode_secs":       EPISODE_SECS,
            "frames_per_episode": FRAMES_PER_EPISODE,
            "image_shape":        IMAGE_SHAPE,
            "label_counts":       label_counts,
            "seed_range":         [min(seeds), max(seeds)],
            "perturb_interval":   PERTURB_INTERVAL,
            "perturb_range":      PERTURB_RANGE,
            "perturb_duration":   PERTURB_DURATION,
            "git_hash":           "unknown",
        }
        meta_path = shard_dir / "metadata.json"
        print(f"  writing metadata.json ({len(episode_ids)} episodes, seeds {min(seeds)}–{max(seeds)})")
        if not dry_run:
            meta_path.write_text(json.dumps(metadata, indent=2))

    if need_splits:
        _write_splits(episode_ids, scenario_ids, shard_dir, dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair missing shard metadata and split files.")
    parser.add_argument("--dataset", default="sim/dataset", help="Path to dataset root (default: sim/dataset)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be written without writing")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        print(f"ERROR: dataset dir not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    shard_dirs = sorted(dataset_dir.glob("shard_*"), key=lambda p: int(p.name.split("_")[1]))
    if not shard_dirs:
        print("No shard_* directories found.")
        return

    if args.dry_run:
        print("DRY RUN — no files will be written\n")

    for shard_dir in shard_dirs:
        repair_shard(shard_dir, args.dry_run)

    print("Done.")


if __name__ == "__main__":
    main()
