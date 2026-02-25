#!/usr/bin/env python3
"""
Generate programmatic plain-color background textures (512×512 PNGs).

These serve as simple, clutter-free backgrounds that prevent the VLA from
using background texture as a tracking cue, and approximate real-world
deployment against plain walls.  Target ~30% of episodes use a plain background.

Usage:
    python3 sim/scenario_generator/generate_backgrounds.py \
        --out sim/assets/backgrounds/

The script also writes/updates backgrounds_manifest.json in the output directory,
preserving any existing photo-texture entries already in the manifest.

Requirements:
    pip install Pillow numpy   (both available in .venv)

Photo textures (indoor/outdoor) must be downloaded separately.
Recommended sources: Unsplash (free license) or the same gpt-image-1 pipeline
used for face textures.  Add them to sim/assets/backgrounds/ and register them
in backgrounds_manifest.json with appropriate tags.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


# 6 plain solid colors.  RGB tuples.
PLAIN_COLORS = [
    ("plain_white",       (255, 255, 255)),
    ("plain_light_gray",  (210, 210, 210)),
    ("plain_dark_gray",   (80,  80,  80)),
    ("plain_beige",       (230, 215, 195)),
    ("plain_off_white",   (245, 240, 230)),
    ("plain_blue_gray",   (180, 190, 205)),
]


def generate_plain_backgrounds(out_dir: Path) -> list[dict]:
    """Write plain-color PNGs and return manifest entries for them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, rgb in PLAIN_COLORS:
        arr = np.full((512, 512, 3), rgb, dtype=np.uint8)
        filename = f"{name}.png"
        Image.fromarray(arr).save(out_dir / filename)
        print(f"  Wrote {out_dir / filename}")
        entries.append({"id": name, "tags": ["plain"], "file": filename})
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Generate plain-color background textures for the scenario generator."
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("sim/assets/backgrounds"),
        help="Output directory (default: sim/assets/backgrounds/)",
    )
    args = parser.parse_args()

    print(f"Generating plain backgrounds → {args.out}/")
    plain_entries = generate_plain_backgrounds(args.out)

    # Merge with any existing manifest (preserve photo texture entries).
    manifest_path = args.out / "backgrounds_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        existing_ids = {e["id"] for e in existing}
        # Keep existing entries that are not plain colors (photos),
        # then append/replace the programmatically generated plain ones.
        photo_entries = [e for e in existing if e["id"] not in {p["id"] for p, _ in [(x, None) for x in plain_entries]}]
        # Rebuild: plain entries first (predictable order), then photos
        plain_ids = {e["id"] for e in plain_entries}
        photo_entries = [e for e in existing if e["id"] not in plain_ids]
        merged = plain_entries + photo_entries
    else:
        merged = plain_entries

    manifest_path.write_text(json.dumps(merged, indent=2) + "\n")
    print(f"\nWrote {len(merged)} entries → {manifest_path}")
    print(f"  Plain: {sum(1 for e in merged if 'plain' in e['tags'])}")
    print(f"  Photo: {sum(1 for e in merged if 'plain' not in e['tags'])}")
    print()
    print("To add photo textures:")
    print("  1. Place .jpg/.png files in", args.out)
    print("  2. Add entries to", manifest_path)
    print('     e.g.: {"id": "indoor_office", "tags": ["indoor", "busy"], "file": "indoor_office.jpg"}')


if __name__ == "__main__":
    main()
