#!/usr/bin/env python3
"""
Generate face textures via the OpenAI gpt-image-1 API.

Reads a face_descriptions.json file (produced by face_descriptions.py),
calls the image generation API for each entry, and saves transparent-background
PNG files suitable for use as Gazebo billboard textures.

Already-existing output files are skipped, so the script is safe to re-run
after interruptions.

Usage:
    python3 generate_face_images.py --input sim/faces/face_descriptions.json
    python3 generate_face_images.py --input sim/faces/face_descriptions.json \\
        --out sim/faces/ --quality high --size 1024x1024
    python3 generate_face_images.py --input ... --dry-run      # preview only
    python3 generate_face_images.py --input ... --start 10 --end 20  # subset

Environment:
    OPENAI_API_KEY  — required; loaded automatically from a .env file in the
                      project root (two directories above this script), or
                      falls back to the shell environment variable.
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------

# Project root is two levels up from this file (sim/scenario_generator/ → root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_dotenv() -> None:
    """
    Load variables from <project_root>/.env into os.environ if the file exists.
    Uses python-dotenv when available; falls back to a minimal parser so there
    is no hard dependency.
    """
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)  # shell env takes precedence
        return
    except ImportError:
        pass

    # Minimal fallback: parse KEY=VALUE lines (no multiline / substitution)
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)   # don't override shell env


# ---------------------------------------------------------------------------
# Prompt sanitization
# ---------------------------------------------------------------------------

# The face_descriptions generator embeds a background phrase that conflicts
# with the API's background=transparent parameter.  Strip it so the model
# doesn't fight itself trying to paint a gray background it's supposed to omit.
_BG_PHRASES = [
    "plain neutral gray background",
    "plain white background",
    "plain gray background",
    "neutral gray background",
    "white background",
    "gray background",
]


def _sanitize_prompt(prompt: str) -> str:
    """Remove background-color phrases that conflict with transparent mode."""
    for phrase in _BG_PHRASES:
        prompt = prompt.replace(", " + phrase, "")
        prompt = prompt.replace(phrase + ", ", "")
        prompt = prompt.replace(phrase, "")
    return prompt.strip(", ")


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------

def _generate_one(client, face_id: str, prompt: str, *, size: str, quality: str) -> bytes:
    """
    Call the API and return raw PNG bytes.
    Retries on rate-limit errors with exponential backoff (up to 5 attempts).
    """
    clean_prompt = _sanitize_prompt(prompt)

    delay = 5.0
    for attempt in range(1, 6):
        try:
            response = client.images.generate(
                model="gpt-image-1",
                prompt=clean_prompt,
                n=1,
                size=size,
                quality=quality,
                background="transparent",
                output_format="png",
            )
            b64_data = response.data[0].b64_json
            return base64.b64decode(b64_data)

        except Exception as exc:
            exc_type = type(exc).__name__
            is_rate_limit = "RateLimitError" in exc_type or "rate_limit" in str(exc).lower()
            is_last = attempt == 5

            if is_last:
                raise

            if is_rate_limit:
                wait = delay * (2 ** (attempt - 1))   # 5, 10, 20, 40 s
                print(f"  [{face_id}] rate limit (attempt {attempt}/5) — waiting {wait:.0f}s")
            else:
                wait = delay
                print(f"  [{face_id}] error: {exc_type}: {exc} (attempt {attempt}/5) — retrying in {wait:.0f}s")
            time.sleep(wait)

    raise RuntimeError("unreachable")  # loop above always raises on attempt 5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate face images with gpt-image-1 (transparent background)."
    )
    parser.add_argument(
        "--input", "-i", required=True, type=Path,
        help="Path to face_descriptions.json",
    )
    parser.add_argument(
        "--out", "-o", type=Path, default=None,
        help="Output directory for PNGs (default: same directory as --input)",
    )
    parser.add_argument(
        "--quality", choices=["low", "medium", "high"], default="high",
        help="Image quality (default: high)",
    )
    parser.add_argument(
        "--size", default="1024x1024",
        choices=["1024x1024", "1536x1024", "1024x1536", "auto"],
        help="Image size (default: 1024x1024)",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds to wait between API calls (default: 1.0)",
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=None,
        help="Maximum number of images to generate (from the start of the file, or from --start)",
    )
    parser.add_argument(
        "--start", type=int, default=None,
        help="0-based index of first face to process (inclusive)",
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="0-based index of last face to process (exclusive)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be generated without calling the API",
    )
    args = parser.parse_args()

    # ---- Validate input file -----------------------------------------------
    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    with open(args.input) as f:
        faces = json.load(f)

    if not isinstance(faces, list) or not faces:
        print("ERROR: JSON must be a non-empty list of face descriptions.", file=sys.stderr)
        sys.exit(1)

    # ---- Output directory ---------------------------------------------------
    out_dir = args.out if args.out is not None else args.input.parent
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Subset selection ---------------------------------------------------
    start = args.start if args.start is not None else 0
    end   = args.end   if args.end   is not None else len(faces)
    subset = faces[start:end]
    if args.limit is not None:
        subset = subset[:args.limit]

    if not subset:
        print("Nothing to generate (empty subset).")
        sys.exit(0)

    # ---- Dry-run ------------------------------------------------------------
    if args.dry_run:
        limit_str = f", limit={args.limit}" if args.limit is not None else ""
        print(f"Dry run — {len(subset)} face(s), indices [{start}, {end}){limit_str}")
        print(f"  out_dir : {out_dir}")
        print(f"  quality : {args.quality}  size: {args.size}")
        print()
        for face in subset:
            out_path   = out_dir / f"{face['face_id']}.png"
            face_size  = "1024x1536" if face.get("crop_level") == "waist_up" else args.size
            status     = "EXISTS" if out_path.exists() else "to generate"
            clean      = _sanitize_prompt(face["prompt"])
            print(f"  [{face['face_id']}] {face_size}  {status}")
            print(f"    prompt: {clean[:120]}{'...' if len(clean) > 120 else ''}")
        return

    # ---- API client ---------------------------------------------------------
    _load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            f"ERROR: OPENAI_API_KEY not set.\n"
            f"  Add it to {_PROJECT_ROOT / '.env'}  (OPENAI_API_KEY=sk-...)\n"
            f"  or export it in your shell.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed.  Run: pip install openai", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # ---- Generation loop ----------------------------------------------------
    total = len(subset)
    skipped = 0
    generated = 0
    failed = []

    print(f"Generating {total} face image(s) → {out_dir}")
    print(f"  quality={args.quality}  size={args.size} (waist_up→1024x1536)  delay={args.delay}s")
    print()

    for idx, face in enumerate(subset, start=1):
        face_id    = face["face_id"]
        prompt     = face["prompt"]
        crop_level = face.get("crop_level", "")
        size       = "1024x1536" if crop_level == "waist_up" else args.size
        out_path   = out_dir / f"{face_id}.png"

        prefix = f"[{idx:>{len(str(total))}}/{total}] {face_id}"

        if out_path.exists():
            print(f"{prefix}  SKIP (already exists)")
            skipped += 1
            continue

        print(f"{prefix} [{size}]  generating...", end="", flush=True)
        t0 = time.monotonic()

        try:
            png_bytes = _generate_one(
                client, face_id, prompt,
                size=size, quality=args.quality,
            )
        except Exception as exc:
            print(f"\n  ERROR: {exc}")
            failed.append(face_id)
            continue

        out_path.write_bytes(png_bytes)
        elapsed = time.monotonic() - t0
        print(f"  done ({elapsed:.1f}s, {len(png_bytes)//1024} KB)")
        generated += 1

        if idx < total and args.delay > 0:
            time.sleep(args.delay)

    # ---- Summary ------------------------------------------------------------
    print()
    print(f"Done.  generated={generated}  skipped={skipped}  failed={len(failed)}")
    if failed:
        print(f"  Failed face IDs: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
