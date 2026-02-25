#!/usr/bin/env python3
"""
Generate face textures via AI image generation APIs.

Reads a face_descriptions.json file (produced by face_descriptions.py),
calls the image generation API for each entry, and saves transparent-background
PNG files suitable for use as Gazebo billboard textures.

Already-existing output files are skipped, so the script is safe to re-run
after interruptions.

Usage:
    python3 generate_face_images.py --input sim/faces/face_descriptions.json
    python3 generate_face_images.py --input sim/faces/face_descriptions.json \\
        --model gpt-image-1 --quality high --size 1024x1024
    python3 generate_face_images.py --input sim/faces/face_descriptions.json \\
        --model runware-layerdiffuse
    python3 generate_face_images.py --input ... --dry-run      # preview only
    python3 generate_face_images.py --input ... --start 10 --end 20  # subset

Models (all produce native transparent-background RGBA PNG):
    gpt-image-1          — default; OpenAI; ~$0.05–0.25/image (quality: low/med/high)
    gpt-image-1.5        — OpenAI updated variant; same pricing tiers
    gpt-image-1-mini     — OpenAI cheaper variant; ~$0.011–0.016/image
    runware-layerdiffuse — Runware FLUX Dev + LayerDiffuse; ~$0.003–0.01/image;
                           alpha channel baked into diffusion (best hair/edge quality)
                           Docs: https://runware.ai/docs/image-inference/api-reference

Environment:
    OPENAI_API_KEY    — required for gpt-image-* models
    RUNWARE_API_KEY   — required for runware-layerdiffuse
                        Sign up at https://runware.ai (new accounts get $10 free)
    Both are loaded automatically from a .env file in the project root
    (two directories above this script), or from the shell environment.
"""

import argparse
import base64
import json
import os
import sys
import time
import uuid as _uuid_mod
from pathlib import Path

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

_OPENAI_MODELS  = frozenset({"gpt-image-1", "gpt-image-1.5", "gpt-image-1-mini"})
_RUNWARE_MODELS = frozenset({"runware-layerdiffuse"})
_ALL_MODELS     = sorted(_OPENAI_MODELS | _RUNWARE_MODELS)

# Runware internal model ID for FLUX Dev (the only backend that supports LayerDiffuse)
_RUNWARE_INTERNAL_MODEL = "runware:101@1"
_RUNWARE_API_URL        = "https://api.runware.ai/v1"

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
# Size parsing
# ---------------------------------------------------------------------------

def _parse_wh(size: str) -> tuple[int, int]:
    """Parse '1024x1024' → (1024, 1024).  'auto' maps to (1024, 1024)."""
    if size == "auto":
        return 1024, 1024
    w, h = size.split("x")
    return int(w), int(h)


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

def _generate_openai(client, face_id: str, prompt: str, *, model: str, size: str, quality: str) -> bytes:
    """
    Call the OpenAI image generation API and return raw PNG bytes.
    Retries on rate-limit errors with exponential backoff (up to 5 attempts).
    """
    clean_prompt = _sanitize_prompt(prompt)

    delay = 5.0
    for attempt in range(1, 6):
        try:
            response = client.images.generate(
                model=model,
                prompt=clean_prompt,
                n=1,
                size=size,
                quality=quality,
                background="transparent",
                output_format="png",
            )
            return base64.b64decode(response.data[0].b64_json)

        except Exception as exc:
            exc_type = type(exc).__name__
            is_rate_limit = "RateLimitError" in exc_type or "rate_limit" in str(exc).lower()
            if attempt == 5:
                raise
            wait = delay * (2 ** (attempt - 1)) if is_rate_limit else delay
            label = "rate limit" if is_rate_limit else f"{exc_type}: {exc}"
            print(f"  [{face_id}] {label} (attempt {attempt}/5) — waiting {wait:.0f}s")
            time.sleep(wait)

    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Runware backend (FLUX Dev + LayerDiffuse)
# ---------------------------------------------------------------------------

def _generate_runware(api_key: str, face_id: str, prompt: str, *, size: str) -> bytes:
    """
    Call the Runware LayerDiffuse API and return raw PNG bytes.

    LayerDiffuse bakes the alpha channel into the diffusion process itself
    (latent transparency), giving cleaner edges on hair and fine detail
    boundaries than any post-processing removal approach.

    Docs: https://runware.ai/docs/image-inference/api-reference
    """
    try:
        import requests as _requests
    except ImportError:
        print("ERROR: requests package not installed.  Run: pip install requests", file=sys.stderr)
        sys.exit(1)

    clean_prompt = _sanitize_prompt(prompt)
    w, h = _parse_wh(size)

    payload = [{
        "taskType":       "imageInference",
        "taskUUID":       str(_uuid_mod.uuid4()),
        "positivePrompt": clean_prompt + ", isolated subject, no background, cutout",
        "model":          _RUNWARE_INTERNAL_MODEL,
        "width":          w,
        "height":         h,
        "numberResults":  1,
        "outputType":     ["base64Data"],
        "outputFormat":   "PNG",
        "layerDiffuse":   True,
    }]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    delay = 5.0
    for attempt in range(1, 6):
        try:
            resp = _requests.post(
                _RUNWARE_API_URL,
                json=payload,
                headers=headers,
                timeout=120,
            )
            resp.raise_for_status()
            body = resp.json()

            # Runware returns {"data": [...]} with one result per task
            results = body.get("data") or body.get("results") or []
            if not results:
                raise ValueError(f"Empty result list from Runware: {body}")

            result = results[0]
            if "error" in result:
                raise ValueError(f"Runware task error: {result['error']}")

            # Field name for base64 payload (prefer imageBase64Data, fall back to URL)
            b64 = result.get("imageBase64Data") or result.get("base64Data")
            if b64:
                return base64.b64decode(b64)

            # Fallback: if the server returned a URL despite outputType=base64Data
            url = result.get("imageURL")
            if url:
                img_resp = _requests.get(url, timeout=60)
                img_resp.raise_for_status()
                return img_resp.content

            raise ValueError(f"No image data in Runware result: {result}")

        except Exception as exc:
            exc_type = type(exc).__name__
            is_rate_limit = "429" in str(exc) or "rate_limit" in str(exc).lower()
            if attempt == 5:
                raise
            wait = delay * (2 ** (attempt - 1)) if is_rate_limit else delay
            label = "rate limit" if is_rate_limit else f"{exc_type}: {exc}"
            print(f"  [{face_id}] {label} (attempt {attempt}/5) — waiting {wait:.0f}s")
            time.sleep(wait)

    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _generate_one(ctx: dict, face_id: str, prompt: str, *, model: str, size: str, quality: str) -> bytes:
    """Route to the correct backend based on model name."""
    if model in _OPENAI_MODELS:
        return _generate_openai(ctx["openai_client"], face_id, prompt, model=model, size=size, quality=quality)
    if model in _RUNWARE_MODELS:
        return _generate_runware(ctx["runware_api_key"], face_id, prompt, size=size)
    raise ValueError(f"Unknown model: {model!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate transparent-background face images via AI image APIs."
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
        "--model", "-m", default="gpt-image-1.5", choices=_ALL_MODELS,
        help=(
            "Image generation model/backend (default: gpt-image-1.5).  "
            "gpt-image-* require OPENAI_API_KEY; "
            "runware-layerdiffuse requires RUNWARE_API_KEY."
        ),
    )
    parser.add_argument(
        "--quality", choices=["low", "medium", "high"], default="high",
        help="Image quality for OpenAI models (default: high; ignored for Runware)",
    )
    parser.add_argument(
        "--size", default="1024x1024",
        choices=["1024x1024", "1536x1024", "1024x1536", "auto"],
        help="Image size (default: 1024x1024; waist_up crops always use 1024x1536)",
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
    start  = args.start if args.start is not None else 0
    end    = args.end   if args.end   is not None else len(faces)
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
        print(f"  model   : {args.model}")
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

    # ---- Load API credentials ----------------------------------------------
    _load_dotenv()

    ctx: dict = {}

    if args.model in _OPENAI_MODELS:
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
        ctx["openai_client"] = OpenAI(api_key=api_key)

    elif args.model in _RUNWARE_MODELS:
        api_key = os.environ.get("RUNWARE_API_KEY")
        if not api_key:
            print(
                f"ERROR: RUNWARE_API_KEY not set.\n"
                f"  Add it to {_PROJECT_ROOT / '.env'}  (RUNWARE_API_KEY=...)\n"
                f"  or export it in your shell.\n"
                f"  Sign up at https://runware.ai — new accounts receive $10 free credits.",
                file=sys.stderr,
            )
            sys.exit(1)
        ctx["runware_api_key"] = api_key

    # ---- Generation loop ----------------------------------------------------
    total     = len(subset)
    skipped   = 0
    generated = 0
    failed    = []

    quality_note = f"  quality={args.quality}" if args.model in _OPENAI_MODELS else ""
    print(f"Generating {total} face image(s) → {out_dir}")
    print(f"  model={args.model}{quality_note}  size={args.size} (waist_up→1024x1536)  delay={args.delay}s")
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
                ctx, face_id, prompt,
                model=args.model, size=size, quality=args.quality,
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
