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
        --model gpt-image-1.5 --provider runware
    python3 generate_face_images.py --input ... --dry-run      # preview only
    python3 generate_face_images.py --input ... --start 10 --end 20  # subset

Models:
    gpt-image-1          — OpenAI; ~$0.05–0.25/image (quality: low/med/high)
    gpt-image-1.5        — OpenAI updated variant; same pricing tiers
                           also available via --provider runware (~$0.009–0.20/image)
    gpt-image-1-mini     — OpenAI cheaper variant; ~$0.011–0.016/image

Providers:
    openai   (default) — calls OpenAI API directly; requires OPENAI_API_KEY
    runware            — routes through Runware; requires RUNWARE_API_KEY
                         only gpt-image-1.5 is currently supported on Runware

Environment:
    OPENAI_API_KEY    — required for provider=openai
    RUNWARE_API_KEY   — required for provider=runware
    Both are loaded automatically from a .env file in the project root
    or from the shell environment.
"""

import argparse
import base64
import json
import os
import sys
import threading
import time
import uuid as _uuid_mod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

_print_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Model / provider constants
# ---------------------------------------------------------------------------

_OPENAI_MODELS  = frozenset({"gpt-image-1", "gpt-image-1.5", "gpt-image-1-mini"})
_ALL_MODELS     = sorted(_OPENAI_MODELS)

# Models available via Runware's OpenAI-compatible proxy and their internal IDs
_RUNWARE_MODEL_IDS = {
    "gpt-image-1.5": "openai:4@1",
}
_RUNWARE_API_URL = "https://api.runware.ai/v1"

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
# OpenAI backend (direct)
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
# Runware backend (OpenAI models proxied through Runware)
# ---------------------------------------------------------------------------

def _generate_runware(api_key: str, face_id: str, prompt: str, *, model: str, size: str, quality: str) -> bytes:
    """
    Call the Runware API with an OpenAI model and return raw PNG bytes.
    Uses providerSettings.openai to pass background=transparent and quality.
    Docs: https://runware.ai/docs/en/providers/openai
    """
    try:
        import requests as _requests
    except ImportError:
        print("ERROR: requests package not installed.  Run: pip install requests", file=sys.stderr)
        sys.exit(1)

    runware_model_id = _RUNWARE_MODEL_IDS[model]
    clean_prompt = _sanitize_prompt(prompt)
    w, h = _parse_wh(size)

    payload = [{
        "taskType":        "imageInference",
        "taskUUID":        str(_uuid_mod.uuid4()),
        "model":           runware_model_id,
        "positivePrompt":  clean_prompt,
        "width":           w,
        "height":          h,
        "numberResults":   1,
        "outputType":      ["base64Data"],
        "outputFormat":    "PNG",
        "providerSettings": {
            "openai": {
                "quality":    quality,
                "background": "transparent",
            }
        },
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

            results = body.get("data") or body.get("results") or []
            if not results:
                raise ValueError(f"Empty result list from Runware: {body}")

            result = results[0]
            if "error" in result:
                raise ValueError(f"Runware task error: {result['error']}")

            b64 = result.get("imageBase64Data") or result.get("base64Data")
            if b64:
                return base64.b64decode(b64)

            # Fallback: download from URL if base64 wasn't returned
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

def _generate_one(ctx: dict, face_id: str, prompt: str, *, model: str, provider: str, size: str, quality: str) -> bytes:
    """Route to the correct backend based on provider."""
    if provider == "runware":
        return _generate_runware(ctx["runware_api_key"], face_id, prompt, model=model, size=size, quality=quality)
    return _generate_openai(ctx["openai_client"], face_id, prompt, model=model, size=size, quality=quality)


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
        help="Image generation model (default: gpt-image-1.5)",
    )
    parser.add_argument(
        "--provider", default="openai", choices=["openai", "runware"],
        help=(
            "API provider (default: openai).  "
            "'runware' proxies the request through Runware; "
            "only gpt-image-1.5 is currently supported there."
        ),
    )
    parser.add_argument(
        "--quality", choices=["low", "medium", "high"], default="high",
        help="Image quality (default: high)",
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
        "--workers", "-w", type=int, default=1,
        help="Number of concurrent worker threads (default: 1, sequential). "
             "Values of 3-5 work well for Runware; use 1 for OpenAI to avoid rate limits.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be generated without calling the API",
    )
    args = parser.parse_args()

    # ---- Validate provider/model combo -------------------------------------
    if args.provider == "runware" and args.model not in _RUNWARE_MODEL_IDS:
        supported = ", ".join(sorted(_RUNWARE_MODEL_IDS))
        print(
            f"ERROR: --provider runware only supports: {supported}\n"
            f"  Got: --model {args.model}",
            file=sys.stderr,
        )
        sys.exit(1)

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
        print(f"  model    : {args.model}  provider={args.provider}")
        print(f"  out_dir  : {out_dir}")
        print(f"  quality  : {args.quality}  size: {args.size}")
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

    if args.provider == "openai":
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

    elif args.provider == "runware":
        api_key = os.environ.get("RUNWARE_API_KEY")
        if not api_key:
            print(
                f"ERROR: RUNWARE_API_KEY not set.\n"
                f"  Add it to {_PROJECT_ROOT / '.env'}  (RUNWARE_API_KEY=...)\n"
                f"  or export it in your shell.",
                file=sys.stderr,
            )
            sys.exit(1)
        ctx["runware_api_key"] = api_key

    # ---- Generation loop ----------------------------------------------------
    total     = len(subset)
    skipped   = 0
    generated = 0
    failed    = []

    concurrent = args.workers > 1
    delay_note = f"  delay={args.delay}s" if not concurrent else f"  workers={args.workers}"
    print(f"Generating {total} face image(s) → {out_dir}")
    print(f"  model={args.model}  provider={args.provider}  quality={args.quality}  size={args.size} (waist_up→1024x1536){delay_note}")
    print()

    def _run_task(face, idx):
        """Generate one face. Returns ('skip'|'ok'|'fail', face_id)."""
        face_id    = face["face_id"]
        prompt     = face["prompt"]
        crop_level = face.get("crop_level", "")
        size       = "1024x1536" if crop_level == "waist_up" else args.size
        out_path   = out_dir / f"{face_id}.png"
        w          = len(str(total))
        prefix     = f"[{idx:>{w}}/{total}] {face_id}"

        if out_path.exists():
            with _print_lock:
                print(f"{prefix}  SKIP (already exists)")
            return ("skip", face_id)

        with _print_lock:
            print(f"{prefix} [{size}]  generating...", flush=True)
        t0 = time.monotonic()

        try:
            png_bytes = _generate_one(
                ctx, face_id, prompt,
                model=args.model, provider=args.provider, size=size, quality=args.quality,
            )
        except Exception as exc:
            with _print_lock:
                print(f"{prefix}  ERROR: {exc}")
            return ("fail", face_id)

        out_path.write_bytes(png_bytes)
        elapsed = time.monotonic() - t0
        with _print_lock:
            print(f"{prefix}  done ({elapsed:.1f}s, {len(png_bytes)//1024} KB)")
        return ("ok", face_id)

    if not concurrent:
        for idx, face in enumerate(subset, start=1):
            status, face_id = _run_task(face, idx)
            if status == "skip":
                skipped += 1
            elif status == "ok":
                generated += 1
                if idx < total and args.delay > 0:
                    time.sleep(args.delay)
            else:
                failed.append(face_id)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_run_task, face, idx): face["face_id"]
                for idx, face in enumerate(subset, start=1)
            }
            for future in as_completed(futures):
                try:
                    status, face_id = future.result()
                except Exception as exc:
                    face_id = futures[future]
                    with _print_lock:
                        print(f"  [{face_id}] unexpected error: {exc}")
                    failed.append(face_id)
                    continue
                if status == "skip":
                    skipped += 1
                elif status == "ok":
                    generated += 1
                else:
                    failed.append(face_id)

    # ---- Summary ------------------------------------------------------------
    print()
    print(f"Done.  generated={generated}  skipped={skipped}  failed={len(failed)}")
    if failed:
        print(f"  Failed face IDs: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
