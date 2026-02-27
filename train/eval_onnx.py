"""train/eval_onnx.py — Offline ONNX evaluation of a trained VLAModel.

Iterates over the test (or val) split of an Ocelot dataset, runs ONNX
inference for each frame, and reports per-axis MSE and a per-label-key
breakdown.

Pass/fail gate (determines whether a model can be tagged 'deployable'):
  • overall_mse < MSE_THRESHOLD (default 0.05)
  • no per-label MSE > PER_LABEL_LIMIT (default 0.20)

Exit code: 0 if PASS, 1 if FAIL.

Usage:
    source .venv/bin/activate

    python3 train/eval_onnx.py \\
        --model_path runs/v0.0-smoke/best.onnx \\
        --dataset_dir sim/dataset/

    # With pre-computed token cache (avoids transformers at runtime):
    python3 train/eval_onnx.py \\
        --model_path runs/v0.0-smoke/best.onnx \\
        --dataset_dir sim/dataset/ \\
        --token_cache runs/v0.0-smoke/best_tokens.json \\
        --split test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import h5py
import numpy as np

from train.dataset import IMAGENET_MEAN, IMAGENET_STD, OcelotDataset

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gate thresholds (Phase 3 plan Step 6c)
# ---------------------------------------------------------------------------

MSE_THRESHOLD   = 0.05   # overall MSE must be below this to PASS
PER_LABEL_LIMIT = 0.20   # no individual label-key MSE may exceed this

_MAX_LEN = 77  # CLIP max token length — must match export_onnx.py


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess(frames_nhwc: np.ndarray) -> np.ndarray:
    """uint8 (N, H, W, 3) → float32 (N, 3, H, W) ImageNet-normalised."""
    f = frames_nhwc.astype(np.float32) / 255.0
    f = (f - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(f.transpose(0, 3, 1, 2))


# ---------------------------------------------------------------------------
# Tokeniser — token-cache JSON first, transformers fallback
# ---------------------------------------------------------------------------

def _build_tokenize_fn(token_cache_path: Path | None):
    """Return a (cmd: str) → (input_ids (1,77), attn_mask (1,77)) callable.

    Loads from a token-cache JSON if provided (no transformers dependency),
    and falls back to CLIPTokenizerFast for commands not in the cache.
    All results are memoised so each unique command is tokenised once.
    """
    memo: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    if token_cache_path is not None and token_cache_path.exists():
        raw = json.loads(token_cache_path.read_text())
        for cmd, entry in raw.items():
            memo[cmd] = (
                np.array([entry["input_ids"]],      dtype=np.int64),
                np.array([entry["attention_mask"]], dtype=np.int64),
            )
        log.info("Token cache: %d commands from %s", len(memo), token_cache_path)

    # Lazy transformers tokeniser — only loaded on cache miss
    _hf: list = [None]

    def _get_hf():
        if _hf[0] is None:
            try:
                from transformers import CLIPTokenizerFast
                from train.model import VLAModel
                _hf[0] = CLIPTokenizerFast.from_pretrained(VLAModel.CLIP_ID)
                log.info("CLIPTokenizerFast loaded as fallback tokenizer.")
            except ImportError:
                raise RuntimeError(
                    "transformers not installed and command not in token cache. "
                    "Either pass --token_cache or: pip install transformers"
                )
        return _hf[0]

    def tokenize(cmd: str) -> tuple[np.ndarray, np.ndarray]:
        if cmd not in memo:
            tok = _get_hf()
            enc = tok(
                cmd,
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=_MAX_LEN,
            )
            memo[cmd] = (
                enc["input_ids"].astype(np.int64),
                enc["attention_mask"].astype(np.int64),
            )
        return memo[cmd]  # (1, 77), (1, 77)

    return tokenize


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def run_eval(
    model_path: Path,
    dataset_dir: Path,
    split: str = "test",
    batch_size: int = 64,
    token_cache_path: Path | None = None,
    mse_threshold: float = MSE_THRESHOLD,
    per_label_limit: float = PER_LABEL_LIMIT,
) -> dict:
    """Run offline evaluation and return a report dict.

    The returned dict is JSON-serialisable and contains a boolean ``pass``
    key indicating whether the model cleared both gate thresholds.
    """
    import onnxruntime as ort

    log.info("Loading ONNX model: %s", model_path)
    sess = ort.InferenceSession(
        str(model_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    log.info("Provider: %s", sess.get_providers()[0])

    tokenize = _build_tokenize_fn(token_cache_path)

    # Discover episodes in the split
    split_files = OcelotDataset._find_split_files(dataset_dir, split)
    if not split_files:
        raise FileNotFoundError(f"No {split}.txt files found under {dataset_dir}")

    episode_paths: list[tuple[str, Path]] = []
    for sf in sorted(split_files):
        shard_dir = sf.parent
        ep_ids = [line.strip() for line in sf.read_text().splitlines() if line.strip()]
        for ep_id in ep_ids:
            h5_path = shard_dir / "episodes" / f"ep_{ep_id}.h5"
            if h5_path.exists():
                episode_paths.append((ep_id, h5_path))

    if not episode_paths:
        raise RuntimeError(f"No episode HDF5 files found for split={split!r}")

    log.info("Evaluating %d episodes (%s split) …", len(episode_paths), split)

    ep_results: list[dict] = []
    for ep_idx, (ep_id, h5_path) in enumerate(episode_paths):
        with h5py.File(h5_path, "r") as f:
            frames_nhwc   = f["frames"][:]   # (T, 224, 224, 3) uint8
            pan_gt        = f["pan_vel"][:]  # (T,) float32
            tilt_gt       = f["tilt_vel"][:] # (T,) float32
            cmd_raw       = f["cmd"][()]
            label_key_raw = f["label_key"][()]

        cmd = cmd_raw.decode("utf-8") if isinstance(cmd_raw, bytes) else str(cmd_raw)
        label_key = (
            label_key_raw.decode("utf-8")
            if isinstance(label_key_raw, bytes)
            else str(label_key_raw)
        )

        ids, mask = tokenize(cmd)  # (1, 77), (1, 77)

        T = len(frames_nhwc)
        frames_nchw = _preprocess(frames_nhwc)  # (T, 3, 224, 224)

        preds: list[np.ndarray] = []
        for start in range(0, T, batch_size):
            end    = min(start + batch_size, T)
            fb     = frames_nchw[start:end]              # (B, 3, 224, 224)
            B      = len(fb)
            ids_b  = np.repeat(ids,  B, axis=0)          # (B, 77)
            mask_b = np.repeat(mask, B, axis=0)          # (B, 77)
            pred   = sess.run(
                ["actions"],
                {"frames": fb, "input_ids": ids_b, "attention_mask": mask_b},
            )[0]  # (B, 2)
            preds.append(pred)

        pred_arr = np.concatenate(preds, axis=0)             # (T, 2)
        gt_arr   = np.stack([pan_gt, tilt_gt], axis=1)      # (T, 2)
        mse      = float(np.mean((pred_arr - gt_arr) ** 2))

        ep_results.append({
            "ep_id":     ep_id,
            "label_key": label_key,
            "mse":       mse,
            "n_frames":  T,
        })

        if (ep_idx + 1) % 50 == 0 or ep_idx == 0:
            log.info("  [%d/%d] ep_%s  mse=%.5f  label=%s",
                     ep_idx + 1, len(episode_paths), ep_id, mse, label_key)

    # Aggregate — weighted average by episode length for a true per-frame MSE
    all_mse    = np.array([r["mse"]      for r in ep_results])
    all_frames = np.array([r["n_frames"] for r in ep_results])
    overall_mse = float(np.average(all_mse, weights=all_frames))

    label_keys = sorted({r["label_key"] for r in ep_results})
    per_label_mse: dict[str, float] = {}
    for lk in label_keys:
        subset = [r for r in ep_results if r["label_key"] == lk]
        w = np.array([r["n_frames"] for r in subset])
        m = np.array([r["mse"]      for r in subset])
        per_label_mse[lk] = float(np.average(m, weights=w))

    passed = (
        overall_mse < mse_threshold
        and all(v < per_label_limit for v in per_label_mse.values())
    )

    return {
        "model_path":      str(model_path),
        "dataset_dir":     str(dataset_dir),
        "split":           split,
        "n_episodes":      len(ep_results),
        "overall_mse":     overall_mse,
        "per_label_mse":   per_label_mse,
        "mse_threshold":   mse_threshold,
        "per_label_limit": per_label_limit,
        "pass":            passed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline ONNX evaluation with pass/fail gate"
    )
    p.add_argument("--model_path",    required=True, type=Path, help="Path to .onnx model")
    p.add_argument("--dataset_dir",   required=True, type=Path)
    p.add_argument("--split",         default="test", choices=["val", "test"])
    p.add_argument("--output",        type=Path, default=None,
                   help="Write report JSON here (default: <model_stem>_eval_<split>.json)")
    p.add_argument("--batch_size",    type=int, default=64)
    p.add_argument("--token_cache",   type=Path, default=None,
                   help="Precomputed token cache JSON from export_onnx.py")
    p.add_argument("--mse_threshold", type=float, default=MSE_THRESHOLD,
                   help=f"Overall MSE pass threshold (default: {MSE_THRESHOLD})")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    report = run_eval(
        model_path=args.model_path,
        dataset_dir=args.dataset_dir,
        split=args.split,
        batch_size=args.batch_size,
        token_cache_path=args.token_cache,
        mse_threshold=args.mse_threshold,
    )

    out_path = args.output or args.model_path.with_name(
        args.model_path.stem + f"_eval_{args.split}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    log.info("Report written to %s", out_path)

    verdict = "PASS" if report["pass"] else "FAIL"
    print(f"\n{'='*52}")
    print(f"  Episodes:    {report['n_episodes']}")
    print(f"  Overall MSE: {report['overall_mse']:.5f}  (threshold: {report['mse_threshold']})")
    if report["per_label_mse"]:
        print(f"  Per-label MSE:")
        for lk, mse in sorted(report["per_label_mse"].items()):
            flag = "  !" if mse >= report["per_label_limit"] else "   "
            print(f"    {flag} {lk:30s}: {mse:.5f}")
    print(f"  Verdict: {verdict}")
    print(f"{'='*52}\n")

    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
