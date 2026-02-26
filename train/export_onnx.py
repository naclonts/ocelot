"""train/export_onnx.py — Export VLAModel checkpoint to ONNX + tokenized command cache.

The ONNX model has three inputs (all batch-dynamic on axis 0):
    frames         float32  (B, 3, 224, 224)  — ImageNet-normalised frames
    input_ids      int64    (B, 77)            — CLIP token IDs (padded to max_length=77)
    attention_mask int64    (B, 77)            — 1 for real tokens, 0 for padding

Output:
    actions        float32  (B, 2)             — (pan_vel, tilt_vel) in rad/s, bounded ±2.0

A companion JSON file (<stem>_tokens.json) is written alongside the ONNX model.
It contains pre-tokenized versions of all label-registry commands so that
vla_node.py can look up token arrays without needing transformers at runtime.

Run on the HOST in the project .venv (needs torch + transformers):

    source .venv/bin/activate

    python3 train/export_onnx.py \\
        --checkpoint runs/v0.0-smoke/best.pt \\
        --output     runs/v0.0-smoke/best.onnx

    # Verify the ONNX model runs:
    python3 train/export_onnx.py \\
        --checkpoint runs/v0.0-smoke/best.pt \\
        --output     runs/v0.0-smoke/best.onnx \\
        --verify
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

import numpy as np
import torch

from train.model import VLAModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Max token sequence length — must match dataset.py collate_fn
_MAX_LEN = 77

# All commands from the label registry that don't require attribute substitution.
# Multi-attr templates are omitted (they'd contain "{attr}").
_STATIC_COMMANDS: list[str] = [
    # track (single face)
    "look at the person",
    "look at me",
    "watch the person",
    "track the face",
    "follow the person",
    "keep your eye on the person",
    # multi_left
    "look at the person on the left",
    "look at the one on the left",
    "follow the person on the left",
    "track the leftmost person",
    "watch the person on the left",
    "keep your eye on the person furthest left",
    "the one on the left — follow them",
    # multi_right
    "look at the person on the right",
    "look at the one on the right",
    "follow the person on the right",
    "track the rightmost person",
    "watch the person on the right",
    "keep your eye on the person furthest right",
    "the one on the right — follow them",
    # multi_closest
    "look at the closest person",
    "look at whoever is nearest",
    "watch the closest person",
    "track the closest person",
    "follow the nearest one",
    "keep the person closest to you centered",
    "focus on the closest face",
    "stay with the one right in front of you",
]


def _tokenize_commands(commands: list[str]) -> dict[str, dict]:
    """Return {command: {input_ids: list[int], attention_mask: list[int]}}."""
    from transformers import CLIPTokenizerFast
    log.info("Loading CLIP tokenizer …")
    tokenizer = CLIPTokenizerFast.from_pretrained(VLAModel.CLIP_ID)

    cache: dict[str, dict] = {}
    for cmd in commands:
        enc = tokenizer(
            cmd,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=_MAX_LEN,
        )
        cache[cmd] = {
            "input_ids":      enc["input_ids"][0].tolist(),
            "attention_mask": enc["attention_mask"][0].tolist(),
        }
    log.info("Tokenized %d commands.", len(cache))
    return cache


def export(
    checkpoint: Path,
    output: Path,
    n_fusion_layers: int = 2,
    n_heads: int = 6,
    verify: bool = False,
) -> None:
    device = torch.device("cpu")  # export on CPU for portability

    # Load model
    log.info("Loading VLAModel from %s …", checkpoint)
    model = VLAModel(
        n_fusion_layers=n_fusion_layers,
        n_heads=n_heads,
        pretrained=True,
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    log.info("Checkpoint loaded.")

    # Dummy inputs for tracing (batch_size=1, max_len=77)
    dummy_frames  = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    dummy_ids     = torch.zeros(1, _MAX_LEN, dtype=torch.long)
    dummy_mask    = torch.ones(1,  _MAX_LEN, dtype=torch.long)

    output.parent.mkdir(parents=True, exist_ok=True)

    log.info("Exporting to ONNX → %s …", output)
    torch.onnx.export(
        model,
        (dummy_frames, dummy_ids, dummy_mask),
        str(output),
        input_names=["frames", "input_ids", "attention_mask"],
        output_names=["actions"],
        dynamic_axes={
            "frames":         {0: "batch"},
            "input_ids":      {0: "batch"},
            "attention_mask": {0: "batch"},
            "actions":        {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    log.info("ONNX export complete.")

    # Token cache
    token_cache_path = output.with_name(output.stem + "_tokens.json")
    token_cache = _tokenize_commands(_STATIC_COMMANDS)
    token_cache_path.write_text(json.dumps(token_cache, indent=2))
    log.info("Token cache → %s  (%d commands)", token_cache_path, len(token_cache))

    if verify:
        _verify(output, token_cache)


def _verify(onnx_path: Path, token_cache: dict) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("onnxruntime not installed — skipping verification.")
        return

    log.info("Verifying ONNX model …")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    cmd = "track the face"
    tokens = token_cache[cmd]
    ids   = np.array([tokens["input_ids"]],      dtype=np.int64)   # (1, 77)
    mask  = np.array([tokens["attention_mask"]], dtype=np.int64)   # (1, 77)
    frame = np.zeros((1, 3, 224, 224), dtype=np.float32)

    actions = sess.run(["actions"], {
        "frames":         frame,
        "input_ids":      ids,
        "attention_mask": mask,
    })[0]

    log.info(
        "Verification OK — command=%r  actions=(pan=%.4f, tilt=%.4f)",
        cmd, actions[0, 0], actions[0, 1],
    )
    assert actions.shape == (1, 2), f"Unexpected output shape {actions.shape}"
    assert np.all(np.abs(actions) <= 2.0 + 1e-4), "Output out of tanh bounds"
    log.info("All assertions passed.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export VLAModel to ONNX")
    p.add_argument("--checkpoint",      required=True, type=Path)
    p.add_argument("--output",          required=True, type=Path, help="Output .onnx path")
    p.add_argument("--n_fusion_layers", type=int, default=2)
    p.add_argument("--n_heads",         type=int, default=6)
    p.add_argument("--verify",          action="store_true",
                   help="Run a quick onnxruntime inference check after export")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    export(args.checkpoint, args.output, args.n_fusion_layers, args.n_heads, args.verify)


if __name__ == "__main__":
    main()
