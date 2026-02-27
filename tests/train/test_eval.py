"""tests/train/test_eval.py — unit tests for train/eval_onnx.py.

All tests run without a GPU, without ROS, and without Gazebo.
They use tiny synthetic HDF5 fixtures and a dummy ONNX model built
from a trivial PyTorch module (torch.onnx.export — available in .venv).
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

from train.eval_onnx import (
    MSE_THRESHOLD,
    PER_LABEL_LIMIT,
    _build_tokenize_fn,
    _preprocess,
    run_eval,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

N_FRAMES   = 10
N_EPISODES = 3

CMDS       = ["track the face", "follow slowly", "track the face"]
LABEL_KEYS = ["basic_track",    "slow_follow",   "basic_track"]


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

class _ZeroModel(nn.Module):
    """Outputs values ≈ 0 while routing all three inputs through the graph.

    torch.onnx.export prunes inputs that are not connected to any output node.
    Multiplying by 1e-20 keeps all inputs in the graph without affecting the
    effective output (1e-20 is below float32 precision for typical inputs).
    """

    def forward(
        self,
        frames: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return (
            frames[:, 0, 0:2, 0]          * 1e-20
            + input_ids[:, 0:2].float()    * 1e-20
            + attention_mask[:, 0:2].float() * 1e-20
        )


def _export_onnx(tmp_path: Path, model: nn.Module | None = None) -> Path:
    """Export *model* (default: _ZeroModel) to ONNX and return the path."""
    if model is None:
        model = _ZeroModel()
    model.eval()

    out = tmp_path / "model.onnx"
    dummy = (
        torch.zeros(1, 3, 224, 224),
        torch.zeros(1, 77, dtype=torch.long),
        torch.ones(1, 77, dtype=torch.long),
    )
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["frames", "input_ids", "attention_mask"],
        output_names=["actions"],
        dynamic_axes={
            "frames":         {0: "batch"},
            "input_ids":      {0: "batch"},
            "attention_mask": {0: "batch"},
            "actions":        {0: "batch"},
        },
        opset_version=17,
    )
    return out


def _make_dataset(
    root: Path,
    pan_val: float = 0.0,
    tilt_val: float = 0.0,
    n_episodes: int = N_EPISODES,
    n_frames: int = N_FRAMES,
) -> Path:
    """Write a minimal sharded dataset under *root*. All targets = pan_val/tilt_val."""
    dataset_dir = root / "dataset"
    shard = dataset_dir / "shard_0"
    eps_dir = shard / "episodes"
    eps_dir.mkdir(parents=True)

    ep_ids = [f"{i:06d}" for i in range(n_episodes)]
    for i, ep_id in enumerate(ep_ids):
        with h5py.File(eps_dir / f"ep_{ep_id}.h5", "w") as f:
            f.create_dataset(
                "frames",
                data=np.zeros((n_frames, 224, 224, 3), dtype=np.uint8),
            )
            f.create_dataset("pan_vel",  data=np.full(n_frames, pan_val,  dtype=np.float32))
            f.create_dataset("tilt_vel", data=np.full(n_frames, tilt_val, dtype=np.float32))
            f["cmd"]       = CMDS[i % len(CMDS)]
            f["label_key"] = LABEL_KEYS[i % len(LABEL_KEYS)]

    # Splits: ep0 → val; all → test; ep0 → train (unused but must be present)
    (shard / "train.txt").write_text(ep_ids[0])
    (shard / "val.txt").write_text(ep_ids[0])
    (shard / "test.txt").write_text("\n".join(ep_ids))
    return dataset_dir


def _make_token_cache(root: Path) -> Path:
    """Write a minimal token-cache JSON for all commands used in the fixtures."""
    cache = {
        cmd: {
            "input_ids":      [1] + [0] * 76,
            "attention_mask": [1] + [0] * 76,
        }
        for cmd in set(CMDS)
    }
    path = root / "tokens.json"
    path.write_text(json.dumps(cache))
    return path


# ---------------------------------------------------------------------------
# Tests: _preprocess
# ---------------------------------------------------------------------------

class TestPreprocess:

    def test_output_shape(self):
        out = _preprocess(np.zeros((4, 224, 224, 3), dtype=np.uint8))
        assert out.shape == (4, 3, 224, 224)

    def test_dtype_float32(self):
        out = _preprocess(np.zeros((2, 224, 224, 3), dtype=np.uint8))
        assert out.dtype == np.float32

    def test_zero_pixels_become_negative(self):
        """Zero pixels map to negative values after ImageNet normalisation."""
        out = _preprocess(np.zeros((1, 224, 224, 3), dtype=np.uint8))
        assert (out < 0).all()

    def test_full_white_pixels_become_positive(self):
        out = _preprocess(np.full((1, 224, 224, 3), 255, dtype=np.uint8))
        assert (out > 0).all()

    def test_contiguous(self):
        out = _preprocess(np.zeros((2, 224, 224, 3), dtype=np.uint8))
        assert out.flags["C_CONTIGUOUS"]

    def test_single_frame(self):
        out = _preprocess(np.zeros((1, 224, 224, 3), dtype=np.uint8))
        assert out.shape == (1, 3, 224, 224)


# ---------------------------------------------------------------------------
# Tests: _build_tokenize_fn
# ---------------------------------------------------------------------------

class TestBuildTokenizeFn:

    def test_cache_hit_shapes(self, tmp_path):
        tokenize = _build_tokenize_fn(_make_token_cache(tmp_path))
        ids, mask = tokenize("track the face")
        assert ids.shape  == (1, 77)
        assert mask.shape == (1, 77)
        assert ids.dtype  == np.int64
        assert mask.dtype == np.int64

    def test_memoised(self, tmp_path):
        """Same command returns the identical tuple object (no re-allocation)."""
        tokenize = _build_tokenize_fn(_make_token_cache(tmp_path))
        a = tokenize("track the face")
        b = tokenize("track the face")
        assert a is b

    def test_none_path_returns_callable(self):
        """No cache path should still return a callable (lazy transformers load)."""
        tokenize = _build_tokenize_fn(None)
        assert callable(tokenize)

    def test_empty_cache_path_returns_callable(self, tmp_path):
        """Non-existent cache path should behave like no cache."""
        tokenize = _build_tokenize_fn(tmp_path / "nonexistent.json")
        assert callable(tokenize)


# ---------------------------------------------------------------------------
# Tests: run_eval — main gate logic
# ---------------------------------------------------------------------------

class TestRunEval:

    def test_fail_zero_model_large_targets(self, tmp_path):
        """Zero-output model on targets=1.0 dataset → FAIL."""
        onnx   = _export_onnx(tmp_path)
        ds_dir = _make_dataset(tmp_path / "d1", pan_val=1.0, tilt_val=1.0)
        cache  = _make_token_cache(tmp_path)

        report = run_eval(
            model_path=onnx,
            dataset_dir=ds_dir,
            split="test",
            batch_size=8,
            token_cache_path=cache,
        )

        assert report["pass"] is False
        # MSE = mean((0-1)², (0-1)²) over (T, 2) → mean element error = 1.0
        assert report["overall_mse"] == pytest.approx(1.0, rel=1e-4)

    def test_pass_zero_model_zero_targets(self, tmp_path):
        """Zero-output model on targets=0 dataset → MSE=0 → PASS."""
        onnx   = _export_onnx(tmp_path)
        ds_dir = _make_dataset(tmp_path / "d2", pan_val=0.0, tilt_val=0.0)
        cache  = _make_token_cache(tmp_path)

        report = run_eval(
            model_path=onnx,
            dataset_dir=ds_dir,
            split="test",
            batch_size=8,
            token_cache_path=cache,
        )

        assert report["pass"] is True
        assert report["overall_mse"] == pytest.approx(0.0, abs=1e-7)

    def test_report_required_keys(self, tmp_path):
        onnx   = _export_onnx(tmp_path)
        ds_dir = _make_dataset(tmp_path / "d3")
        cache  = _make_token_cache(tmp_path)

        report = run_eval(
            model_path=onnx,
            dataset_dir=ds_dir,
            split="test",
            token_cache_path=cache,
        )

        required = {
            "model_path", "dataset_dir", "split", "n_episodes",
            "overall_mse", "per_label_mse", "mse_threshold",
            "per_label_limit", "pass",
        }
        assert required <= set(report.keys())

    def test_per_label_mse_keys(self, tmp_path):
        """per_label_mse has one entry per unique label_key in the split."""
        onnx   = _export_onnx(tmp_path)
        ds_dir = _make_dataset(tmp_path / "d4", pan_val=0.0)
        cache  = _make_token_cache(tmp_path)

        report = run_eval(
            model_path=onnx,
            dataset_dir=ds_dir,
            split="test",
            token_cache_path=cache,
        )

        assert set(report["per_label_mse"].keys()) == {"basic_track", "slow_follow"}

    def test_n_episodes(self, tmp_path):
        onnx   = _export_onnx(tmp_path)
        ds_dir = _make_dataset(tmp_path / "d5", n_episodes=3)
        cache  = _make_token_cache(tmp_path)

        report = run_eval(
            model_path=onnx,
            dataset_dir=ds_dir,
            split="test",
            token_cache_path=cache,
        )

        assert report["n_episodes"] == 3

    def test_per_label_limit_triggers_fail(self, tmp_path):
        """overall_mse < threshold but one label exceeds per_label_limit → FAIL."""
        ds_dir = tmp_path / "d6" / "dataset"
        shard  = ds_dir / "shard_0"
        eps    = shard / "episodes"
        eps.mkdir(parents=True)

        n = N_FRAMES
        # basic_track: targets=1.0 → MSE=2.0 for zero model
        with h5py.File(eps / "ep_000000.h5", "w") as f:
            f.create_dataset("frames",   data=np.zeros((n, 224, 224, 3), dtype=np.uint8))
            f.create_dataset("pan_vel",  data=np.ones(n, dtype=np.float32))
            f.create_dataset("tilt_vel", data=np.ones(n, dtype=np.float32))
            f["cmd"]       = "track the face"
            f["label_key"] = "basic_track"

        # slow_follow: targets=0.0 → MSE=0 for zero model
        with h5py.File(eps / "ep_000001.h5", "w") as f:
            f.create_dataset("frames",   data=np.zeros((n, 224, 224, 3), dtype=np.uint8))
            f.create_dataset("pan_vel",  data=np.zeros(n, dtype=np.float32))
            f.create_dataset("tilt_vel", data=np.zeros(n, dtype=np.float32))
            f["cmd"]       = "follow slowly"
            f["label_key"] = "slow_follow"

        (shard / "train.txt").write_text("")
        (shard / "val.txt").write_text("")
        (shard / "test.txt").write_text("000000\n000001")

        onnx  = _export_onnx(tmp_path)
        cache = _make_token_cache(tmp_path)

        # Lenient overall threshold (overall_mse=1.0 < 10.0) but tight per-label
        report = run_eval(
            model_path=onnx,
            dataset_dir=ds_dir,
            split="test",
            token_cache_path=cache,
            mse_threshold=10.0,
            per_label_limit=0.05,
        )

        # basic_track MSE = mean((0-1)², (0-1)²) = 1.0 → exceeds per_label_limit=0.05
        assert report["pass"] is False
        assert report["per_label_mse"]["basic_track"] == pytest.approx(1.0, rel=1e-4)
        assert report["per_label_mse"]["slow_follow"]  == pytest.approx(0.0, abs=1e-7)

    def test_val_split(self, tmp_path):
        """Eval on val split should use val.txt (1 episode in fixture)."""
        onnx   = _export_onnx(tmp_path)
        ds_dir = _make_dataset(tmp_path / "d7")
        cache  = _make_token_cache(tmp_path)

        report = run_eval(
            model_path=onnx,
            dataset_dir=ds_dir,
            split="val",
            token_cache_path=cache,
        )

        assert report["split"] == "val"
        assert report["n_episodes"] == 1

    def test_custom_mse_threshold(self, tmp_path):
        """Raising the threshold to 5.0 should make the large-target case PASS."""
        onnx   = _export_onnx(tmp_path)
        ds_dir = _make_dataset(tmp_path / "d8", pan_val=1.0, tilt_val=1.0)
        cache  = _make_token_cache(tmp_path)

        report = run_eval(
            model_path=onnx,
            dataset_dir=ds_dir,
            split="test",
            token_cache_path=cache,
            mse_threshold=5.0,
            per_label_limit=5.0,  # also relax per-label so it doesn't block PASS
        )

        assert report["pass"] is True
        assert report["mse_threshold"] == 5.0

    def test_report_is_json_serialisable(self, tmp_path):
        onnx   = _export_onnx(tmp_path)
        ds_dir = _make_dataset(tmp_path / "d9")
        cache  = _make_token_cache(tmp_path)

        report = run_eval(
            model_path=onnx,
            dataset_dir=ds_dir,
            split="test",
            token_cache_path=cache,
        )

        # Should not raise
        serialised = json.dumps(report)
        roundtripped = json.loads(serialised)
        assert roundtripped["n_episodes"] == report["n_episodes"]

    def test_weighted_mse(self, tmp_path):
        """overall_mse is weighted by episode length, not a simple mean."""
        ds_dir = tmp_path / "d10" / "dataset"
        shard  = ds_dir / "shard_0"
        eps    = shard / "episodes"
        eps.mkdir(parents=True)

        # ep0: 90 frames, targets=1.0 → MSE=2.0
        with h5py.File(eps / "ep_000000.h5", "w") as f:
            f.create_dataset("frames",   data=np.zeros((90, 224, 224, 3), dtype=np.uint8))
            f.create_dataset("pan_vel",  data=np.ones(90, dtype=np.float32))
            f.create_dataset("tilt_vel", data=np.ones(90, dtype=np.float32))
            f["cmd"]       = "track the face"
            f["label_key"] = "basic_track"

        # ep1: 10 frames, targets=0.0 → MSE=0.0
        with h5py.File(eps / "ep_000001.h5", "w") as f:
            f.create_dataset("frames",   data=np.zeros((10, 224, 224, 3), dtype=np.uint8))
            f.create_dataset("pan_vel",  data=np.zeros(10, dtype=np.float32))
            f.create_dataset("tilt_vel", data=np.zeros(10, dtype=np.float32))
            f["cmd"]       = "track the face"
            f["label_key"] = "basic_track"

        (shard / "train.txt").write_text("")
        (shard / "val.txt").write_text("")
        (shard / "test.txt").write_text("000000\n000001")

        onnx  = _export_onnx(tmp_path)
        cache = _make_token_cache(tmp_path)

        report = run_eval(
            model_path=onnx,
            dataset_dir=ds_dir,
            split="test",
            token_cache_path=cache,
        )

        # Weighted MSE = (90*1.0 + 10*0.0) / 100 = 0.9, NOT (1.0+0.0)/2=0.5
        assert report["overall_mse"] == pytest.approx(0.9, rel=1e-4)
