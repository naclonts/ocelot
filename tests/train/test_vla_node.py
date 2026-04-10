"""tests/train/test_vla_node.py — unit tests for ocelot/vla_node.py.

All tests run without ROS, without a GPU, and without Gazebo.
ROS2 packages (rclpy, cv_bridge, geometry_msgs, sensor_msgs) are mocked
at module level so the import succeeds in the host .venv.

Tests cover:
  - _preprocess: BGR HxWx3 → float32 (1,3,224,224) ImageNet-normalised
  - build_tokenize_fn: cache hits and runtime tokenization for arbitrary commands
  - ONNX inference path: shape, dtype, and value-bound checks using a
    lightweight dummy model (no HuggingFace downloads needed)
"""

from __future__ import annotations

import json
import sys
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Mock ROS2 packages before importing vla_node
# (rclpy etc. are not installed in the host .venv)
# ---------------------------------------------------------------------------
for _mod in (
    "rclpy",
    "rclpy.node",
    "cv_bridge",
    "geometry_msgs",
    "geometry_msgs.msg",
    "sensor_msgs",
    "sensor_msgs.msg",
):
    sys.modules.setdefault(_mod, mock.MagicMock())

from ocelot.vla_inference import VLAInferenceEngine, build_tokenize_fn  # noqa: E402
from ocelot.vla_node import _preprocess  # noqa: E402

# ---------------------------------------------------------------------------
# Shared ONNX helpers
# ---------------------------------------------------------------------------

class _BoundedModel(nn.Module):
    """Dummy ONNX-exportable model whose outputs are bounded to (-2, 2).

    Mirrors the real VLAModel output bound: 2.0 * tanh(x).
    All three inputs are wired into the output so ONNX export keeps them.
    """

    def forward(
        self,
        frames: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = (
            frames[:, 0, 0, 0:2] * 0.0
            + input_ids[:, 0:2].float() * 0.0
            + attention_mask[:, 0:2].float() * 0.0
        )
        return 2.0 * torch.tanh(x)  # (B, 2), values in (-2, 2)


class _InputSensitiveModel(nn.Module):
    """Dummy model whose output varies with the frame pixel values."""

    def forward(
        self,
        frames: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return (
            frames[:, 0, 0, 0:2]
            + input_ids[:, 0:2].float() * 0.0
            + attention_mask[:, 0:2].float() * 0.0
        )


def _export_onnx(tmp_path: Path, model: nn.Module | None = None) -> Path:
    """Export *model* (default: _BoundedModel) to ONNX and return the path."""
    if model is None:
        model = _BoundedModel()
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


def _make_token_cache(root: Path, cmds: list[str] | None = None) -> Path:
    if cmds is None:
        cmds = ["track the face", "follow slowly", "look at the person"]
    cache = {
        cmd: {
            "input_ids":      [1] + [0] * 76,
            "attention_mask": [1] + [0] * 76,
        }
        for cmd in cmds
    }
    path = root / "tokens.json"
    path.write_text(json.dumps(cache))
    return path


# ---------------------------------------------------------------------------
# Tests: _preprocess
# ---------------------------------------------------------------------------

class TestPreprocess:

    def test_output_shape_exact(self):
        """224×224 BGR input → (1, 3, 224, 224)."""
        out = _preprocess(np.zeros((224, 224, 3), dtype=np.uint8))
        assert out.shape == (1, 3, 224, 224)

    def test_output_shape_resize(self):
        """Non-standard resolution is resized to 224×224."""
        out = _preprocess(np.zeros((480, 640, 3), dtype=np.uint8))
        assert out.shape == (1, 3, 224, 224)

    def test_dtype_float32(self):
        out = _preprocess(np.zeros((224, 224, 3), dtype=np.uint8))
        assert out.dtype == np.float32

    def test_zero_pixels_negative(self):
        """Black pixels normalise to negative values under ImageNet stats."""
        out = _preprocess(np.zeros((224, 224, 3), dtype=np.uint8))
        assert (out < 0).all()

    def test_white_pixels_positive(self):
        """White pixels normalise to positive values under ImageNet stats."""
        out = _preprocess(np.full((224, 224, 3), 255, dtype=np.uint8))
        assert (out > 0).all()

    def test_contiguous(self):
        out = _preprocess(np.zeros((224, 224, 3), dtype=np.uint8))
        assert out.flags["C_CONTIGUOUS"]

    def test_channel_order_bgr(self):
        """Blue-heavy BGR ≠ red-heavy BGR after RGB conversion + normalisation."""
        blue_bgr = np.zeros((224, 224, 3), dtype=np.uint8)
        blue_bgr[:, :, 0] = 200  # channel 0 = B in BGR

        red_bgr = np.zeros((224, 224, 3), dtype=np.uint8)
        red_bgr[:, :, 2] = 200  # channel 2 = R in BGR

        assert not np.allclose(_preprocess(blue_bgr), _preprocess(red_bgr))

    def test_imagenet_mean_approx(self):
        """ImageNet-grey image (mean pixel) should normalise close to zero."""
        # ImageNet mean ≈ (0.485, 0.456, 0.406) in RGB → (104, 116, 124) BGR uint8
        grey = np.zeros((224, 224, 3), dtype=np.uint8)
        grey[:, :, 0] = 104   # B
        grey[:, :, 1] = 116   # G
        grey[:, :, 2] = 124   # R
        out = _preprocess(grey)
        assert np.abs(out).mean() < 0.15


# ---------------------------------------------------------------------------
# Tests: build_tokenize_fn
# ---------------------------------------------------------------------------

class TestBuildTokenizeFn:

    def test_cache_hit_reuses_exported_arrays(self, tmp_path):
        tokenize = build_tokenize_fn(_make_token_cache(tmp_path))

        ids, mask = tokenize("track the face")

        assert ids.shape == (1, 77)
        assert mask.shape == (1, 77)
        assert ids.dtype == np.int64
        assert mask.dtype == np.int64
        assert ids[0, 0] == 1
        assert mask[0, 0] == 1

    def test_cache_hit_is_memoized(self, tmp_path):
        tokenize = build_tokenize_fn(_make_token_cache(tmp_path))

        first = tokenize("track the face")
        second = tokenize("track the face")

        assert first is second

    def test_uncached_command_tokenized_at_runtime(self, tmp_path, monkeypatch):
        calls: list[str] = []

        class _Tokenizer:
            def __call__(self, text, return_tensors, padding, truncation, max_length):
                calls.append(text)
                assert return_tensors == "np"
                assert padding == "max_length"
                assert truncation is True
                assert max_length == 77
                return {
                    "input_ids": np.full((1, 77), 9, dtype=np.int64),
                    "attention_mask": np.ones((1, 77), dtype=np.int64),
                }

        class _TokenizerClass:
            @staticmethod
            def from_pretrained(model_id):
                calls.append(f"load:{model_id}")
                return _Tokenizer()

        transformers_mod = mock.MagicMock(CLIPTokenizerFast=_TokenizerClass)
        monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

        tokenize = build_tokenize_fn(_make_token_cache(tmp_path))
        ids, mask = tokenize("turn slightly left and keep the face centered")

        assert ids.shape == (1, 77)
        assert mask.shape == (1, 77)
        assert calls[0].startswith("load:")
        assert calls[1] == "turn slightly left and keep the face centered"

    def test_uncached_command_is_memoized(self, tmp_path, monkeypatch):
        calls = {"loads": 0, "tokenizes": 0}

        class _Tokenizer:
            def __call__(self, text, return_tensors, padding, truncation, max_length):
                calls["tokenizes"] += 1
                return {
                    "input_ids": np.full((1, 77), 3, dtype=np.int64),
                    "attention_mask": np.ones((1, 77), dtype=np.int64),
                }

        class _TokenizerClass:
            @staticmethod
            def from_pretrained(model_id):
                calls["loads"] += 1
                return _Tokenizer()

        transformers_mod = mock.MagicMock(CLIPTokenizerFast=_TokenizerClass)
        monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

        tokenize = build_tokenize_fn(_make_token_cache(tmp_path))
        first = tokenize("new arbitrary command")
        second = tokenize("new arbitrary command")

        assert first is second
        assert calls == {"loads": 1, "tokenizes": 1}

    def test_uncached_command_without_transformers_raises(self, tmp_path, monkeypatch):
        tokenize = build_tokenize_fn(_make_token_cache(tmp_path))

        monkeypatch.delitem(sys.modules, "transformers", raising=False)
        real_import = __import__

        def _fake_import(name, *args, **kwargs):
            if name == "transformers":
                raise ImportError("missing transformers")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _fake_import)

        with pytest.raises(RuntimeError, match="transformers is required"):
            tokenize("arbitrary uncached command")


# ---------------------------------------------------------------------------
# Tests: ONNX inference path (simulates VLANode._image_cb minus ROS)
# ---------------------------------------------------------------------------

class TestONNXInferencePath:
    """Exercises the numpy operations VLANode uses in its hot path."""

    def test_output_shape(self, tmp_path):
        """ONNX model with correct I/O signature returns (1, 2) actions."""
        import onnxruntime as ort

        sess = ort.InferenceSession(
            str(_export_onnx(tmp_path)), providers=["CPUExecutionProvider"]
        )
        frame = _preprocess(np.zeros((224, 224, 3), dtype=np.uint8))
        ids   = np.array([[1] + [0] * 76], dtype=np.int64)
        mask  = np.array([[1] + [0] * 76], dtype=np.int64)

        actions = sess.run(
            ["actions"],
            {"frames": frame, "input_ids": ids, "attention_mask": mask},
        )[0]

        assert actions.shape == (1, 2)

    def test_output_dtype_float32(self, tmp_path):
        import onnxruntime as ort

        sess = ort.InferenceSession(
            str(_export_onnx(tmp_path)), providers=["CPUExecutionProvider"]
        )
        frame = _preprocess(np.zeros((224, 224, 3), dtype=np.uint8))
        ids   = np.array([[1] + [0] * 76], dtype=np.int64)
        mask  = np.ones((1, 77), dtype=np.int64)

        actions = sess.run(
            ["actions"],
            {"frames": frame, "input_ids": ids, "attention_mask": mask},
        )[0]

        assert actions.dtype == np.float32

    def test_output_bounded(self, tmp_path):
        """_BoundedModel outputs stay in (-2, 2) for arbitrary inputs."""
        import onnxruntime as ort

        sess = ort.InferenceSession(
            str(_export_onnx(tmp_path, model=_BoundedModel())),
            providers=["CPUExecutionProvider"],
        )
        rng = np.random.default_rng(42)
        for _ in range(8):
            frame = _preprocess(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))
            ids   = rng.integers(0, 49408, (1, 77), dtype=np.int64)
            mask  = np.ones((1, 77), dtype=np.int64)
            out   = sess.run(
                ["actions"],
                {"frames": frame, "input_ids": ids, "attention_mask": mask},
            )[0]
            assert np.all(out > -2.0) and np.all(out < 2.0), f"Out of bounds: {out}"

    def test_different_frames_different_outputs(self, tmp_path):
        """Distinct frames produce distinct model outputs."""
        import onnxruntime as ort

        sens_dir = tmp_path / "sens"
        sens_dir.mkdir()
        sess = ort.InferenceSession(
            str(_export_onnx(sens_dir, model=_InputSensitiveModel())),
            providers=["CPUExecutionProvider"],
        )
        ids  = np.array([[1] + [0] * 76], dtype=np.int64)
        mask = np.ones((1, 77), dtype=np.int64)

        frame_a = _preprocess(np.zeros((224, 224, 3), dtype=np.uint8))
        frame_b = _preprocess(np.full((224, 224, 3), 200, dtype=np.uint8))

        inputs = {"frames": frame_a, "input_ids": ids,
                  "attention_mask": mask}
        out_a = sess.run(["actions"], inputs)[0]
        inputs["frames"] = frame_b
        out_b = sess.run(["actions"], inputs)[0]

        assert not np.allclose(out_a, out_b)

    def test_token_cache_roundtrip(self, tmp_path):
        """Token cache JSON → numpy arrays → ONNX produces correct shape for every command."""
        import onnxruntime as ort

        sess  = ort.InferenceSession(
            str(_export_onnx(tmp_path)), providers=["CPUExecutionProvider"]
        )
        cache = json.loads(_make_token_cache(tmp_path).read_text())
        frame = _preprocess(np.zeros((224, 224, 3), dtype=np.uint8))

        for cmd, tokens in cache.items():
            ids  = np.array([tokens["input_ids"]],      dtype=np.int64)
            mask = np.array([tokens["attention_mask"]], dtype=np.int64)
            assert ids.shape  == (1, 77), f"Bad ids shape for {cmd!r}"
            assert mask.shape == (1, 77), f"Bad mask shape for {cmd!r}"
            out = sess.run(
                ["actions"],
                {"frames": frame, "input_ids": ids, "attention_mask": mask},
            )[0]
            assert out.shape == (1, 2), f"Bad output shape for {cmd!r}"

    def test_prebuilt_tokens_reused(self, tmp_path):
        """Pre-building token arrays once and reusing them is idempotent."""
        import onnxruntime as ort

        sess  = ort.InferenceSession(
            str(_export_onnx(tmp_path)), providers=["CPUExecutionProvider"]
        )
        ids  = np.array([[1] + [0] * 76], dtype=np.int64)
        mask = np.ones((1, 77), dtype=np.int64)
        frame = _preprocess(np.zeros((224, 224, 3), dtype=np.uint8))

        inputs = {"frames": frame, "input_ids": ids,
                  "attention_mask": mask}
        out_first = sess.run(["actions"], inputs)[0]
        out_second = sess.run(["actions"], inputs)[0]

        np.testing.assert_array_equal(out_first, out_second)

    def test_shared_engine_predicts_cached_command(self, tmp_path):
        """Shared inference engine returns pan/tilt for cached commands."""
        model_path = _export_onnx(tmp_path, model=_InputSensitiveModel())
        token_cache = _make_token_cache(tmp_path)
        engine = VLAInferenceEngine(
            checkpoint=model_path,
            token_cache=token_cache,
            providers=["CPUExecutionProvider"],
        )

        result = engine.predict_bgr(
            np.full((224, 224, 3), 64, dtype=np.uint8),
            "track the face",
        )

        assert result["command"] == "track the face"
        assert isinstance(result["pan_vel"], float)
        assert isinstance(result["tilt_vel"], float)
        assert result["inference_latency_ms"] >= 0.0

    def test_shared_engine_predicts_arbitrary_command(self, tmp_path, monkeypatch):
        """Shared inference engine tokenizes uncached commands at runtime."""
        model_path = _export_onnx(tmp_path, model=_InputSensitiveModel())
        token_cache = _make_token_cache(tmp_path)
        engine = VLAInferenceEngine(
            checkpoint=model_path,
            token_cache=token_cache,
            providers=["CPUExecutionProvider"],
        )

        calls = {"loads": 0, "tokenizes": 0}

        class _Tokenizer:
            def __call__(self, text, return_tensors, padding, truncation, max_length):
                calls["tokenizes"] += 1
                return {
                    "input_ids": np.full((1, 77), 5, dtype=np.int64),
                    "attention_mask": np.ones((1, 77), dtype=np.int64),
                }

        class _TokenizerClass:
            @staticmethod
            def from_pretrained(model_id):
                calls["loads"] += 1
                return _Tokenizer()

        transformers_mod = mock.MagicMock(CLIPTokenizerFast=_TokenizerClass)
        monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

        command = "turn slightly left and keep the face centered"
        first = engine.predict_bgr(np.full((224, 224, 3), 64, dtype=np.uint8), command)
        second = engine.predict_bgr(np.full((224, 224, 3), 64, dtype=np.uint8), command)

        assert first["command"] == command
        assert second["command"] == command
        assert calls == {"loads": 1, "tokenizes": 1}
