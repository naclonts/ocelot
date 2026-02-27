"""tests/train/test_vla_node.py — unit tests for ocelot/vla_node.py.

All tests run without ROS, without a GPU, and without Gazebo.
ROS2 packages (rclpy, cv_bridge, geometry_msgs, sensor_msgs) are mocked
at module level so the import succeeds in the host .venv.

Tests cover:
  - _preprocess: BGR HxWx3 → float32 (1,3,224,224) ImageNet-normalised
  - _find_best_command: exact / substring / fallback matching
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

from ocelot.vla_node import _find_best_command, _preprocess  # noqa: E402

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
# Tests: _find_best_command
# ---------------------------------------------------------------------------

class TestFindBestCommand:

    CACHE = {
        "track the face":    {},
        "follow slowly":     {},
        "look at the person": {},
    }

    def test_exact_match(self):
        assert _find_best_command("track the face", self.CACHE) == "track the face"

    def test_exact_match_second_key(self):
        assert _find_best_command("follow slowly", self.CACHE) == "follow slowly"

    def test_key_substring_of_request(self):
        """Cache key is a substring of the requested string."""
        result = _find_best_command("please track the face now", self.CACHE)
        assert result == "track the face"

    def test_request_substring_of_key(self):
        """Requested string is a substring of a cache key."""
        result = _find_best_command("follow", self.CACHE)
        assert result == "follow slowly"

    def test_fallback_to_track_the_face(self):
        """Unknown command falls back to 'track the face' when present."""
        result = _find_best_command("completely unknown xyz", self.CACHE)
        assert result == "track the face"

    def test_fallback_to_first_key_when_no_standard_key(self):
        """No standard fallbacks present → return first key."""
        cache = {"custom_a": {}, "custom_b": {}}
        result = _find_best_command("completely unknown xyz", cache)
        assert result == "custom_a"

    def test_look_at_fallback(self):
        """'look at the person' is used as a fallback when track/follow absent."""
        cache = {"look at the person": {}}
        result = _find_best_command("unknown command", cache)
        assert result == "look at the person"


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

        out_a = sess.run(["actions"], {"frames": frame_a, "input_ids": ids, "attention_mask": mask})[0]
        out_b = sess.run(["actions"], {"frames": frame_b, "input_ids": ids, "attention_mask": mask})[0]

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

        out_first  = sess.run(["actions"], {"frames": frame, "input_ids": ids, "attention_mask": mask})[0]
        out_second = sess.run(["actions"], {"frames": frame, "input_ids": ids, "attention_mask": mask})[0]

        np.testing.assert_array_equal(out_first, out_second)
