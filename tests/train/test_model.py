"""tests/train/test_model.py — unit tests for VLAModel.

All tests use pretrained=False (lightweight stubs, no HuggingFace downloads).
Runs on CPU with no GPU required.
"""

from __future__ import annotations

import pytest
import torch

from train.model import VLAModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

B       = 2    # batch size used throughout
SEQ_LEN = 10   # tokenised sequence length (< 77 — tests variable-length text)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model() -> VLAModel:
    return VLAModel(pretrained=False)


@pytest.fixture
def inputs():
    """Random frame + token tensors on CPU."""
    frames         = torch.randn(B, 3, 224, 224)
    input_ids      = torch.randint(0, 1000, (B, SEQ_LEN))
    attention_mask = torch.ones(B, SEQ_LEN, dtype=torch.long)
    return frames, input_ids, attention_mask


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

class TestForward:

    def test_output_shape(self, model, inputs):
        frames, ids, mask = inputs
        out = model(frames, ids, mask)
        assert out.shape == (B, 2), f"Expected ({B}, 2), got {out.shape}"

    def test_output_dtype(self, model, inputs):
        frames, ids, mask = inputs
        out = model(frames, ids, mask)
        assert out.dtype == torch.float32

    def test_output_is_finite(self, model, inputs):
        frames, ids, mask = inputs
        out = model(frames, ids, mask)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"

    def test_batch_size_1(self, model):
        out = model(
            torch.randn(1, 3, 224, 224),
            torch.randint(0, 1000, (1, SEQ_LEN)),
            torch.ones(1, SEQ_LEN, dtype=torch.long),
        )
        assert out.shape == (1, 2)

    def test_batch_size_8(self, model):
        out = model(
            torch.randn(8, 3, 224, 224),
            torch.randint(0, 1000, (8, SEQ_LEN)),
            torch.ones(8, SEQ_LEN, dtype=torch.long),
        )
        assert out.shape == (8, 2)

    def test_output_pan_tilt_separate(self, model, inputs):
        """Verify the two output channels are accessible by index."""
        frames, ids, mask = inputs
        out = model(frames, ids, mask)
        assert out[:, 0].shape == (B,), "pan_vel channel wrong shape"
        assert out[:, 1].shape == (B,), "tilt_vel channel wrong shape"

    def test_padding_mask_applied(self, model):
        """Output should differ when padding positions are masked vs unmasked.

        Construct a batch where the second half of the sequence is padding.
        Comparing masked vs all-ones mask verifies key_padding_mask is wired up.
        """
        frames    = torch.randn(1, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (1, SEQ_LEN))

        # All tokens real
        mask_full = torch.ones(1, SEQ_LEN, dtype=torch.long)
        # Only first 3 tokens real, rest padding
        mask_partial = torch.cat([
            torch.ones(1, 3, dtype=torch.long),
            torch.zeros(1, SEQ_LEN - 3, dtype=torch.long),
        ], dim=1)

        out_full    = model(frames, input_ids, mask_full)
        out_partial = model(frames, input_ids, mask_partial)

        # With different padding masks the cross-attention keys/values differ →
        # outputs must differ (unless all attention weights happen to be equal,
        # which is astronomically unlikely with random init weights).
        assert not torch.allclose(out_full, out_partial), (
            "Output identical for full vs partial attention mask — "
            "key_padding_mask may not be wired up"
        )

    def test_eval_mode_no_side_effects(self, model, inputs):
        """model.eval() should not break the forward pass."""
        frames, ids, mask = inputs
        model.eval()
        with torch.no_grad():
            out = model(frames, ids, mask)
        assert out.shape == (B, 2)
        model.train()


# ---------------------------------------------------------------------------
# Gradient / requires_grad behaviour
# ---------------------------------------------------------------------------

class TestGradients:

    def test_frozen_params_require_no_grad(self, model):
        """Encoder params must have requires_grad=False after construction."""
        for name, p in model.named_parameters():
            if "dino" in name or "clip_text" in name:
                assert not p.requires_grad, \
                    f"Frozen param '{name}' has requires_grad=True"

    def test_trainable_params_require_grad(self, model):
        """txt_proj, fusion, fusion_norms, and action_head params must be trainable."""
        trainable = [
            (n, p) for n, p in model.named_parameters()
            if "dino" not in n and "clip_text" not in n
        ]
        assert len(trainable) > 0, "No trainable parameters found"
        for name, p in trainable:
            assert p.requires_grad, \
                f"Expected trainable param '{name}' to have requires_grad=True"

    def test_backward_only_trains_trainable_params(self, model, inputs):
        """After loss.backward(), frozen params have no grad and trainable ones do."""
        frames, ids, mask = inputs

        model.zero_grad()
        out  = model(frames, ids, mask)
        loss = out.sum()
        loss.backward()

        for name, p in model.named_parameters():
            if "dino" in name or "clip_text" in name:
                assert p.grad is None, \
                    f"Frozen param '{name}' unexpectedly received a gradient"
            else:
                assert p.grad is not None, \
                    f"Trainable param '{name}' has no gradient after backward()"

    def test_fusion_norms_receive_gradients(self, model, inputs):
        """LayerNorm layers in fusion_norms must receive gradients."""
        frames, ids, mask = inputs
        model.zero_grad()
        model(frames, ids, mask).sum().backward()
        for i, norm in enumerate(model.fusion_norms):
            assert norm.weight.grad is not None, \
                f"fusion_norms[{i}].weight has no gradient"
            assert norm.bias.grad is not None, \
                f"fusion_norms[{i}].bias has no gradient"

    def test_zero_grad_clears_gradients(self, model, inputs):
        """model.zero_grad() should clear all accumulated gradients."""
        frames, ids, mask = inputs
        model(frames, ids, mask).sum().backward()
        model.zero_grad()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert p.grad.abs().max() == 0, \
                    f"Gradient not cleared for '{name}'"


# ---------------------------------------------------------------------------
# Architecture / hyperparameters
# ---------------------------------------------------------------------------

class TestArchitecture:

    def test_n_fusion_layers_default(self, model):
        assert len(model.fusion) == 2
        assert len(model.fusion_norms) == 2

    @pytest.mark.parametrize("n_layers", [1, 3, 4])
    def test_n_fusion_layers_param(self, n_layers):
        m = VLAModel(n_fusion_layers=n_layers, pretrained=False)
        assert len(m.fusion) == n_layers
        assert len(m.fusion_norms) == n_layers

    def test_fusion_and_norms_same_length(self, model):
        assert len(model.fusion) == len(model.fusion_norms)

    def test_trainable_param_count(self, model):
        """Trainable params should be in a reasonable range (100k – 5M)."""
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert 100_000 < n < 5_000_000, \
            f"Unexpected trainable param count: {n:,} — check architecture dimensions"

    def test_frozen_stubs_have_no_grad_params(self, model):
        """Both encoder stubs must have all their params frozen."""
        for name, p in model.named_parameters():
            if "dino" in name or "clip_text" in name:
                assert not p.requires_grad

    def test_action_head_has_two_outputs(self):
        """Action head final layer must output dim=2."""
        m = VLAModel(pretrained=False)
        last = list(m.action_head.children())[-1]
        assert isinstance(last, torch.nn.Linear)
        assert last.out_features == 2

    def test_fusion_norm_hidden_dim(self, model):
        """Each LayerNorm in fusion_norms must match VIS_DIM."""
        for i, norm in enumerate(model.fusion_norms):
            assert norm.normalized_shape == (VLAModel.VIS_DIM,), \
                f"fusion_norms[{i}] has wrong shape: {norm.normalized_shape}"


# ---------------------------------------------------------------------------
# Output bounds (scaled tanh)
# ---------------------------------------------------------------------------

class TestOutputBounds:

    def test_output_in_range(self, model, inputs):
        """Scaled tanh output must be strictly within (−2, 2) rad/s."""
        frames, ids, mask = inputs
        out = model(frames, ids, mask)
        assert (out.abs() < 2.0).all(), f"Output outside (−2, 2): {out}"

    def test_output_bounded_with_extreme_inputs(self, model):
        """Even with large-magnitude pixel values, tanh keeps outputs within (−2, 2)."""
        frames = torch.randn(4, 3, 224, 224) * 100
        ids    = torch.randint(0, 1000, (4, SEQ_LEN))
        mask   = torch.ones(4, SEQ_LEN, dtype=torch.long)
        out = model(frames, ids, mask)
        assert (out.abs() < 2.0).all(), f"Output not bounded for extreme inputs: {out}"


# ---------------------------------------------------------------------------
# Dropout behaviour
# ---------------------------------------------------------------------------

class TestDropout:

    def test_train_mode_stochastic(self, model):
        """Repeated forward passes in train mode differ due to dropout."""
        # Use batch=8 for good statistical power (p(all same) ≈ 0^256 with p=0.1)
        frames = torch.randn(8, 3, 224, 224)
        ids    = torch.randint(0, 1000, (8, SEQ_LEN))
        mask   = torch.ones(8, SEQ_LEN, dtype=torch.long)
        model.train()
        out1 = model(frames, ids, mask)
        out2 = model(frames, ids, mask)
        assert not torch.allclose(out1, out2), (
            "Outputs identical across two train-mode forward passes — "
            "dropout may be missing or p=0"
        )

    def test_eval_mode_deterministic(self, model, inputs):
        """Repeated forward passes in eval mode are identical (dropout disabled)."""
        frames, ids, mask = inputs
        model.eval()
        with torch.no_grad():
            out1 = model(frames, ids, mask)
            out2 = model(frames, ids, mask)
        model.train()
        assert torch.allclose(out1, out2), (
            "Outputs differ across two eval-mode forward passes — "
            "dropout not disabled by model.eval()"
        )

    def test_zero_dropout_deterministic_in_train(self):
        """dropout=0.0 model is deterministic in train mode (sanity check)."""
        m = VLAModel(pretrained=False, dropout=0.0)
        m.train()
        frames = torch.randn(4, 3, 224, 224)
        ids    = torch.randint(0, 1000, (4, SEQ_LEN))
        mask   = torch.ones(4, SEQ_LEN, dtype=torch.long)
        out1 = m(frames, ids, mask)
        out2 = m(frames, ids, mask)
        assert torch.allclose(out1, out2), (
            "dropout=0.0 model produced different outputs in train mode"
        )


# ---------------------------------------------------------------------------
# Frozen encoder eval mode
# ---------------------------------------------------------------------------

class TestFrozenEncoderEvalMode:

    def test_encoders_in_eval_at_construction(self, model):
        """Frozen encoders must be in eval mode immediately after construction."""
        assert not model.dino.training, "dino should be in eval mode after __init__"
        assert not model.clip_text.training, "clip_text should be in eval mode after __init__"

    def test_encoders_stay_eval_after_model_train(self, model):
        """model.train() must NOT flip frozen encoders to training mode.

        Frozen parameters do not receive gradients, but .training still controls
        dropout and stochastic-depth inside the encoder. If encoders switch to
        train mode, they inject noise into what should be a deterministic frozen
        forward pass.
        """
        model.train()
        assert not model.dino.training, (
            "dino switched to training mode after model.train() — "
            "override train() to keep frozen encoders in eval"
        )
        assert not model.clip_text.training, (
            "clip_text switched to training mode after model.train() — "
            "override train() to keep frozen encoders in eval"
        )

    def test_encoders_stay_eval_after_model_eval_then_train(self, model):
        """Cycling eval() → train() must still keep encoders in eval mode."""
        model.eval()
        model.train()
        assert not model.dino.training
        assert not model.clip_text.training
