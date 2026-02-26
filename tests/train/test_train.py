"""tests/train/test_train.py — unit tests for train/train.py.

All tests use VLAModel(pretrained=False) — no HuggingFace downloads.
Runs on CPU; no real dataset or tokenizer required.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from train.model import VLAModel
from train.train import evaluate, train_one_epoch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

B       = 4   # batch size
SEQ_LEN = 10  # tokenised sequence length


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_batch(
    label_keys: list[str] | None = None,
    targets: torch.Tensor | None = None,
) -> dict:
    """Create a collate_fn-style batch dict on CPU."""
    if label_keys is None:
        label_keys = ["basic_track"] * B
    if targets is None:
        targets = torch.randn(B, 2)
    return {
        "frames":         torch.randn(B, 3, 224, 224),
        "input_ids":      torch.randint(0, 1000, (B, SEQ_LEN)),
        "attention_mask": torch.ones(B, SEQ_LEN, dtype=torch.long),
        "targets":        targets,
        "label_keys":     label_keys,
    }


class FakeLoader:
    """Yields the same batch N times (simulates a DataLoader epoch)."""

    def __init__(self, batch: dict, n: int = 3):
        self._batch = batch
        self._n = n

    def __iter__(self):
        for _ in range(self._n):
            yield self._batch

    def __len__(self) -> int:
        return self._n


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model() -> VLAModel:
    return VLAModel(pretrained=False)


@pytest.fixture
def criterion() -> nn.MSELoss:
    return nn.MSELoss()


@pytest.fixture
def optimizer(model: VLAModel) -> torch.optim.Adam:
    return torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
    )


# ---------------------------------------------------------------------------
# train_one_epoch
# ---------------------------------------------------------------------------

class TestTrainOneEpoch:

    def test_returns_float(self, model, criterion, optimizer):
        loss = train_one_epoch(
            model, FakeLoader(make_batch()), optimizer, criterion,
            torch.device("cpu"), scaler=None,
        )
        assert isinstance(loss, float)

    def test_loss_is_non_negative(self, model, criterion, optimizer):
        loss = train_one_epoch(
            model, FakeLoader(make_batch()), optimizer, criterion,
            torch.device("cpu"), scaler=None,
        )
        assert loss >= 0.0

    def test_loss_is_finite(self, model, criterion, optimizer):
        loss = train_one_epoch(
            model, FakeLoader(make_batch()), optimizer, criterion,
            torch.device("cpu"), scaler=None,
        )
        assert loss == loss        # not NaN
        assert loss < float("inf")

    def test_trainable_params_update(self, model, criterion, optimizer):
        batch = make_batch()
        # Snapshot trainable params before training
        before = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        train_one_epoch(
            model, FakeLoader(batch, n=1), optimizer, criterion,
            torch.device("cpu"), scaler=None,
        )
        changed = any(
            not torch.equal(p.detach(), before[n])
            for n, p in model.named_parameters()
            if p.requires_grad
        )
        assert changed, "No trainable parameters changed after training step"

    def test_frozen_params_unchanged(self, model, criterion, optimizer):
        batch = make_batch()
        frozen_before = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
            if not p.requires_grad
        }
        train_one_epoch(
            model, FakeLoader(batch, n=1), optimizer, criterion,
            torch.device("cpu"), scaler=None,
        )
        for n, p in model.named_parameters():
            if not p.requires_grad:
                assert torch.equal(p.detach(), frozen_before[n]), (
                    f"Frozen param changed during training: {n}"
                )

    def test_model_in_train_mode_after_call(self, model, criterion, optimizer):
        train_one_epoch(
            model, FakeLoader(make_batch()), optimizer, criterion,
            torch.device("cpu"), scaler=None,
        )
        # Model itself should be in training mode …
        assert model.training
        # … but frozen encoders must remain in eval mode
        assert not model.dino.training
        assert not model.clip_text.training

    def test_empty_loader_returns_nan(self, model, criterion, optimizer):
        loss = train_one_epoch(
            model, FakeLoader(make_batch(), n=0), optimizer, criterion,
            torch.device("cpu"), scaler=None,
        )
        import math
        assert math.isnan(loss), "Expected NaN for empty loader"


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:

    def test_returns_tuple(self, model, criterion):
        result = evaluate(model, FakeLoader(make_batch()), criterion, torch.device("cpu"))
        assert isinstance(result, tuple) and len(result) == 2

    def test_mean_loss_is_float(self, model, criterion):
        mean_loss, _ = evaluate(
            model, FakeLoader(make_batch()), criterion, torch.device("cpu")
        )
        assert isinstance(mean_loss, float)

    def test_per_label_is_dict(self, model, criterion):
        _, per_label = evaluate(
            model, FakeLoader(make_batch()), criterion, torch.device("cpu")
        )
        assert isinstance(per_label, dict)

    def test_mean_loss_is_non_negative(self, model, criterion):
        mean_loss, _ = evaluate(
            model, FakeLoader(make_batch()), criterion, torch.device("cpu")
        )
        assert mean_loss >= 0.0

    def test_mean_loss_is_finite(self, model, criterion):
        mean_loss, _ = evaluate(
            model, FakeLoader(make_batch()), criterion, torch.device("cpu")
        )
        assert mean_loss == mean_loss        # not NaN
        assert mean_loss < float("inf")

    def test_per_label_keys_match_input(self, model, criterion):
        keys = ["basic_track", "fast_follow", "basic_track", "slow_follow"]
        batch = make_batch(label_keys=keys)
        _, per_label = evaluate(
            model, FakeLoader(batch), criterion, torch.device("cpu")
        )
        assert set(per_label.keys()) == {"basic_track", "fast_follow", "slow_follow"}

    def test_per_label_values_non_negative(self, model, criterion):
        _, per_label = evaluate(
            model, FakeLoader(make_batch()), criterion, torch.device("cpu")
        )
        for k, v in per_label.items():
            assert v >= 0.0, f"Negative per-label MSE for '{k}': {v}"

    def test_model_in_eval_mode_after_call(self, model, criterion):
        evaluate(model, FakeLoader(make_batch()), criterion, torch.device("cpu"))
        assert not model.training

    def test_single_label_per_label_matches_mean_loss(self, model, criterion):
        """When all samples share one label, per_label[k] ≈ mean_loss."""
        batch = make_batch(label_keys=["track"] * B)
        mean_loss, per_label = evaluate(
            model, FakeLoader(batch, n=1), criterion, torch.device("cpu")
        )
        # per_label MSE is per-sample mean (mean over dim=1),
        # mean_loss is per-batch mean (mean over B and 2 outputs combined).
        # Both should be close for a single batch with one label type.
        assert "track" in per_label
        assert abs(per_label["track"] - mean_loss) < 1e-4, (
            f"per_label mse={per_label['track']:.6f} != mean_loss={mean_loss:.6f}"
        )

    def test_empty_loader_returns_nan(self, model, criterion):
        import math
        mean_loss, per_label = evaluate(
            model, FakeLoader(make_batch(), n=0), criterion, torch.device("cpu")
        )
        assert math.isnan(mean_loss)
        assert per_label == {}


# ---------------------------------------------------------------------------
# Overfit (integration)
# ---------------------------------------------------------------------------

class TestOverfit:
    """Model should overfit a single repeated batch with enough gradient steps.

    Note: stub encoders always return constant tensors regardless of input, so
    training is pure parameter optimisation (fixed input → fixed target).  A
    stable learning rate (3e-4) with enough steps should reliably converge.
    lr=1e-2 overshoots and saturates the 2*tanh output bound, so we use the
    same default lr as the real training script.
    """

    def test_loss_decreases_over_training_steps(self):
        torch.manual_seed(0)
        model     = VLAModel(pretrained=False)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=3e-4,
        )
        # Targets well within the output range (±2) — non-trivial to hit from init
        batch  = make_batch(targets=torch.full((B, 2), 1.2))
        loader = FakeLoader(batch, n=1)

        val_loss_before, _ = evaluate(model, loader, criterion, torch.device("cpu"))
        for _ in range(100):
            train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"), None)
        val_loss_after, _ = evaluate(model, loader, criterion, torch.device("cpu"))

        assert val_loss_after < val_loss_before * 0.5, (
            f"Loss did not decrease by 50%: before={val_loss_before:.5f} "
            f"after={val_loss_after:.5f}"
        )

    def test_val_loss_lower_after_training(self):
        torch.manual_seed(1)
        model     = VLAModel(pretrained=False)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=3e-4,
        )
        # Use a non-trivial target that requires the model to change its output
        batch  = make_batch(targets=torch.full((B, 2), -1.0))
        loader = FakeLoader(batch, n=1)

        val_loss_before, _ = evaluate(model, loader, criterion, torch.device("cpu"))
        for _ in range(100):
            train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"), None)
        val_loss_after, _ = evaluate(model, loader, criterion, torch.device("cpu"))

        assert val_loss_after < val_loss_before, (
            f"Val loss did not improve: before={val_loss_before:.5f} "
            f"after={val_loss_after:.5f}"
        )
