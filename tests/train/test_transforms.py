"""tests/train/test_transforms.py — unit tests for training augmentations."""

from __future__ import annotations

import torch

from train.transforms import DomainRandomizationConfig, DomainRandomizationTransform


def test_identity_when_all_probs_zero():
    frame = torch.rand(3, 32, 32)
    transform = DomainRandomizationTransform(
        DomainRandomizationConfig(
            color_jitter_prob=0.0,
            blur_prob=0.0,
            noise_prob=0.0,
            cutout_prob=0.0,
            gradient_prob=0.0,
        )
    )

    out = transform(frame)
    assert torch.allclose(out, frame)


def test_color_jitter_changes_frame():
    torch.manual_seed(0)
    frame = torch.full((3, 32, 32), 0.5)
    transform = DomainRandomizationTransform(
        DomainRandomizationConfig(
            color_jitter_prob=1.0,
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            blur_prob=0.0,
            noise_prob=0.0,
            cutout_prob=0.0,
            gradient_prob=0.0,
        )
    )

    out = transform(frame)
    assert out.shape == frame.shape
    assert torch.isfinite(out).all()
    assert (0.0 <= out).all() and (out <= 1.0).all()
    assert not torch.allclose(out, frame)


def test_gaussian_blur_spreads_energy():
    torch.manual_seed(0)
    frame = torch.zeros(3, 33, 33)
    frame[:, 16, 16] = 1.0
    transform = DomainRandomizationTransform(
        DomainRandomizationConfig(
            color_jitter_prob=0.0,
            blur_prob=1.0,
            blur_kernel_min=3,
            blur_kernel_max=3,
            blur_sigma_min=1.0,
            blur_sigma_max=1.0,
            noise_prob=0.0,
            cutout_prob=0.0,
            gradient_prob=0.0,
        )
    )

    out = transform(frame)
    assert out[:, 16, 16].mean() < 1.0
    assert out[:, 16, 17].mean() > 0.0


def test_noise_and_cutout_modify_frame():
    torch.manual_seed(0)
    frame = torch.full((3, 32, 32), 0.5)
    transform = DomainRandomizationTransform(
        DomainRandomizationConfig(
            color_jitter_prob=0.0,
            blur_prob=0.0,
            noise_prob=1.0,
            noise_sigma_max=0.05,
            cutout_prob=1.0,
            cutout_min_scale=0.25,
            cutout_max_scale=0.25,
            cutout_max_patches=1,
            gradient_prob=0.0,
        )
    )

    out = transform(frame)
    assert out.shape == frame.shape
    assert torch.isfinite(out).all()
    assert (out == 0.0).any()
    assert not torch.allclose(out, frame)


def test_brightness_gradient_changes_spatial_profile():
    torch.manual_seed(0)
    frame = torch.ones(3, 32, 32)
    transform = DomainRandomizationTransform(
        DomainRandomizationConfig(
            color_jitter_prob=0.0,
            blur_prob=0.0,
            noise_prob=0.0,
            cutout_prob=0.0,
            gradient_prob=1.0,
            gradient_strength_min=0.2,
            gradient_strength_max=0.2,
        )
    )

    out = transform(frame)
    assert out.min() < out.max()
    assert (0.0 <= out).all() and (out <= 1.0).all()
