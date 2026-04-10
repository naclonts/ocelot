"""train/transforms.py — training-only domain randomization utilities.

The transforms operate on single-frame tensors shaped (3, H, W) with values
in [0, 1]. They are intentionally pure PyTorch so they can run in the
DataLoader worker process without extra dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DomainRandomizationConfig:
    """Configurable probabilities and strengths for training augmentations."""

    color_jitter_prob: float = 1.0
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.1
    blur_prob: float = 0.3
    blur_kernel_min: int = 3
    blur_kernel_max: int = 7
    blur_sigma_min: float = 0.1
    blur_sigma_max: float = 2.0
    noise_prob: float = 0.3
    noise_sigma_max: float = 0.05
    cutout_prob: float = 0.2
    cutout_min_scale: float = 0.02
    cutout_max_scale: float = 0.08
    cutout_max_patches: int = 1
    gradient_prob: float = 0.3
    gradient_strength_min: float = 0.08
    gradient_strength_max: float = 0.25


class DomainRandomizationTransform(nn.Module):
    """Apply sim-to-real style augmentations to a single training frame."""

    def __init__(self, config: DomainRandomizationConfig | None = None) -> None:
        super().__init__()
        self.config = config or DomainRandomizationConfig()

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.ndim != 3 or frame.shape[0] != 3:
            raise ValueError(f"expected (3, H, W) frame, got {tuple(frame.shape)}")

        frame = frame.to(dtype=torch.float32)
        frame = frame.clamp(0.0, 1.0)

        if torch.rand(()) < self.config.color_jitter_prob:
            frame = self._apply_color_jitter(frame)
        if torch.rand(()) < self.config.gradient_prob:
            frame = self._apply_brightness_gradient(frame)
        if torch.rand(()) < self.config.blur_prob:
            frame = self._apply_gaussian_blur(frame)
        if torch.rand(()) < self.config.noise_prob:
            sigma = torch.empty(()).uniform_(0.0, self.config.noise_sigma_max).item()
            if sigma > 0:
                frame = (frame + torch.randn_like(frame) * sigma).clamp(0.0, 1.0)
        if torch.rand(()) < self.config.cutout_prob:
            frame = self._apply_cutout(frame)

        return frame.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Color jitter
    # ------------------------------------------------------------------

    def _apply_color_jitter(self, frame: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        if cfg.brightness > 0:
            factor = torch.empty(()).uniform_(1.0 - cfg.brightness, 1.0 + cfg.brightness).item()
            frame = (frame * factor).clamp(0.0, 1.0)
        if cfg.contrast > 0:
            factor = torch.empty(()).uniform_(1.0 - cfg.contrast, 1.0 + cfg.contrast).item()
            mean = frame.mean(dim=(1, 2), keepdim=True)
            frame = ((frame - mean) * factor + mean).clamp(0.0, 1.0)
        if cfg.saturation > 0:
            factor = torch.empty(()).uniform_(1.0 - cfg.saturation, 1.0 + cfg.saturation).item()
            gray = _rgb_to_grayscale(frame)
            frame = ((frame - gray) * factor + gray).clamp(0.0, 1.0)
        if cfg.hue > 0:
            delta = torch.empty(()).uniform_(-cfg.hue, cfg.hue).item()
            hsv = _rgb_to_hsv(frame)
            hsv[0] = (hsv[0] + delta) % 1.0
            frame = _hsv_to_rgb(hsv).clamp(0.0, 1.0)
        return frame

    # ------------------------------------------------------------------
    # Brightness gradient
    # ------------------------------------------------------------------

    def _apply_brightness_gradient(self, frame: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        strength = torch.empty(()).uniform_(
            cfg.gradient_strength_min, cfg.gradient_strength_max
        ).item()
        if torch.rand(()) < 0.5:
            axis = 2
        else:
            axis = 1

        if axis == 2:
            ramp = torch.linspace(-1.0, 1.0, frame.shape[2], device=frame.device, dtype=frame.dtype)
            mask = 1.0 + strength * ramp.view(1, 1, -1)
        else:
            ramp = torch.linspace(-1.0, 1.0, frame.shape[1], device=frame.device, dtype=frame.dtype)
            mask = 1.0 + strength * ramp.view(1, -1, 1)

        return (frame * mask).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Gaussian blur
    # ------------------------------------------------------------------

    def _apply_gaussian_blur(self, frame: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        kernel_size = _sample_odd_int(cfg.blur_kernel_min, cfg.blur_kernel_max)
        sigma = torch.empty(()).uniform_(cfg.blur_sigma_min, cfg.blur_sigma_max).item()
        kernel = _gaussian_kernel_2d(kernel_size, sigma, frame.device, frame.dtype)

        x = frame.unsqueeze(0)
        pad = kernel_size // 2
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)
        blurred = F.conv2d(x, kernel, groups=3)
        return blurred.squeeze(0).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Random erasing / cutout
    # ------------------------------------------------------------------

    def _apply_cutout(self, frame: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        n_patches = max(0, int(cfg.cutout_max_patches))
        if n_patches == 0:
            return frame
        h, w = frame.shape[1:]
        result = frame.clone()

        for _ in range(n_patches):
            erase_h = max(1, int(torch.empty(()).uniform_(
                cfg.cutout_min_scale * h, cfg.cutout_max_scale * h
            ).item()))
            erase_w = max(1, int(torch.empty(()).uniform_(
                cfg.cutout_min_scale * w, cfg.cutout_max_scale * w
            ).item()))
            if erase_h >= h or erase_w >= w:
                continue

            top = int(torch.randint(0, h - erase_h + 1, ()).item())
            left = int(torch.randint(0, w - erase_w + 1, ()).item())
            result[:, top:top + erase_h, left:left + erase_w] = 0.0

        return result


def _sample_odd_int(min_size: int, max_size: int) -> int:
    candidates = [n for n in range(min_size, max_size + 1) if n % 2 == 1]
    if not candidates:
        raise ValueError(f"no odd kernel size in range [{min_size}, {max_size}]")
    idx = int(torch.randint(0, len(candidates), ()).item())
    return candidates[idx]


def _rgb_to_grayscale(frame: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=frame.device, dtype=frame.dtype)
    return (frame * weights.view(3, 1, 1)).sum(dim=0, keepdim=True).expand_as(frame)


def _rgb_to_hsv(frame: torch.Tensor) -> torch.Tensor:
    r, g, b = frame[0], frame[1], frame[2]
    maxc, idx = frame.max(dim=0)
    minc = frame.min(dim=0).values
    delt = maxc - minc

    h = torch.zeros_like(maxc)
    s = torch.zeros_like(maxc)
    v = maxc

    nonzero = maxc > 0
    s[nonzero] = delt[nonzero] / maxc[nonzero]

    mask = delt > 1e-6
    r_max = (idx == 0) & mask
    g_max = (idx == 1) & mask
    b_max = (idx == 2) & mask

    h[r_max] = ((g[r_max] - b[r_max]) / delt[r_max]) % 6.0
    h[g_max] = 2.0 + (b[g_max] - r[g_max]) / delt[g_max]
    h[b_max] = 4.0 + (r[b_max] - g[b_max]) / delt[b_max]
    h = (h / 6.0) % 1.0

    return torch.stack([h, s, v], dim=0)


def _hsv_to_rgb(frame: torch.Tensor) -> torch.Tensor:
    h, s, v = frame[0], frame[1], frame[2]
    h = (h % 1.0) * 6.0
    i = torch.floor(h).to(torch.int64)
    f = h - i.to(h.dtype)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    i_mod = i % 6
    shape = h.shape
    out = torch.zeros((3, *shape), device=frame.device, dtype=frame.dtype)

    cases = {
        0: (v, t, p),
        1: (q, v, p),
        2: (p, v, t),
        3: (p, q, v),
        4: (t, p, v),
        5: (v, p, q),
    }
    for case, (r, g, b) in cases.items():
        mask = i_mod == case
        out[0][mask] = r[mask]
        out[1][mask] = g[mask]
        out[2][mask] = b[mask]

    return out


def _gaussian_kernel_2d(kernel_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel
