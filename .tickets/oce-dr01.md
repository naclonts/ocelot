---
id: oce-dr01
status: open
deps: []
links: [oce-wp85, oce-8n83]
created: 2026-04-09T00:00:00Z
type: task
priority: 2
assignee: Nathan Clonts
tags: [training, phase4, sim2real]
---
# Domain randomization for sim-to-real visual transfer

DINOv2 features are reasonably domain-agnostic, but real-world images differ from sim in lighting, color balance, background complexity, and face appearance. When the model is uncertain on out-of-distribution inputs, predictions regress toward the training mean (which has a slight positive tilt bias), contributing to drift on hardware.

Add domain randomization augmentations to the training pipeline to make learned features more robust to visual domain shift.

## Augmentations to add (in train/dataset.py or a separate transforms module)

Applied randomly per-frame during training only (not val/test):
- **Color jitter**: brightness +-20%, contrast +-20%, saturation +-20%, hue +-10%
- **Gaussian blur**: kernel 3-7, sigma 0.1-2.0, probability ~30%
- **Gaussian noise**: sigma 0-0.05 (on normalized [0,1] image), probability ~30%
- **Random erasing / cutout**: small rectangular patches occluded, probability ~20%
- **Brightness gradient**: simulate uneven lighting (e.g. brighter on one side)

These should be applied *before* ImageNet normalization and after the uint8->float32 conversion.

## Acceptance Criteria

- Augmentation pipeline added to training with configurable probability per transform
- Augmentations disabled for val/test splits
- Retrained model with augmentations still passes sim eval gate (MSE < 0.05)
- Visual inspection: augmented training frames look plausibly real-world-like
- Compare sim eval MSE with/without augmentations (small regression acceptable if real-world tracking improves)
