---
id: oce-nf01
status: closed
deps: []
links: [oce-wp85]
created: 2026-04-09T00:00:00Z
type: task
priority: 1
assignee: Nathan Clonts
tags: [training, phase4, vla]
---
# Add no-face / face-centered training data and confidence output head

The current VLA model always outputs a non-zero velocity because training data only contains frames where a face is present. When deployed on hardware, the model drifts (up and to the right) when no face is visible or the face is centered, because it has no training signal for "output zero."

## Part 1: No-face and face-centered training episodes

Collect additional training episodes in sim where:
- **No face present**: empty scene or only distractors. Oracle outputs (0, 0) for every frame.
- **Face perfectly centered**: face is stationary at camera center. Oracle outputs (0, 0) once converged.

This teaches the model that some visual inputs should produce zero velocity. Integrate these into the existing HDF5 dataset pipeline (new label_keys, e.g. `no_face`, `centered`).

Target: ~10-15% of total training frames should be zero-velocity examples.

## Part 2: Confidence / gate output head (optional, evaluate after Part 1)

Add a 3rd output to the model: `face_confidence in [0, 1]`. Architecture change:
- Branch after attention pooling: separate `nn.Linear(384, 1)` with sigmoid.
- Train with binary label: 1.0 when face is present, 0.0 when absent.
- At inference, multiply velocity outputs by confidence (or gate with a threshold).

This gives the model an explicit way to say "I don't see a face" rather than relying on the action head to output near-zero values.

Evaluate whether Part 1 alone is sufficient before implementing Part 2.

## Acceptance Criteria

- Data collection script supports no-face and face-centered episode types
- At least 500 no-face + 500 face-centered episodes collected
- Retrained model produces near-zero output on no-face frames (mean |vel| < 0.01)
- Retrained model still passes sim eval gate on tracking episodes (MSE < 0.05)
- If Part 2 implemented: confidence head achieves > 95% accuracy on face present/absent
