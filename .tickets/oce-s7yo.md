---
id: oce-s7yo
status: open
deps: []
links: []
created: 2026-04-03T05:38:19Z
type: task
priority: 1
assignee: Nathan Clonts
tags: [training, phase3]
---
# Single-face-only training run to improve tracking quality

Current v0.1.0 model trained on all command types (track, multi_left, multi_right, multi_attr) simultaneously. Real-world and sim tracking performance is sketchy. Per-command eval shows single-face MSE ~0.004 which looks OK on paper but doesn't translate to good real behavior.

Hypothesis: the model is spending capacity learning multi-face selection and attribute matching, diluting single-face tracking quality — the only capability we actually need for Phase 4 hardware deployment.

Proposed approaches (pick one or combine):
1. **Single-face-only training run** — filter dataset to label_key=track only (607 val episodes, ~5k train episodes). Retrain from scratch. This eliminates multi-face/attribute commands entirely.
2. **Curriculum learning** — train on single-face first until convergence, then fine-tune on the full dataset.
3. **Reduce command vocabulary** — collapse all 6 single-face commands into 1-2 canonical forms to reduce language variance the model must handle.

Eval showed all 6 single-face commands perform nearly identically (MSE 0.0039–0.0041), so command phrasing isn't the issue — the model just needs more capacity/focus on the core tracking task.

Start with approach #1 as simplest experiment. Compare against v0.1.0 on single-face val episodes.

## Acceptance Criteria

- New model achieves measurably better single-face tracking in sim (lower MSE and visually smoother)
- Validated on hardware (Pi deployment) with acceptable real-world tracking
- ONNX export + eval pipeline passes gate

