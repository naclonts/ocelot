---
id: oce-jzkq
status: open
deps: [oce-ju2p]
links: []
created: 2026-04-05T21:16:15Z
type: task
priority: 2
assignee: Nathan Clonts
tags: [phase4, eval, hardware]
---
# A/B comparison: classical tracker vs VLA on hardware

Phase 4 Step 4c. Run the same test scenario (person sitting 1-2m away, slowly moving head) with both controllers and document the comparison.

Metrics to capture (using eval_hardware_node):
- Mean tracking error (px)
- p95 tracking error (px)  
- Face-lost rate (%)
- Control smoothness (cmd_vel angular velocity std dev)
- VLA-only: response to language commands (e.g. 'follow slowly', 'track person on left')

Run each controller for at least 60s per trial, 3+ trials. Record results in a markdown table in docs/ or as a note on this ticket.

The VLA doesn't need to beat the classical tracker on single-face — the value is language-conditional behavior.

## Acceptance Criteria

- Both controllers tested under same conditions (distance, lighting, movement pattern)
- Results documented with mean, p95, lost rate, smoothness
- At least 3 trials per controller
- VLA tested with 2+ distinct language commands

