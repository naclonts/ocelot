---
id: oce-ju2p
status: open
deps: []
links: []
created: 2026-04-05T21:16:05Z
type: task
priority: 1
assignee: Nathan Clonts
tags: [phase4, eval, hardware]
---
# Hardware tracking eval node

Phase 4 Step 4b. Create a node that measures real-world tracking quality on Pi hardware. Subscribes to /camera/image_raw, runs Haar cascade face detection, computes pixel error from image center to face center, and logs rolling statistics.

Create ocelot/eval_hardware_node.py:
- Subscribe to /camera/image_raw
- Run Haar cascade detection per frame
- Compute pixel distance from image center to detected face center
- Track face-lost rate (no detection = -1)
- Every 100 frames: log mean error, p95, face-lost rate
- Publish diagnostics to /tracking/eval topic
- Add entry point in setup.py

This node can run alongside either the classical tracker or VLA node. It's the primary tool for A/B comparison (Step 4c) and measuring sim-to-real gap.

## Acceptance Criteria

- eval_hardware_node runs on Pi alongside VLA or tracker
- Logs mean_err, p95, face_lost_rate every 100 frames
- Works with real camera feed (Haar cascade detects real faces)
- Entry point in setup.py

