---
id: oce-mas6
status: open
deps: []
links: []
created: 2026-04-05T21:15:58Z
type: task
priority: 3
assignee: Nathan Clonts
tags: [phase4, launch, reliability]
---
# VLA graceful fallback to classical tracker

Phase 4 Step 3c. If the VLA node crashes (missing ONNX file, corrupt model, ORT error), the robot is left with no controller. Add an OnProcessExit handler in tracker_launch.py that detects VLA failure and starts tracker_node as fallback.

Implementation:
- Wrap vla_node launch in an action variable
- Add OnProcessExit(target_action=vla_node, on_exit=[LogInfo('VLA failed — falling back'), tracker_node])
- Only trigger fallback on non-zero exit code (normal shutdown shouldn't start the tracker)

This ensures the robot always has an active controller.

## Acceptance Criteria

- VLA crash (e.g. missing model file) automatically starts classical tracker
- Normal VLA shutdown (Ctrl+C) does NOT trigger fallback
- Tested by launching with a bad checkpoint path

