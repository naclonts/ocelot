---
id: oce-r6fw
status: open
deps: []
links: []
created: 2026-04-05T21:16:44Z
type: task
priority: 2
assignee: Nathan Clonts
tags: [phase4, deploy, mlops]
---
# Deploy script: dev machine → Pi

Phase 4 Step 7c. Create scripts/deploy.sh that automates model deployment from the dev machine to the Pi.

Functionality:
- SCP models/vla.onnx and models/vla_tokens.json to Pi
- SSH to Pi, docker compose down, USE_VLA=true docker compose up -d
- Print log tail command for verification
- Support --rollback flag: dvc checkout models/vla.onnx --rev HEAD~1, then deploy previous version
- Configurable PI_HOST (default: pi@raspberrypi.local)

This replaces the current manual scp + ssh workflow.

## Acceptance Criteria

- scripts/deploy.sh deploys model and restarts robot stack
- --rollback reverts to previous model version
- PI_HOST is configurable via env var
- Script is idempotent (safe to run multiple times)

