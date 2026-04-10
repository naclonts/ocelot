---
id: oce-xvnw
status: open
deps: []
links: []
created: 2026-04-05T21:16:57Z
type: task
priority: 3
assignee: Nathan Clonts
tags: [phase4, production, reliability]
---
# Production hardening: watchdog, restart policy, thermal

Phase 4 Step 10. Harden the robot stack for reliable long-running operation.

**10a — Restart policy**: Add 'restart: unless-stopped' to docker-compose.yml ocelot service. Add watchdog in vla_node.py: if no camera frame for 5s, publish zero velocity and log warning (prevents drift on stale commands).

**10b — Thermal management**: Monitor Pi 5 CPU temp via /sys/class/thermal/thermal_zone0/temp. Skip every other frame if temp > 80°C. Log thermal throttling events.

**10c — Structured logging**: Write JSON logs to /ws/logs/ with inference latency, tracking error (when Haar cascade available), temperature, active command. Support post-hoc analysis.

These can be implemented incrementally.

## Acceptance Criteria

- Docker restart policy set to unless-stopped
- Watchdog publishes zero velocity after 5s with no camera frame
- Thermal throttling kicks in at 80°C (frame skip)
- JSON logs written with inference latency per frame

