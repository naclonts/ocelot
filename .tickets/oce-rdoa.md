---
id: oce-rdoa
status: open
deps: []
links: []
created: 2026-04-05T21:16:49Z
type: task
priority: 3
assignee: Nathan Clonts
tags: [phase4, vla, ux]
---
# Dynamic language command switching at runtime

Phase 4 Step 8a. vla_node.py currently uses a fixed command set at startup. Add dynamic reconfiguration so commands can be changed at runtime via ros2 param set.

Changes to ocelot/vla_node.py:
- Add add_on_set_parameters_callback
- When 'command' parameter changes: look up new command in token cache, update _input_ids and _attention_mask, re-cache text encoder output (if split model is implemented)
- Log the command switch

Usage:
  ros2 param set /vla_node command 'follow the person on the left'

This is a prerequisite for any future UI (web, voice) that switches commands.

## Acceptance Criteria

- ros2 param set /vla_node command '<new cmd>' works at runtime
- Token cache lookup handles exact and fuzzy matches
- Robot behavior changes visibly when switching commands
- Tested with at least 2 distinct commands

