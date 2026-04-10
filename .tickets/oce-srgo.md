---
id: oce-srgo
status: open
deps: []
links: []
created: 2026-04-05T21:16:25Z
type: task
priority: 2
assignee: Nathan Clonts
tags: [phase4, data, pipeline]
---
# Rosbag to HDF5 conversion pipeline

Phase 4 Step 5a/5b. Build the pipeline to convert real-world rosbag recordings into the HDF5 format used by OcelotDataset, enabling fine-tuning on real data.

**Recording** (5a): Already supported — RECORD=true launches ros2 bag record for /camera/image_raw and /cmd_vel. The classical tracker's /cmd_vel serves as ground-truth labels.

**Conversion** (5b): Create scripts/rosbag_to_hdf5.py:
- Read MCAP rosbag (ros2 bag record uses MCAP + zstd)
- Synchronize (image, cmd_vel) pairs by nearest timestamp
- Resize images to 224x224
- Extract pan_vel = cmd_vel.angular.z, tilt_vel = cmd_vel.angular.y
- Write HDF5 with same schema as sim episodes (frames, actions, commands)
- Use a fixed language command (e.g. 'track the face') or accept --command arg
- Use label_key prefix 'real_track' for separate tracking in eval

Should produce shards compatible with train/dataset.py OcelotDataset.

## Acceptance Criteria

- scripts/rosbag_to_hdf5.py converts MCAP bags to HDF5
- Output loads correctly via OcelotDataset
- label_key uses 'real_' prefix for real-world episodes
- Handles missing/dropped frames gracefully

