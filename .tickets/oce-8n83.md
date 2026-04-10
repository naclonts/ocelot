---
id: oce-8n83
status: open
deps: [oce-srgo]
links: []
created: 2026-04-05T21:16:34Z
type: task
priority: 2
assignee: Nathan Clonts
tags: [phase4, training, sim2real]
---
# Fine-tuning on mixed sim + real data

Phase 4 Step 5c/5d. Fine-tune the sim-trained model on a mix of sim + real data to bridge the sim-to-real gap.

Prerequisites: rosbag_to_hdf5.py exists and real-world recordings are collected.

**train.py --checkpoint flag**: train/train.py has --resume (loads full training state). Add --checkpoint that loads only the model state_dict (not optimizer/scheduler) for fine-tuning from a different run. --resume continues a run; --checkpoint starts a new run with pretrained weights.

**Mixed dataset**: Merge sim and real HDF5 shards into one dataset dir. Start with ~10% real data. Monitor:
- val_loss on sim test split (must still pass eval gate)
- Real-world tracking error (from eval_hardware_node)

Adjust real data ratio: 5% if sim perf degrades, 20% if real tracking still poor.

## Acceptance Criteria

- train.py supports --checkpoint for fine-tuning (loads model weights only, starts fresh optimizer)
- Fine-tuned model passes sim eval gate (overall MSE < 0.05)
- Fine-tuned model achieves < 20px mean tracking error on real hardware
- MLflow experiment tracks both sim and real metrics

