---
id: oce-cpqj
status: open
deps: []
links: []
created: 2026-04-05T21:15:52Z
type: task
priority: 2
assignee: Nathan Clonts
tags: [phase4, optimization, edge]
---
# Quantization script and text encoder caching

Formalize model optimization for Pi 5 edge deployment (Phase 4 Step 2).

Two sub-tasks:

**2b — train/quantize.py**: INT8 quantization already done ad-hoc (models/vla_int8.onnx.dvc exists), but there's no repeatable script. Create train/quantize.py that wraps onnxruntime.quantization.quantize_dynamic, runs eval_onnx.py on the result, and reports pass/fail. Should support QInt8 and optionally selective quantization (encoders only) if full-model quant degrades accuracy.

**2d — Text encoder caching / split model**: The CLIP text encoder (~63M params) processes the same command string every frame. Export two ONNX models: vla_text_encoder.onnx (input_ids, attention_mask → text_features) and vla_vision_head.onnx (frames, text_features → actions). Update vla_node.py to run the text encoder once at startup and cache the output. This could cut per-frame latency significantly on Pi 5.

Benchmark before/after on Pi hardware.

## Acceptance Criteria

- train/quantize.py exists and produces a quantized ONNX that passes eval gate
- Split-model export option in train/export_onnx.py
- vla_node.py supports both single-model and split-model inference
- Documented latency improvement on Pi 5

