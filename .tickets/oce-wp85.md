---
id: oce-wp85
status: open
deps: []
links: []
created: 2026-04-04T07:13:15Z
type: bug
priority: 2
assignee: Nathan Clonts
tags: [vla, model, deploy]
---
# Investigate VLA model overshoot and upward tilt drift

The VLA model exhibits two issues during deployment:

1. **Overshoot in sim**: The model overshoots when tracking faces. This is
   surprising since it's a stateless per-frame model (no temporal state), yet
   it produces oscillatory behavior when approaching the target.

2. **Upward tilt drift on real hardware**: On the Pi, the model tracks
   initially but tends to drift upward (toward ceiling). The servo can move
   both up and down — this is a bias, not a sign inversion.

## Relevant data

### Oracle training labels (across ~841 episodes, 84100 frames):
- pan_vel:  mean=-0.00434, std=0.155, range [-0.3, 0.3]
- tilt_vel: mean=+0.00215, std=0.132, range [-0.3, 0.3]
- 22.1% of pan labels are at ±0.3 saturation
- 8.2% of tilt labels are at ±0.3 saturation
- ~20% of all labels are exactly zero (from oracle deadband)
- Slight positive tilt mean (+0.00215) — may contribute to upward drift

### Model architecture (stateless):
- Per-frame: image + text command → (pan_vel, tilt_vel)
- Output bounded by 2.0*tanh → [-2, 2] rad/s
- No temporal/recurrent component

### VLA node inference pipeline (vla_node.py):
- max_vel clips output to ±0.3 rad/s
- max_accel rate-limits at 1.5 rad/s² (max_delta = 0.15/frame at 10 Hz)
- Publishes to /cmd_vel (angular.z=pan, angular.y=tilt)

### Servo node (servo_node.py):
- Integrates cmd_vel at 30 Hz: pos += vel * velocity_scale * dt
- velocity_scale = 30.0
- tilt_center=180, tilt_limits=[90, 180]
- pan_center=90, pan_limits=[0, 180]

### v0.1.0 ONNX eval (833 val episodes):
- overall_mse: 0.00501
- track label mse: 0.00404
- Scatter correlation: pan r=0.850, tilt r=0.834

### Key files:
- ocelot/vla_node.py — ONNX inference node
- ocelot/servo_node.py — real hardware servo driver
- ocelot/oracle_node.py — oracle P-controller (training label source)
- train/model.py — VLAModel architecture
- runs/v0.1.0/eval_val_report.json — full eval report with per-command MSE
- runs/v0.1.0/eval_val/scatter.png — pred vs GT scatter plots
- runs/v0.1.0/eval_val/episodes.png — per-episode trajectory plots

## Suggested investigation areas:
- Analyze model prediction distribution near zero (does the model have
  trouble outputting zero? how does its zero-region behavior compare to
  oracle deadband?)
- Check whether tilt drift correlates with the +0.00215 tilt mean in
  training data, or if the servo integration loop amplifies small biases
- Examine the interaction between 10 Hz inference and 30 Hz servo
  integration (servo integrates stale velocity commands between camera frames)
- Look at the episode trajectory plots for overshoot patterns

## Root cause hypotheses

### H1: Rate limiter creates hidden temporal state → overshoot
The VLA model is stateless, but `vla_node.py` is not. The `max_accel=1.5`
rate limiter (lines 196-199) caps velocity change to ±0.15 rad/s per frame
at 10 Hz. Going from +0.3 → −0.3 takes 4 frames (400ms) of ramp-down, during
which the servo continues integrating the now-wrong velocity direction. The
episode trajectory plots confirm this: at sharp GT transitions the model
prediction responds within 1-2 frames, but the published (rate-limited)
velocity lags behind.

### H2: Model can't reproduce oracle deadband zeros → micro-oscillation
The oracle outputs exact 0.0 when error < 0.002 rad (~20% of all labels). The
model outputs `2.0*tanh(x)` — continuous, never exactly zero. The scatter plot
confirms predictions scatter ±0.02-0.05 when GT=0. These small non-zero
predictions get integrated by the servo at 30 Hz, producing micro-oscillations
around the target.

### H3: Velocity integration amplifies tiny prediction bias → tilt drift
At `velocity_scale=30`, a model bias of just 0.01 rad/s → 0.3 deg/s → 18
deg/min. The tilt range is 90° (limits [90,180]), so a 0.01 rad/s bias rails
the camera in ~5 minutes. The oracle doesn't drift because it's a closed-loop
P controller (error reverses when camera overshoots). The VLA is also
closed-loop (sees the image), but any systematic per-frame prediction bias
accumulates through integration.

### H4: 10 Hz inference / 30 Hz servo integration mismatch amplifies errors
Between camera frames (100ms), the servo runs 3 integration steps with stale
velocity. This multiplies the effect of any prediction error by 3× compared to
a matched-rate system. When the model wants to stop, the servo integrates the
previous non-zero command for 2 extra ticks before the correction arrives.

## Proposed experiments

### Experiment A: Zero-region prediction analysis (diagnostic)
Run the v0.1.0 ONNX on the val set and compute: mean prediction when |GT|<0.01
(per axis), overall mean prediction bias per axis, histogram of predictions vs
GT. Validates H2/H3 — tells us the magnitude of bias and zero-region spread.

### Experiment B: Add inference deadband to vla_node.py
Add `~deadband` parameter (default 0.02 rad/s). If `|predicted_vel| < deadband`,
snap to 0.0. Addresses H2. Test in sim with thresholds 0.01, 0.02, 0.05.

### Experiment C: Remove/relax the rate limiter
Set `max_accel` very large or remove clamping. The model was trained on oracle
data without rate limiting, so the limiter adds dynamics the model never learned
to compensate for. Validates H1. Test by replaying val predictions with and
without the limiter, measuring lag and overshoot.

### Experiment D: Bias correction at inference
After Experiment A, subtract the measured mean prediction bias per axis in
`vla_node.py`. Validates H3 as tilt drift root cause.

### Experiment E: Reduce velocity_scale on servo
Try `velocity_scale=15` or `10`. Halves/thirds the integration gain, reducing
sensitivity to bias. Trade-off: slower tracking response.

## Experiment results

### Experiment C — Rate limiter impact (2026-04-04)

**Method**: Replayed 200 val episodes (14,500 frames) through ONNX inference.
Compared raw model predictions (clipped to ±0.3) against rate-limited predictions
(max_accel=1.5, max_delta=0.15/frame). Measured MSE, transition lag, and
overshoot (opposite-sign frames in 5-frame window after GT zero-crossings).

**Script**: `train/experiment_c_ratelimiter.py`
**Full results**: `runs/v0.1.0/experiment_c_results.json`

| Metric                           | Raw (no limiter) | Rate-limited |
|----------------------------------|------------------|--------------|
| MSE vs GT                        | 0.00411          | 0.00536      |
| Pan transition lag (mean frames) | 1.67             | 2.68         |
| Tilt transition lag (mean frames) | 2.09            | 2.49         |
| Overshoot frames (of 3946)       | 437 (11.1%)      | 567 (14.4%)  |

**MSE increase from rate limiter: +30.5%**

**Interpretation**: The rate limiter degrades MSE by 30% and adds ~1 frame of
pan lag at transitions (+60%). Overshoot increases from 11.1% → 14.4% (+30%).
This **partially validates H1** — the limiter does contribute to overshoot and
lag. However, the raw model still has 11.1% overshoot even without the limiter,
which means the model itself has trouble at transitions (likely H2 — can't
output clean zeros near the deadband). The rate limiter makes an existing
problem worse but is not the sole cause.

**Recommendation**: Remove or substantially relax the rate limiter
(e.g. `max_accel=10.0`). The 30% MSE improvement alone justifies this. The
remaining 11% baseline overshoot should be investigated via Experiment A
(zero-region analysis) to understand the model's near-zero prediction behavior.

