# Phase 4 — Hardware Deployment and Continuous Improvement

**Goal**: Deploy the trained VLA model to the Raspberry Pi 5, validate real-world tracking
performance, build a sim-to-real feedback loop for iterative model improvement, and establish
the MLOps infrastructure for autonomous retraining.

**When to run**: Phase 3 is complete when `train/eval_onnx.py` reports PASS on the test split,
CI eval gate is live, the ONNX model is DVC-tracked, and `vla_node.py` runs in sim without
crashing.

**Where to run**: Deployment target is the **Raspberry Pi 5** (aarch64, 8 GB RAM, Pi Camera V2,
PCA9685 servo driver). Model optimization and retraining run on the **dev machine** (Debian
Bookworm x86_64, RTX 2070 8 GB). The sim container handles online evaluation.

As you complete work, update this doc. These are guides, not rigid requirements. Use judgement
at each step and follow best practices.

---

## Step 1 — ONNX Runtime on Pi 5

Get the ONNX model running on Pi hardware with acceptable latency. The VLA node
(`ocelot/vla_node.py`) already uses `onnxruntime` — the work here is making it fast
enough on aarch64.

### 1a — Install onnxruntime on aarch64

The robot container (`Dockerfile.robot`) currently has no ML dependencies. Add onnxruntime:

```dockerfile
# In Dockerfile.robot — after the existing pip install block
RUN pip3 install --break-system-packages onnxruntime
```

If the default pip wheel is too slow, build with XNNPACK delegate for ARM NEON acceleration:

```bash
# Alternative: build from source with XNNPACK (only if pip wheel is >100ms/frame)
pip3 install --break-system-packages onnxruntime --prefer-binary
```

**Validation**: inside the robot container, run a quick inference benchmark:

```bash
python3 -c "
import onnxruntime as ort, numpy as np, time
sess = ort.InferenceSession('/ws/src/ocelot/models/vla.onnx', providers=['CPUExecutionProvider'])
frame = np.zeros((1,3,224,224), dtype=np.float32)
ids   = np.zeros((1,77), dtype=np.int64)
mask  = np.ones((1,77), dtype=np.int64)
# Warmup
for _ in range(5): sess.run(None, {'frames': frame, 'input_ids': ids, 'attention_mask': mask})
# Benchmark
times = []
for _ in range(50):
    t0 = time.perf_counter()
    sess.run(None, {'frames': frame, 'input_ids': ids, 'attention_mask': mask})
    times.append(time.perf_counter() - t0)
print(f'mean={np.mean(times)*1000:.1f}ms  p95={np.percentile(times,95)*1000:.1f}ms')
"
```

**Success gate**: mean inference latency < 200 ms (5 Hz minimum control rate). If > 200 ms,
proceed to Step 2 (model optimization) before continuing.

### 1b — Model file delivery to Pi

The ONNX model and token cache must be on the Pi filesystem. Options:

1. **DVC pull on Pi** (preferred for reproducibility):
   ```bash
   # On Pi, inside container or on host
   pip3 install dvc[s3]
   dvc pull models/vla.onnx
   ```

2. **SCP from dev machine** (simpler for initial testing):
   ```bash
   scp models/vla.onnx models/vla_tokens.json pi@<pi-ip>:/home/nathan/projects/ocelot/models/
   ```

3. **Bake into Docker image** (for production):
   ```dockerfile
   COPY models/vla.onnx models/vla_tokens.json /ws/src/ocelot/models/
   ```

Start with option 2 for rapid iteration. Move to option 1 or 3 once the model is validated.

**Success gate**: `ls /ws/src/ocelot/models/vla.onnx` succeeds inside the robot container.

---

## Step 2 — Model Optimization for Edge

If Step 1b benchmark shows > 100 ms mean latency, optimize the model. Apply these techniques
in order of effort (stop when latency is acceptable).

### 2a — ONNX Graph Optimization

`onnxruntime` can fuse operators at load time. Enable this explicitly:

```python
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.optimized_model_filepath = "models/vla_optimized.onnx"
sess = ort.InferenceSession("models/vla.onnx", opts, providers=["CPUExecutionProvider"])
# Save the optimized graph for deployment
```

Run the benchmark again with the optimized model.

### 2b — INT8 Quantization

Post-training quantization of the full ONNX graph. The frozen encoders (DINOv2-small, CLIP text)
are the bottleneck — quantizing them typically yields 2-3x speedup on ARM with minimal accuracy
loss.

```python
# train/quantize.py
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="models/vla.onnx",
    model_output="models/vla_int8.onnx",
    weight_type=QuantType.QInt8,
)
```

**Validation**: run `train/eval_onnx.py` on the quantized model — MSE must still pass the
gate (overall < 0.05, no per-label > 0.20). If accuracy degrades beyond the gate:

- Try quantizing only the encoders (exclude fusion + action head)
- Try QUInt8 instead of QInt8
- Fall back to FP16 (half the memory, modest speedup on ARM)

### 2c — Input Resolution Reduction (last resort)

If quantization is still too slow, reduce input resolution from 224x224 to 160x160 or 128x128.
This requires retraining (or fine-tuning) since DINOv2 positional embeddings change shape.
Only do this if all other optimizations fail to hit < 200 ms.

### 2d — Encoder Output Caching for Text

The CLIP text encoder processes the same command string every frame. Its output can be cached
and the text encoder removed from the inference graph entirely:

```python
# Split the ONNX model: run text encoder once at startup, cache the output,
# then run only the vision + fusion + head per frame.
# This removes ~63M params from the per-frame inference path.
```

Implementation: export two ONNX models from `export_onnx.py`:
1. `vla_text_encoder.onnx` — input_ids, attention_mask → text_features (B, 77, 384)
2. `vla_vision_head.onnx` — frames, text_features → actions (B, 2)

`vla_node.py` runs the text encoder once at startup and passes cached features each frame.

**Success gate**: mean inference < 150 ms on Pi 5 (targeting ~7-10 Hz control rate).

---

## Step 3 — Robot Launch Integration

Wire the VLA node into the robot's launch file and Docker compose stack so it can replace
the classical tracker on real hardware.

### 3a — Launch file update

Add `use_vla` arg to `tracker_launch.py`, mutually exclusive with the classical tracker:

```python
# launch/tracker_launch.py additions
DeclareLaunchArgument('use_vla', default_value='false'),
DeclareLaunchArgument('vla_checkpoint', default_value='/ws/src/ocelot/models/vla.onnx'),
DeclareLaunchArgument('vla_command', default_value='track the face'),

# VLA node (replaces tracker_node when use_vla:=true)
Node(
    package='ocelot',
    executable='vla_node',
    name='vla_node',
    parameters=[{
        'checkpoint': LaunchConfiguration('vla_checkpoint'),
        'command':    LaunchConfiguration('vla_command'),
        'enabled':    True,
    }],
    condition=IfCondition(LaunchConfiguration('use_vla')),
),
```

The classical `tracker_node` should only launch when `use_vla` is false (add an
`UnlessCondition`). Both nodes publish to `/cmd_vel` — they must never run simultaneously.

### 3b — Docker compose update

Add environment variables to `docker-compose.yml` for VLA mode:

```yaml
environment:
  - USE_VLA=${USE_VLA:-false}
  - VLA_COMMAND=${VLA_COMMAND:-track the face}
command: >
  bash -c "source /opt/ros/jazzy/setup.bash &&
  source /ws/install/setup.bash &&
  exec ros2 launch ocelot tracker_launch.py
  use_vla:=${USE_VLA:-false}
  vla_checkpoint:=/ws/src/ocelot/models/vla.onnx
  vla_command:='${VLA_COMMAND:-track the face}'
  record:=${RECORD:-false}
  visualize:=${VISUALIZE:-false}"
```

Launch with:
```bash
# Classical tracker (default, unchanged)
docker compose -f deploy/docker/docker-compose.yml up

# VLA model
USE_VLA=true docker compose -f deploy/docker/docker-compose.yml up

# VLA with specific command
USE_VLA=true VLA_COMMAND="follow the person on the left" \
  docker compose -f deploy/docker/docker-compose.yml up
```

### 3c — Graceful fallback

If the ONNX model fails to load (missing file, corrupt, wrong format), `vla_node.py` should
log a FATAL error and exit. The launch file should detect this and optionally fall back to
`tracker_node`. Implement with a `launch.actions.OnProcessExit` handler:

```python
# If vla_node exits with error, start tracker_node as fallback
OnProcessExit(
    target_action=vla_node_action,
    on_exit=[
        LogInfo(msg="VLA node failed — falling back to classical tracker"),
        tracker_node_action,
    ],
)
```

This ensures the robot is never left in a state where no controller is running.

**Success gate**: `USE_VLA=true docker compose -f deploy/docker/docker-compose.yml up` starts
the VLA node, it loads the ONNX model, and begins publishing `/cmd_vel` messages when a
camera feed is available.

---

## Step 4 — Real-World Validation

Test the VLA model on real hardware with a real face. This is the first sim-to-real transfer
test.

### 4a — Qualitative smoke test

Run the VLA node on the Pi with a person in front of the camera:

```bash
# On Pi
USE_VLA=true VISUALIZE=true docker compose -f deploy/docker/docker-compose.yml up
```

Open `http://<pi-ip>:8080` in a browser to see the camera feed via `web_video_server`.
Observe:

- Does the camera track the face?
- Does it respond to face movement (left/right/up/down)?
- Is there visible jitter or oscillation?
- Does it lose tracking and recover?

Record observations. This is qualitative — no metrics yet.

### 4b — Quantitative tracking error measurement

Extend `visualizer_node.py` (or create a new `eval_hardware_node.py`) to measure real-world
tracking quality. Use the Haar cascade as a reference detector (same as Phase 1):

```python
class HardwareEvalNode(Node):
    """Measures tracking error on real hardware.

    Subscribes to /camera/image_raw, runs Haar cascade face detection,
    computes pixel error from image center to face center, logs statistics.
    """
    def __init__(self):
        # ...
        self._errors = []

    def _image_cb(self, msg):
        # Detect face with Haar cascade
        # Compute pixel distance from image center to face center
        # Append to self._errors
        # Every 100 frames: log mean, p95, face-lost rate

    def _report(self):
        errors = np.array(self._errors)
        self.get_logger().info(
            f"n={len(errors)} mean_err={np.mean(errors):.1f}px "
            f"p95={np.percentile(errors,95):.1f}px "
            f"lost={np.mean(errors == -1)*100:.1f}%"
        )
```

### 4c — A/B comparison: classical vs. VLA

Run the same test scenario (person sitting 1-2m away, slowly moving head) with both
controllers. Compare:

| Metric | Classical Tracker | VLA Model |
|---|---|---|
| Mean tracking error (px) | | |
| p95 tracking error (px) | | |
| Face-lost rate (%) | | |
| Control smoothness (vel std) | | |
| Response to "follow slowly" | N/A | |
| Response to "track person on left" | N/A | |

The VLA doesn't need to beat the classical tracker on single-face tracking — that's its
simplest case. The value is language-conditional behavior and generalization to scenarios
the classical tracker can't handle (multi-face, attribute-based selection).

### 4d — Sim-to-real gap analysis

Document any systematic failures:

- **Appearance gap**: real faces look different from generated face textures. Symptoms:
  consistently higher error, failure to track at all.
- **Lighting gap**: real indoor lighting differs from sim. Symptoms: works in some rooms
  but not others.
- **Dynamics gap**: real servo response differs from sim velocity controller. Symptoms:
  oscillation, overshoot, or sluggish tracking.
- **Camera gap**: Pi Camera V2 has different FOV, noise, and color response than the sim
  camera. Symptoms: systematic offset.

Each gap type has a corresponding mitigation in Step 6.

**Success gate**: VLA model tracks a single face on real hardware with < 30 px mean error
(relaxed from sim's 15 px threshold to account for sim-to-real gap). If this fails, proceed
to Step 5 (fine-tuning) before continuing.

---

## Step 5 — Real-World Data Collection and Fine-Tuning

Bridge the sim-to-real gap by collecting a small amount of real-world data and fine-tuning
the model.

### 5a — Real-world data recording

Record rosbag data on the Pi while the **classical tracker** (not the VLA) is running.
The classical tracker's `/cmd_vel` output serves as the ground-truth label, analogous to
the oracle in sim.

```bash
# On Pi — record with classical tracker running
RECORD=true docker compose -f deploy/docker/docker-compose.yml up
# Move around naturally for 5-10 minutes per session
```

The rosbag records `/camera/image_raw` and `/cmd_vel` at the camera's frame rate.

### 5b — Rosbag to HDF5 conversion

Convert rosbag recordings to the same HDF5 format used by the sim dataset so the existing
`OcelotDataset` class can load them directly:

```python
# scripts/rosbag_to_hdf5.py
# For each synchronized (image, cmd_vel) pair:
#   - Resize image to 224x224
#   - Extract pan_vel = cmd_vel.angular.z, tilt_vel = cmd_vel.angular.y
#   - Write to HDF5 with same schema as sim episodes
#   - Use a fixed language command (e.g. "track the face") as cmd
```

The real-world episodes get a distinct `label_key` prefix (e.g. `real_track`) so they can
be tracked separately in MLflow metrics.

### 5c — Mixed dataset fine-tuning

Fine-tune the sim-trained model on a mix of sim + real data. The real data prevents
catastrophic forgetting of sim-learned behaviors while teaching real-world appearance.

```bash
# Merge datasets
# sim_dataset/: 50k sim episodes (Phase 2/3)
# real_dataset/: ~500 real episodes (5-10 min at 10 Hz ≈ 3000-6000 frames per session)

python3 train/train.py \
    --dataset_dir merged_dataset/ \
    --output_dir runs/v0.2-finetune/ \
    --epochs 5 \
    --lr 1e-4 \
    --batch_size 64 \
    --amp \
    --checkpoint runs/v0.1/best.pt \
    --experiment ocelot-finetune-v0.2
```

Add a `--checkpoint` flag to `train/train.py` to support loading a pretrained model for
fine-tuning (load state dict before training, do NOT reset the optimizer).

### 5d — Sim-real data ratio

Start with 10% real data mixed into the sim dataset. If fine-tuning hurts sim eval
performance, reduce to 5%. If real-world tracking is still poor, increase to 20%.

Monitor both metrics:
- `val_loss` on sim test split (must still pass eval gate)
- Real-world tracking error (from Step 4b hardware eval)

**Success gate**: Fine-tuned model achieves < 20 px mean tracking error on real hardware
while still passing the sim eval gate.

---

## Step 6 — Sim-to-Real Mitigations

Apply targeted fixes based on the gap analysis from Step 4d.

### 6a — Appearance gap: real face textures

If the model fails on real faces but works on sim faces, the face texture library needs
diversification:

1. **Photo-realistic face generation**: Use a higher-quality image generator or
   photo-sourced face textures (with appropriate licensing).
2. **Real face injection**: Capture a few real faces (yourself, friends) with the Pi camera
   and add them to the sim texture library. Run the sim data collection pipeline with these
   textures mixed in.
3. **Domain randomization**: Increase augmentation aggressiveness in `collect_data.py` —
   higher noise sigma, stronger brightness offsets, color jitter.

### 6b — Lighting gap

If tracking degrades under specific lighting conditions:

1. Widen the lighting randomization range in `ScenarioGenerator` (lower minimum ambient,
   higher maximum intensity).
2. Add color temperature variation (warm/cool white light) to the scenario generator.
3. Collect a few real-world rosbag sessions under the problematic lighting and fine-tune.

### 6c — Dynamics gap

If the robot oscillates or overshoots on real hardware:

1. **Tune `max_vel` and `max_accel`** in `vla_node.py` — the sim-trained model outputs
   velocities calibrated to the sim's velocity controller. Real servos may have different
   response curves.
2. **Add a low-pass filter** on the VLA output in `vla_node.py`:
   ```python
   # Exponential moving average
   self._pan_ema = alpha * pan_vel + (1 - alpha) * self._pan_ema
   ```
3. If the servo response is fundamentally different, add a `servo_calibration.yaml`
   config file that maps VLA output velocities to servo-specific PWM commands.

### 6d — Camera gap

If the Pi Camera V2's FOV or color response causes issues:

1. **FOV mismatch**: Adjust the sim camera's `<horizontal_fov>` in the URDF to match the
   Pi Camera V2's actual FOV (62.2 deg horizontal). Recollect sim data with the corrected
   FOV and retrain.
2. **Color calibration**: Apply a simple color transform to real frames before inference
   (or add color jitter to sim training augmentation).

---

## Step 7 — MLOps Pipeline

Automate the train-evaluate-deploy cycle so model improvements can be shipped continuously.

### 7a — Model registry

Use MLflow's model registry to manage model versions:

```bash
# After a successful training run
mlflow models create -n ocelot-vla

# Register a new version
mlflow models create-version \
    -n ocelot-vla \
    -s runs://<run_id>/best.pt \
    --description "v0.2: sim+real fine-tune, <20px real-world error"
```

Lifecycle stages: `Staging` → `Production`. Only `Production` models get deployed to the Pi.

### 7b — Automated retraining trigger

Set up a script that monitors for new data and triggers retraining:

```bash
# scripts/retrain.sh
#!/bin/bash
set -e

# Pull latest dataset
dvc pull

# Train
python3 train/train.py \
    --dataset_dir sim/dataset/ \
    --output_dir runs/auto-$(date +%Y%m%d)/ \
    --epochs 10 \
    --lr 1e-4 \
    --batch_size 64 \
    --amp \
    --checkpoint runs/latest/best.pt \
    --experiment ocelot-auto

# Export
python3 train/export_onnx.py \
    --checkpoint runs/auto-$(date +%Y%m%d)/best.pt \
    --output models/vla.onnx \
    --verify

# Evaluate
python3 train/eval_onnx.py \
    --model_path models/vla.onnx \
    --dataset_dir sim/dataset/ \
    --output runs/auto-$(date +%Y%m%d)/eval.json

# Gate check
python3 -c "
import json, sys
r = json.load(open('runs/auto-$(date +%Y%m%d)/eval.json'))
if not r['pass']:
    print('FAIL — not promoting model')
    sys.exit(1)
print('PASS — promoting model')
"

# Track and push
dvc add models/vla.onnx
git add models/vla.onnx.dvc
git commit -m "auto: update vla.onnx $(date +%Y%m%d)"
dvc push
git push
```

### 7c — Deploy-to-Pi script

Automate model deployment from the dev machine to the Pi:

```bash
# scripts/deploy.sh
#!/bin/bash
set -e
PI_HOST=${PI_HOST:-pi@raspberrypi.local}
MODEL_PATH=models/vla.onnx
TOKEN_PATH=models/vla_tokens.json

echo "Deploying model to $PI_HOST..."
scp $MODEL_PATH $TOKEN_PATH $PI_HOST:/home/nathan/projects/ocelot/models/

echo "Restarting robot stack with VLA..."
ssh $PI_HOST "cd /home/nathan/projects/ocelot && \
    docker compose -f deploy/docker/docker-compose.yml down && \
    USE_VLA=true docker compose -f deploy/docker/docker-compose.yml up -d"

echo "Deployed. Check: ssh $PI_HOST 'docker logs -f ocelot-ocelot-1'"
```

### 7d — Health monitoring

Add a lightweight monitoring node that publishes diagnostic info:

```python
class HealthNode(Node):
    """Publishes /diagnostics with VLA inference stats."""
    # - Inference latency per frame (ms)
    # - Control rate (Hz)
    # - Model file hash (detect stale deployments)
    # - Haar cascade tracking error (when face visible)
    # - Face-lost rate over rolling 100-frame window
```

This integrates with `web_video_server` — add an overlay showing inference FPS and
tracking status.

---

## Step 8 — Language Command Interface

The VLA model accepts language commands but Phase 1-3 only used fixed commands. Add a
user-facing interface for runtime command switching.

### 8a — ROS2 parameter-based command switching

`vla_node.py` already has a `command` parameter. Add dynamic reconfiguration:

```bash
# Change command at runtime
ros2 param set /vla_node command "follow the person on the left"
```

This requires modifying `vla_node.py` to:
1. Listen for parameter changes via `add_on_set_parameters_callback`
2. Look up the new command in the token cache
3. Update `_input_ids` and `_attention_mask`

### 8b — Web UI for command selection

Extend `web_video_server` or add a simple Flask/FastAPI app on the Pi that shows:
- Live camera feed
- Dropdown of available commands (from token cache keys)
- Current tracking status (FPS, error, face detected)
- Manual override buttons (enable/disable, classical fallback)

```bash
# scripts/web_ui.py — minimal Flask app on port 8081
# Communicates with vla_node via ros2 param set / ros2 topic echo
```

### 8c — Voice command integration (stretch goal)

Use a local speech-to-text model (e.g. Whisper tiny) to accept voice commands:

```
User: "Hey Ocelot, follow the person in the hat"
→ Whisper STT → text → closest match in token cache → ros2 param set
```

This is a stretch goal — only pursue if Steps 1-7 are complete and working well.

**Success gate**: Commands can be changed at runtime and the robot responds appropriately
(e.g. switching from "track the face" to "follow the person on the left" changes which
face the robot tracks in a multi-face scenario).

---

## Step 9 — Online Sim Evaluation (Closed-Loop)

Phase 3's eval is offline (replay HDF5 frames, compare VLA output to oracle labels). Add
closed-loop evaluation where the VLA actually controls the sim robot.

### 9a — Online eval script

```python
# sim/eval_vla_live.py (partially exists — extend it)
# For each test scenario:
#   1. Spawn scenario in Gazebo
#   2. Run VLA node (not oracle) controlling the sim robot
#   3. Subscribe to /camera/image_raw and /model/face_0/pose
#   4. Measure pixel tracking error: project face_0 world pose into camera frame,
#      compute distance from image center
#   5. Record per-frame errors over the episode duration
#   6. Aggregate and report
```

Online eval is more realistic than offline because:
- The VLA's actions affect the next observation (closed-loop)
- Compounding errors are captured (offline eval doesn't compound)
- Camera rendering pipeline is exercised end-to-end

### 9b — Online eval in CI (stretch goal)

Run headless Gazebo in CI with the VLA node for a small number of scenarios. This requires
a CI runner with enough resources for Gazebo + onnxruntime (a self-hosted runner, or a
large GitHub Actions runner).

**Success gate**: Online eval reports < 20 px mean tracking error on 10 test scenarios.

---

## Step 10 — Production Hardening

Final polish for reliable long-running operation.

### 10a — Watchdog and auto-restart

Add a systemd service (or Docker restart policy) that ensures the robot stack restarts
after crashes:

```yaml
# docker-compose.yml
services:
  ocelot:
    restart: unless-stopped
```

Add a watchdog timer in `vla_node.py` — if no camera frame arrives for 5 seconds, log
a warning and publish zero velocity (don't let the robot drift with stale commands).

### 10b — Thermal management

ONNX inference on Pi 5 will generate heat. Monitor CPU temperature and throttle inference
rate if > 80 C:

```python
def _get_cpu_temp() -> float:
    return float(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000.0

# In _image_cb: skip every other frame if temp > 80C
```

### 10c — Logging and telemetry

Write structured logs (JSON) to `/ws/logs/` with:
- Inference latency per frame
- Tracking error (when Haar cascade detects a face)
- Temperature
- Active command

These logs support post-hoc analysis of real-world performance and inform the next
training iteration.

### 10d — Model rollback

If a new model performs worse on real hardware, the deploy script should support rollback:

```bash
# scripts/deploy.sh --rollback
# Reverts to the previous vla.onnx from DVC cache
dvc checkout models/vla.onnx --rev HEAD~1
scp models/vla.onnx $PI_HOST:/home/nathan/projects/ocelot/models/
```

---

## Directory Layout After Phase 4

```
ocelot/
├── ocelot/
│   ├── vla_node.py             # MODIFIED — dynamic command switching, watchdog
│   ├── tracker_node.py         # unchanged (fallback)
│   ├── health_node.py          # NEW — diagnostics publisher
│   └── ... (other Phase 1/2 nodes unchanged)
├── train/
│   ├── quantize.py             # NEW — INT8/FP16 quantization
│   └── ... (Phase 3 files, train.py gains --checkpoint flag)
├── scripts/
│   ├── rosbag_to_hdf5.py       # NEW — real-world data conversion
│   ├── retrain.sh              # NEW — automated retrain pipeline
│   ├── deploy.sh               # NEW — model deployment to Pi
│   └── web_ui.py               # NEW — command selection web interface
├── sim/
│   ├── eval_vla_live.py        # MODIFIED — extended online eval
│   └── ... (Phase 2 files unchanged)
├── launch/
│   ├── tracker_launch.py       # MODIFIED — use_vla arg, fallback logic
│   └── sim_launch.py           # unchanged
├── deploy/
│   └── docker/
│       ├── Dockerfile.robot    # MODIFIED — add onnxruntime
│       └── docker-compose.yml  # MODIFIED — USE_VLA env var
├── models/
│   ├── vla.onnx                # DVC-tracked
│   ├── vla_int8.onnx           # DVC-tracked (if quantized)
│   └── vla_tokens.json         # git-tracked
├── config/
│   └── tracker_params.yaml     # unchanged
├── tests/
│   ├── train/                  # Phase 3 tests + new quantization tests
│   └── sim/                    # Phase 2 tests unchanged
└── .github/
    └── workflows/
        └── ci.yml              # MODIFIED — add quantized model eval
```

---

## Phase 4 Validation Checklist

- [ ] ONNX inference runs on Pi 5 at >= 5 Hz (< 200 ms/frame)
- [ ] VLA node launches via `tracker_launch.py` with `use_vla:=true`
- [ ] Classical tracker fallback works when VLA node fails
- [ ] VLA tracks a single real face with < 30 px mean error
- [ ] Language commands change robot behavior (tested with >= 2 distinct commands)
- [ ] A/B comparison documented: classical tracker vs. VLA
- [ ] Rosbag → HDF5 conversion pipeline works
- [ ] Fine-tuned model passes both sim eval gate AND real-world < 20 px gate
- [ ] Deploy script pushes model to Pi and restarts the stack
- [ ] Robot stack auto-restarts after crash (Docker restart policy)
- [ ] Health monitoring reports inference latency and tracking status

---

## Key Risks and Mitigations

| Risk | Mitigation |
|---|---|
| ONNX inference too slow on Pi 5 (> 200 ms) | INT8 quantization, text encoder caching, input resolution reduction. If all fail, consider a Pi 5 AI HAT (NPU accelerator) or offload inference to a nearby machine over WiFi. |
| Sim-to-real appearance gap (model fails on real faces) | Fine-tune on mixed sim+real data. Diversify face texture library with photo-realistic sources. Increase domain randomization. |
| Servo dynamics mismatch causes oscillation | Tune max_vel/max_accel in vla_node.py. Add EMA smoothing. Profile real servo step response and match sim controller to it. |
| INT8 quantization degrades accuracy beyond threshold | Selective quantization (encoders only). FP16 as intermediate step. Accuracy-aware quantization with calibration dataset. |
| Real-world data labeling is noisy (classical tracker isn't perfect) | Classical tracker is good enough for single-face; only use it as label source for `track the face` scenarios. For multi-face scenarios, use sim data exclusively. |
| Pi 5 thermal throttling under sustained inference | Skip frames when CPU > 80C. Add passive heatsink. Reduce inference rate to 5 Hz (sufficient for face tracking). |
| Model update breaks real-world behavior (regression) | Always run sim eval gate before deployment. Keep rollback script ready. Run hardware eval (Step 4b) after every deployment. |

---

## Phase 5 Handoff Criteria

Phase 5 (Polish and Portfolio) can begin when:

1. VLA model runs on Pi 5 at >= 5 Hz with < 20 px mean tracking error on real faces
2. At least 2 distinct language commands produce visibly different behavior
3. Deploy script works end-to-end (dev machine → Pi)
4. CI eval gate catches regressions (demonstrated with a deliberately bad model)
5. The robot reliably tracks a face for 5+ minutes without manual intervention
