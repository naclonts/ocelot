# Training

Install training dependencies:

```bash
pip install -r requirements-train.txt
```

Pull dataset using DVC (>75 GB data) with `dvc pull`.

## Sweep

Hyperparameter sweep over `lr` × `n_fusion_layers`:

```bash
SWEEP=sweep-v0.1   # increment to avoid overwriting previous sweep checkpoints
for lr in 1e-4 3e-4 1e-3; do
  for layers in 1 2 4; do
    python3 train/train.py \
        --dataset_dir sim/dataset/ \
        --output_dir  runs/$SWEEP/lr${lr}_l${layers}/ \
        --epochs 3 \
        --lr $lr \
        --n_fusion_layers $layers \
        --batch_size 64 \
        --max_episodes 1500 \
        --amp \
        --experiment ocelot-sweep
  done
done
```

Each combo writes a separate checkpoint under `runs/$SWEEP/` and a separate MLflow run under the `ocelot-sweep` experiment. View results sorted by `val_loss`:

```bash
mlflow ui    # http://localhost:5000 → experiment "ocelot-sweep"
```

## Full train

Full training run with AMP (use best `lr`/`layers`/`bs` from sweep):

```bash
python3 train/train.py \
    --dataset_dir sim/dataset/ \
    --output_dir  runs/v0.1.0/ \
    --epochs 20 \
    --batch_size 64 \
    --num_workers 12 \
    --amp \
    --confidence_weight 1.0 \
    --experiment ocelot-v0.1.0
```

To enable training-only domain randomization, add `--domain_randomization` and
tune the per-transform probabilities and strengths in `train/train.py`.

## Track-only train

Train on single-face tracking episodes only (`label_key=track`), filtering out
multi-face/attribute commands. Useful when the deployment only needs face tracking:

```bash
python3 train/train.py \
    --dataset_dir sim/dataset/ \
    --output_dir  runs/v0.2-track-only/ \
    --epochs 20 \
    --batch_size 64 \
    --num_workers 12 \
    --amp \
    --label_keys track \
    --experiment ocelot-v0.2-track-only
```

Inspect metrics:

```bash
mlflow ui    # open http://localhost:5000
```

`val_mse_<label_key>` columns show per-label breakdown (e.g. `basic_track`, `multi_left`).
A good model reaches RMSE < 0.015 rad/s per axis (< 10% of the typical oracle signal).

## Data collection

`collect_parallel.sh` generates episodes, runs them, and saves output. Output always goes to
`/ws/src/ocelot/sim/dataset` (bind-mounted to `sim/dataset/` on the host).

```bash
bash sim/data_gen/collect_parallel.sh --shards 7 --episodes 700
```

The script auto-detects the next unused shard index from the output directory, so re-running
never overwrites existing data. Override with `--start-shard N` if needed.

After collection, verify a shard:

```bash
docker exec -e ROS_DOMAIN_ID=1 ocelot-sim-0 \
  python3 /ws/src/ocelot/sim/data_gen/check_dataset.py --dataset /ws/src/ocelot/sim/dataset/shard_0
```

Then merge all shards into one dataset **on the host** (not inside a container — merge only needs h5py from `.venv`):

```bash
source .venv/bin/activate
python3 sim/data_gen/merge_shards.py \
    --parent sim/dataset \
    --output sim/dataset/merged
```

`collect_parallel.sh` runs this automatically at the end of a full run. If containers were killed early, run it manually. The merger auto-discovers all `shard_N/` directories, deduplicates episode IDs across shards, regenerates train/val/test splits, and writes `sim/dataset/merged/`.

For zero-velocity supervision, `collect_data.py` can also sample `no_face` and `centered` episodes:

```bash
python3 sim/data_gen/collect_data.py \
    --n_episodes 1000 \
    --output sim/dataset \
    --no_face_rate 0.10 \
    --centered_rate 0.05
```

## Evaluate a checkpoint

```bash
source .venv/bin/activate

# Text report (RMSE, Pearson r, sign agreement, per-label breakdown):
python3 train/eval.py \
    --checkpoint runs/v0.0-smoke/best.pt \
    --dataset_dir sim/dataset/

# With scatter plot + 4 episode time-series overlays:
python3 train/eval.py \
    --checkpoint runs/v0.0-smoke/best.pt \
    --dataset_dir sim/dataset/ \
    --plot --episodes 4
# → runs/v0.0-smoke/scatter.png, runs/v0.0-smoke/episodes.png
```

## VLA sim validation

Run the trained model inside Gazebo in closed-loop: the model sees each live camera frame and its output drives the pan-tilt joints. The face billboard oscillates automatically so there is always something to track.

**Step 1 — Export to ONNX** (host, one-time per checkpoint):

```bash
source .venv/bin/activate
python3 train/export_onnx.py \
    --checkpoint runs/v0.0-smoke/best.pt \
    --output     runs/v0.0-smoke/best.onnx \
    --verify
# → best.onnx + best_tokens.json alongside the checkpoint
```

**Step 2 — Rebuild sim image** (needed once after the Dockerfile changed to `onnxruntime-gpu`):

```bash
make sim-build
```

**Step 3 — Launch sim with VLA node (CPU)**:

```bash
docker compose -f deploy/docker/docker-compose.sim.yml run --rm sim bash -c "
  source /opt/ros/jazzy/setup.bash && cd /ws &&
  colcon build --symlink-install --packages-select ocelot --event-handlers console_direct- &&
  source /ws/install/setup.bash &&
  ros2 launch ocelot sim_launch.py use_vla:=true headless:=true
"
```

**Step 3 (GPU) — Launch sim with VLA node on NVIDIA GPU** (requires NVIDIA container runtime):

```bash
# Run VLA in sim
make sim-vla VLA_ONNX=runs/sweep-v0.0.2-1500-ep/lr1e-4_l2/best.onnx

# Evaluate against N reproducible scenarios (optional: override seed and count)
make sim-vla-eval VLA_ONNX=runs/sweep-v0.0.2-1500-ep/lr1e-4_l2/best.onnx SCENARIO_SEED=0 N_SCENARIOS=5
```

The `vla_node` logs which ONNX provider it is using on startup:
```
ONNX session ready (provider: CUDAExecutionProvider)   # GPU
ONNX session ready (provider: CPUExecutionProvider)    # CPU fallback
```

The default checkpoint path inside the container is `/ws/src/ocelot/runs/sweep-v0.0.2-1500-ep/lr1e-4_l2/best.onnx`
(the `runs/` directory is bind-mounted from the host). Override checkpoint or command:

```bash
ros2 launch ocelot sim_launch.py use_vla:=true headless:=true \
    vla_checkpoint:=/ws/src/ocelot/runs/v0.1/best.onnx \
    vla_command:="track the face"
```

**Monitor** from a second shell in the same container:

```bash
# Joint positions should change as the face oscillates
ros2 topic echo /joint_states --field position

# VLA velocity commands
ros2 topic echo /cmd_vel
```

The `vla_node` logs `pan=+0.xxx  tilt=+0.xxx rad/s` per frame. If the joints track
the face motion, behavioral cloning is working. If output is near-zero or static,
the model needs more training data or epochs.

## VLA live evaluation (training-distribution scenarios)

`sim-vla-eval` tests the VLA against N reproducible scenarios drawn from the same
distribution used for data collection — varied face textures, backgrounds, lighting,
motion patterns, and distractors. It measures FK angular error while the model drives
the robot and prints a pass/fail table.

```bash
# 5 scenarios from seed 0 (default)
make sim-vla-eval VLA_ONNX=runs/v0.1/best.onnx

# More scenarios, different seed range
make sim-vla-eval VLA_ONNX=runs/v0.1/best.onnx SCENARIO_SEED=50 N_SCENARIOS=10
```

The script waits up to 90 s for Gazebo and the VLA node to publish before starting —
no manual sleep needed. Each scenario runs a 4 s warmup (VLA convergence) then a
10 s measurement window. Output:

```
Scenario 1/5  seed=0  motion=sinusoidal  label=basic_track
  mean=3.2°  max=8.7°  n=100  [PASS]
...
--- Summary ---
  #    seed  motion            label           mean°   max°  pass
  1       0  sinusoidal        basic_track       3.2     8.7  Y
  ...
Overall: mean=4.1°  pass_rate=80%  (threshold=10.0°)
```

Pass threshold: mean FK angular error < 10°. A well-trained model should achieve
< 5° mean and > 80% pass rate.
