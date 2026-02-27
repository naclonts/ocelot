# Phase 3 — VLA Model and Training Infrastructure

**Goal**: Implement the vision-language-action model, train it on the synthetic dataset from Phase 2,
build an automated sim evaluation framework, and wire it into CI — so no bad model can reach hardware.

**When to run**: Phase 2 is complete when `check_dataset.py` passes on ≥ 50k episodes, all label
types appear at ≥ 5%, and `dvc pull` reproduces the dataset from a clean checkout.

**Where to run**: Training runs on the **dev machine** (Debian Bookworm x86_64, RTX 2070 8GB,
CUDA 12.2). The sim container handles headless evaluation. The Pi remains the deployment target
for Phase 4.

As you complete work, update this doc. Also these are all just guides, not requirements. Use judgement at each step and follow best practices.

---

## Phase 2 Completion Prerequisites (do these before Step 1)

Steps 8–10 of Phase 2 are prerequisites for Phase 3. Check them off before starting here.

### P2-Step 8 — Scale data collection to ≥ 50k episodes

```bash
# Inside sim container — start the sim stack
ros2 launch ocelot sim_launch.py world:=scenario_world use_oracle:=true headless:=true &
sleep 25

# Collect 50k episodes (single process; expect ~8–12 hours on dev machine)
python3 /ws/src/ocelot/sim/collect_data.py \
    --n_episodes 50000 \
    --output /ws/src/ocelot/dataset \
    --base_seed 0

# Quality gate
python3 /ws/src/ocelot/sim/check_dataset.py --dataset /ws/src/ocelot/dataset
```

For faster collection, run N parallel collectors on separate `GZ_PARTITION` values and
merge afterward. Each shard needs its own Gazebo+ROS2 stack.

**Success gate**: `check_dataset.py` reports:
- ≥ 50k episodes
- Every label type ≥ 5%
- `pan_vel` std > 0.3
- No corrupt files

### P2-Step 9 — DVC tracking

```bash
# On host (activate .venv first)
source .venv/bin/activate
dvc add dataset/
git add dataset.dvc .gitignore
git commit -m "add phase2 synthetic dataset v0.1 (50k episodes)"
dvc push
```

**Success gate**: `git checkout HEAD && dvc pull` restores `dataset/` from scratch.

### P2-Step 10 — Final validation checklist

- [ ] Oracle achieves < 5 px mean tracking error (validated in Step 5 debugging)
- [ ] ≥ 50k episodes, all label types at ≥ 5%
- [ ] `dvc pull` works from a clean checkout
- [ ] `pytest tests/sim/ -v` passes all 18 tests
- [ ] README updated with dataset section

---

## Step 1 — PyTorch Environment and DataLoader ✅ DONE (2026-02-25)

**Status**: Complete. 13/13 tests pass. Dataset loads 400 real episodes (32k train / 4k val / 4k test frames).

**Note on dataset size**: 100 episodes/shard × 4 shards = 400 total episodes. P2-Step 8 (50k) was skipped
to start Phase 3 early with the existing 400-episode set. Scale up data collection before v0.1 full training.

Set up the training environment on the **host dev machine** (not in the sim container).
The host already has a `.venv` — extend it rather than creating a new one.

### 1a — Install training dependencies

```bash
source .venv/bin/activate
# RTX 2070 + CUDA 12.2 driver → use cu121 wheel (latest CUDA 12.x wheel that works)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers mlflow onnx onnxruntime pytest h5py numpy tqdm

# Verify GPU is visible
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True, NVIDIA GeForce RTX 2070
```

Add a `requirements-train.txt` with pinned versions for reproducibility.

### 1b — Dataset class

**File**: `train/dataset.py`

```python
class OcelotDataset(torch.utils.data.Dataset):
    """
    Loads (frame, language_cmd, pan_vel, tilt_vel) tuples from HDF5 episode files.
    One episode = N frames; each frame is one sample.
    """
    def __init__(self, split_file: Path, dataset_dir: Path, transform=None):
        # Read episode IDs from train.txt / val.txt / test.txt
        # Build index: (episode_path, frame_idx) for every frame in every episode
        ...

    def __len__(self): ...
    def __getitem__(self, idx) -> dict:
        # Returns {"frame": (3,224,224) float32 tensor,
        #          "cmd": str,
        #          "pan_vel": float32 scalar,
        #          "tilt_vel": float32 scalar}
        ...
```

Design notes:
- Index at construction time: iterate split episode files once, build a flat list of
  `(h5_path, frame_idx)` pairs. Store in `self._index`. `__getitem__` opens the HDF5
  file, reads one frame. This avoids loading all frames into RAM.
- Use `h5py` with `swmr=True` (single-writer multi-reader) if running multiple workers.
- Frame tensor: divide uint8 by 255.0, apply ImageNet normalization (DINOv2 expects it).
- Language cmd: returned as a raw string — the model's text encoder tokenizes it.

### 1c — Unit tests

**File**: `tests/train/test_dataset.py`

Use a tiny synthetic HDF5 fixture (3 episodes × 10 frames) instead of the real dataset.
Assert:
- `len(dataset)` = total frames across split
- `dataset[0]` returns correct shapes: frame (3,224,224), scalars for velocities
- Normalization is applied (mean ≈ 0 after ImageNet norm)
- Train/val/test splits are disjoint at episode level

```bash
pytest tests/train/test_dataset.py -v
```

**Success gate**: All tests pass, DataLoader runs 1 epoch over the fixture without error.

---

## Step 2 — VLA Model Architecture ✅ DONE (2026-02-25)

**Status**: Complete. 18/18 tests pass. All tests run on CPU in < 2s (no HuggingFace downloads).

**Key decisions**:
- Uses `CLIPTextModel` directly (not `CLIPModel`) — avoids loading unused CLIP vision encoder (~200 MB).
- `pretrained=False` uses lightweight `_VisualStub` / `_TextStub` with one sentinel param each so gradient-freezing tests exercise the frozen/trainable split without any downloads.
- Text encoder module is named `self.clip_text` (not `self.clip.text_model`) to avoid loading the full CLIPModel.

**File**: `train/model.py`

The architecture has four components. Two are frozen pretrained encoders. Two are trainable.

```
 ┌─────────────────────────────────────────────────────────┐
 │  Input: frame (3,224,224) + language command (string)   │
 └───────────────┬──────────────────────────┬──────────────┘
                 │                          │
        ┌────────▼────────┐       ┌─────────▼─────────┐
        │   DINOv2-small  │       │   CLIP text enc.  │
        │   (21M params)  │       │   (63M params)    │
        │   FROZEN        │       │   FROZEN          │
        └────────┬────────┘       └─────────┬─────────┘
                 │                          │
         (257, 384) tokens          (77, 512) → project →
                 │                  (77, 384) tokens
                 │                          │
        ┌────────▼──────────────────────────▼────────┐
        │   Cross-Attention Fusion (2 layers)        │
        │   Trainable — ~2M params                   │
        │   Visual tokens attend to language tokens  │
        └──────────────────────┬─────────────────────┘
                               │
                   CLS token → (384,)
                               │
        ┌──────────────────────▼─────────────────────┐
        │   MLP Action Head                          │
        │   384 → 256 → 64 → 2                       │
        │   Output: (pan_vel, tilt_vel) float32      │
        │   Trainable — ~130k params                 │
        └────────────────────────────────────────────┘
```

### 2a — Component details

**Visual encoder** (`facebook/dinov2-small`):
```python
from transformers import AutoModel
dino = AutoModel.from_pretrained("facebook/dinov2-small")
for p in dino.parameters():
    p.requires_grad = False
# Output: last_hidden_state (B, 257, 384) — [CLS] + 256 patch tokens
```

**Text encoder** (`openai/clip-vit-base-patch32`):
```python
from transformers import CLIPModel, CLIPTokenizer
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
for p in clip.text_model.parameters():
    p.requires_grad = False
# Usage: clip.get_text_features(input_ids=...) → (B, 512)
# Or use clip.text_model to get per-token states (B, 77, 512)
```

**Text projection**: a small trainable linear `(512, 384)` to match DINOv2's dimension.
This layer is trainable — it adapts CLIP's text features to cross-attend with DINOv2.

**Cross-attention fusion**:
```python
nn.MultiheadAttention(embed_dim=384, num_heads=6, batch_first=True)
# 2 layers, each: visual queries, language keys+values → fused visual tokens
```

**Action head**:
```python
nn.Sequential(
    nn.Linear(384, 256), nn.GELU(),
    nn.Linear(256, 64), nn.GELU(),
    nn.Linear(64, 2)   # (pan_vel, tilt_vel)
)
```

### 2b — Forward pass

```python
def forward(self, frames: Tensor, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
    # frames: (B, 3, 224, 224)
    # input_ids, attention_mask: (B, 77) — CLIP tokenizer output
    with torch.no_grad():
        vis = self.dino(pixel_values=frames).last_hidden_state        # (B, 257, 384)
        txt = self.clip.text_model(input_ids, attention_mask)
        txt_hidden = txt.last_hidden_state                            # (B, 77, 512)
    txt_proj = self.txt_proj(txt_hidden)                              # (B, 77, 384)
    fused = vis
    for attn_layer in self.fusion:
        fused, _ = attn_layer(fused, txt_proj, txt_proj)             # (B, 257, 384)
    cls = fused[:, 0, :]                                             # (B, 384)
    return self.action_head(cls)                                     # (B, 2)
```

### 2c — Unit tests

**File**: `tests/train/test_model.py`

```python
def test_forward_shapes():
    model = VLAModel()
    frames = torch.randn(2, 3, 224, 224)
    tokens = clip_tokenizer(["track the face", "follow slowly"],
                            return_tensors="pt", padding=True, truncation=True)
    out = model(frames, tokens["input_ids"], tokens["attention_mask"])
    assert out.shape == (2, 2)

def test_gradients_only_trainable():
    model = VLAModel()
    out = model(...)
    loss = out.sum()
    loss.backward()
    for name, p in model.named_parameters():
        if "dino" in name or "clip.text_model" in name:
            assert p.grad is None, f"Frozen param has gradient: {name}"
        else:
            assert p.grad is not None, f"Trainable param has no gradient: {name}"
```

**Success gate**: Forward pass runs, shapes correct, gradients only flow through trainable params.

---

## Step 3 — Training Loop and MLflow

**File**: `train/train.py`

### 3a — Loss function

Behavioral cloning with MSE on (pan_vel, tilt_vel):

```python
criterion = nn.MSELoss()
# target: (B, 2) from dataset ["pan_vel", "tilt_vel"]
# pred:   (B, 2) from model forward pass
loss = criterion(pred, target)
```

No fancy loss needed — the oracle outputs are continuous and roughly Gaussian.
Log per-label-type MSE separately in MLflow so you can see if "follow slowly" underperforms.

### 3b — Training script structure

```
train/train.py
  --dataset_dir DATASET_DIR   # path to dataset/ with episodes/ and *.txt splits
  --output_dir  OUTPUT_DIR    # where to save checkpoints and MLflow artifacts
  --epochs      N             # default 10
  --batch_size  B             # default 64 (fits in 8GB VRAM with fp32; try 128 with amp)
  --lr          LR            # default 3e-4
  --n_fusion_layers N         # default 2
  --n_heads     N             # default 6
  --amp                       # flag: use mixed precision (fp16) — halves VRAM, ~2× faster
  --experiment  NAME          # MLflow experiment name
```

The training loop should move everything to GPU:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# In the batch loop:
frames = frames.to(device)
targets = targets.to(device)
input_ids = tokens["input_ids"].to(device)
attention_mask = tokens["attention_mask"].to(device)
```

Use `torch.cuda.amp.autocast()` + `GradScaler` when `--amp` is set — with 8GB VRAM
this lets you run batch size 128 comfortably and cuts per-epoch time roughly in half.

Training loop outline:
```python
mlflow.set_experiment(args.experiment)
with mlflow.start_run():
    mlflow.log_params(vars(args))

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss, per_label_mse = evaluate(model, val_loader)

        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_mse_{k}": v for k, v in per_label_mse.items()}
        }, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best.pt")

    mlflow.log_artifact(output_dir / "best.pt")
```

### 3c — Tokenization

The DataLoader returns raw `cmd` strings. Tokenize in the training loop (or a collate_fn)
to keep the dataset class framework-agnostic:

```python
def collate_fn(batch):
    frames   = torch.stack([b["frame"] for b in batch])
    pan_vels = torch.tensor([b["pan_vel"] for b in batch])
    tilt_vels = torch.tensor([b["tilt_vel"] for b in batch])
    targets  = torch.stack([pan_vels, tilt_vels], dim=1)
    cmds     = [b["cmd"] for b in batch]
    tokens   = tokenizer(cmds, return_tensors="pt",
                         padding=True, truncation=True, max_length=77)
    return {"frames": frames, "tokens": tokens, "targets": targets, "cmds": cmds}
```

### 3d — Smoke test on 10k subset

Before scaling, train for 3 epochs on the 10k split (or a 10k random sample of the 50k
dataset). The loss should decrease monotonically. If loss is flat or diverges, the architecture
or data pipeline has a bug.

```bash
source .venv/bin/activate
python3 train/train.py \
    --dataset_dir dataset_10k/ \
    --output_dir runs/v0.0-smoke/ \
    --epochs 3 \
    --batch_size 32 \
    --experiment ocelot-smoke
mlflow ui  # inspect loss curves
```

**Success gate**: `val_loss` after epoch 3 < `val_loss` after epoch 1. Actions look non-random
(pan/tilt vels have sensible sign and magnitude vs. targets).

---

## Step 4 — Hyperparameter Sweep and v0.1 Model

### 4a — Grid search

Run a small grid over the three most impactful hyperparameters. Each run trains 5 epochs
on the 50k dataset. Use MLflow to compare.

| Hyperparameter     | Values to try |
|--------------------|---------------|
| Learning rate      | 1e-4, 3e-4, 1e-3 |
| N fusion layers    | 1, 2, 4 |
| Batch size         | 32, 64, 128 |

Total: 27 runs × 5 epochs. Run sequentially on a single GPU — launching jobs in parallel
with `&` will CUDA OOM immediately since all processes compete for the same 8 GB.

```bash
cd /home/nathan/projects/ocelot
source .venv/bin/activate

for lr in 1e-4 3e-4 1e-3; do
  for layers in 1 2 4; do
    for bs in 32 64 128; do
      python3 train/train.py \
          --dataset_dir sim/dataset/ \
          --output_dir runs/sweep/lr${lr}_l${layers}_bs${bs}/ \
          --epochs 5 \
          --lr $lr \
          --n_fusion_layers $layers \
          --batch_size $bs \
          --amp \
          --num_workers 8 \
          --experiment ocelot-sweep
    done
  done
done
```

View results when done:

```bash
source .venv/bin/activate
mlflow ui --host 127.0.0.1 --port 5000
# open http://localhost:5000 → experiment "ocelot-sweep", sort by val_loss ascending
```

### 4d — Periodic Perturbation Data Recollection (Distribution Shift Fix)

**Context**: Dataset analysis on 8k episodes showed ~3% of frames have large oracle corrections.
Mean |pan_vel| peaks at frames 2–4 (oracle reacting to face spawn) then collapses immediately —
the oracle is that fast. A start-of-episode perturbation yields the same 3 useful recovery frames
then 97 steady-state frames. The VLA's 13.3° mean / 22.4° max eval error is a direct consequence:
>95% of training frames show near-zero commands, so that becomes the model's default output.

**Fix**: Periodic mid-episode perturbation. Every `--perturb_interval` frames, face_0 is teleported
to a random angular offset from its motion-pattern position and held there for `PERTURB_DURATION=5`
frames before the motion pattern resumes. With `--perturb_interval 15` on 100-frame episodes:
~6 perturbations × 5 recovery frames ≈ **30% recovery frames**, up from 3%.

**Why ±0.5 rad**: Camera half-FOV is 30° ≈ 0.524 rad. Capping at ±0.5 rad (28.6°) keeps the face
just inside the frame at worst case. Uniform distribution (not Gaussian) ensures even coverage of
the full angular range.

**Implementation**: `sim/data_gen/collect_data.py` — `--perturb_interval` and `--perturb_range` args.
After each `runner.step()` call, if a perturbation is active, `bridge.set_pose` is called again on
face_0 with a world-space Y/Z offset derived from the sampled angular perturbation. The perturbation
RNG is seeded from `config.seed ^ 0xBEEF`, independent of augmentation and scenario streams.

```bash
# Collect 1500 perturbed episodes (inside sim container)
python3 /ws/src/ocelot/sim/data_gen/collect_data.py \
    --n_episodes 1500 \
    --output /ws/src/ocelot/sim/dataset_perturbed/ \
    --base_seed 10000 \
    --perturb_interval 15 \
    --perturb_range 0.5

# Quality check
python3 /ws/src/ocelot/sim/data_gen/check_dataset.py \
    --dataset /ws/src/ocelot/sim/dataset_perturbed/
```

**Success gate**: `pan_vel` p95 for the perturbed dataset should be significantly higher than the
clean baseline (p95 ≈ 0.17 rad/s). Expect p95 ≈ 0.5+ rad/s. Use the perturbed dataset (or a
merge with the existing clean dataset) for Step 4b full training.

---

### 4b — Full training (v0.1)

Train the best config from the sweep on the full available dataset for 20 epochs.
Cosine LR schedule and gradient clipping are already implemented in `train.py`.

```bash
# Replace lr, layers, bs with the best values from the sweep
python3 train/train.py \
    --dataset_dir sim/dataset/ \
    --output_dir runs/v0.1/ \
    --epochs 20 \
    --lr 3e-4 \
    --n_fusion_layers 2 \
    --batch_size 64 \
    --amp \
    --num_workers 4 \
    --experiment ocelot-v0.1
```

The trained model is `v0.1`. Tag in git and log to MLflow.

### 4c — ONNX export

```python
# train/export_onnx.py
model.eval()
dummy_frame = torch.zeros(1, 3, 224, 224)
dummy_ids   = torch.zeros(1, 77, dtype=torch.long)
dummy_mask  = torch.ones(1, 77, dtype=torch.long)

torch.onnx.export(
    model,
    (dummy_frame, dummy_ids, dummy_mask),
    "models/vla_v0.1.onnx",
    input_names=["frames", "input_ids", "attention_mask"],
    output_names=["actions"],
    dynamic_axes={"frames": {0: "batch"}},
    opset_version=17,
)
```

Validate ONNX output matches PyTorch to within 1e-4:
```python
import onnxruntime as ort
sess = ort.InferenceSession("models/vla_v0.1.onnx")
pt_out  = model(dummy_frame, dummy_ids, dummy_mask).detach().numpy()
ort_out = sess.run(None, {"frames": ..., "input_ids": ..., "attention_mask": ...})[0]
np.testing.assert_allclose(pt_out, ort_out, atol=1e-4)
```

**Success gate**: ONNX model loads, validates numerically, file size < 200 MB.

---

## Step 5 — ONNX Inference Node

**File**: `ocelot/vla_node.py`

This node replaces the oracle in simulation and eventually replaces the classical tracker
on real hardware (Phase 4). It subscribes to the same topics as `tracker_node` and
publishes to the same `/cmd_vel`.

### 5a — Node design

```python
class VLANode(Node):
    def __init__(self):
        super().__init__('vla_node')
        self.declare_parameter('model_path', '')      # path to .onnx
        self.declare_parameter('language_cmd', 'track the face')
        self.declare_parameter('enabled', False)

        self.session = ort.InferenceSession(model_path)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self._precompute_text_tokens()  # tokenize once at startup

        self.sub_image = self.create_subscription(
            Image, '/camera/image_raw', self._on_image, 1)
        self.pub_cmd   = self.create_publisher(Twist, '/cmd_vel', 1)
        self.bridge    = CvBridge()

    def _on_image(self, msg):
        if not self.get_parameter('enabled').value:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
        frame = cv2.resize(frame, (224, 224))
        frame_t = (frame.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        frame_t = frame_t.transpose(2, 0, 1)[np.newaxis]  # (1,3,224,224)

        actions = self.session.run(None, {
            "frames":           frame_t,
            "input_ids":        self._input_ids,
            "attention_mask":   self._attention_mask,
        })[0]  # (1, 2)

        cmd = Twist()
        cmd.angular.z = float(np.clip(actions[0, 0], -1.0, 1.0))
        cmd.angular.y = float(np.clip(actions[0, 1], -1.0, 1.0))
        self.pub_cmd.publish(cmd)
```

Key design choices:
- **Text tokens precomputed at startup**: language command doesn't change per frame.
  But we need to run them through the ONNX model which includes the text encoder.
  Alternative: cache the text encoder output separately, but the ONNX graph must be split.
  Simpler: keep the full graph end-to-end, precompute tokens (fast), run ONNX each frame.
- **`enabled` param**: off by default. Set true via `ros2 param set` or launch arg.

### 5b — Launch integration

Add `use_vla` arg to `sim_launch.py` (mutually exclusive with `use_oracle`):

```python
# launch/sim_launch.py
if use_vla:
    actions.append(Node(
        package='ocelot',
        executable='vla_node',
        parameters=[{'model_path': vla_model_path, 'enabled': True}]
    ))
```

```bash
# Test VLA node in sim (uses ONNX model, no oracle)
ros2 launch ocelot sim_launch.py use_vla:=true vla_model_path:=/path/to/vla_v0.1.onnx headless:=true
```

### 5c — Unit tests

**File**: `tests/train/test_vla_node.py`

Test the inference path without ROS:
- Load ONNX model
- Run a random (1,3,224,224) frame through
- Assert output shape (1,2) and values in [-2, 2]

**Success gate**: VLA node publishes non-zero `/cmd_vel` messages when enabled and a camera
feed is running.

---

## Step 6 — Automated Sim Evaluation

**File**: `sim/eval.py`

This script drives a complete evaluation run: launches the VLA node in the sim environment,
runs it across a test split of scenarios, and produces a structured metrics report. This
is the gate that determines whether a model is good enough to deploy.

### 6a — Evaluation metrics

| Metric | Definition | Pass threshold |
|---|---|---|
| mean tracking error | mean pixel distance from image center to face bbox center | < 15 px at 2m |
| p95 tracking error | 95th percentile of per-frame tracking error | < 40 px |
| face-lost rate | fraction of frames where Haar cascade detects no face | < 10% |
| per-label-type MSE | mean (pan_err² + tilt_err²) broken out by label key | no label > 0.2 |
| command response time | frames until tracking error < 20px after episode start | < 8 frames |

**Why Haar cascade for eval**: the oracle tracks perfectly via ground-truth pose, but for eval
we want to know if the VLA is centering the *visible* face, not just issuing low-error
velocity commands. Running Haar cascade on eval frames gives a pixel-space error estimate
independent of the oracle. Haar detection rate in eval frames is expected to be ~60–80%.

### 6b — Eval script design

```
sim/eval.py
  --model_path   ONNX_PATH     # path to VLA model
  --dataset_dir  DATASET_DIR   # for test.txt episode list
  --n_scenarios  N             # number of test scenarios (default: all in test.txt)
  --output       EVAL_JSON     # where to write results
  --language_cmd CMD           # language command to use (or "per_episode" to use dataset cmd)
```

Architecture:
1. Read test episode IDs from `test.txt`
2. For each episode: replay its scenario config (seed → ScenarioGenerator) to recreate
   entities in Gazebo
3. Run VLA node inference on each frame (onnxruntime, no ROS — just process the frames
   from the HDF5 file offline)
4. For VLA actions: compare predicted (pan_vel, tilt_vel) vs. oracle ground truth
5. Compute metrics per episode, aggregate across all test episodes
6. Write `eval_results.json`

**Offline vs. online eval**: The simplest eval is offline — replay episode frames from HDF5,
compare VLA outputs to oracle labels, compute MSE. This doesn't require a running Gazebo
instance. Online eval (deploy VLA node into running sim) is more realistic but slower and
harder to automate. Start with offline; add online as a stretch goal.

```python
# Offline eval core
session = ort.InferenceSession(args.model_path)
results = []
for ep_path in test_episodes:
    with h5py.File(ep_path) as f:
        frames   = f["frames"][:]      # (N,224,224,3) uint8
        pan_gt   = f["pan_vel"][:]     # oracle labels
        tilt_gt  = f["tilt_vel"][:]
        cmd      = f["cmd"][()]
        label_key = f["label_key"][()]

    pred = session.run(None, {
        "frames":         preprocess(frames),     # (N,3,224,224) float32
        "input_ids":      tokenize(cmd)[0],
        "attention_mask": tokenize(cmd)[1],
    })[0]  # (N, 2)

    mse = np.mean((pred - np.stack([pan_gt, tilt_gt], axis=1))**2)
    results.append({"ep": ep_path.name, "mse": mse, "label_key": label_key})

# Aggregate
overall_mse = np.mean([r["mse"] for r in results])
per_label   = {k: np.mean([r["mse"] for r in results if r["label_key"] == k])
               for k in label_keys}

report = {
    "model_path": str(args.model_path),
    "n_episodes": len(results),
    "overall_mse": overall_mse,
    "per_label_mse": per_label,
    "pass": overall_mse < MSE_THRESHOLD,
}
json.dump(report, open(args.output, "w"), indent=2)
print("PASS" if report["pass"] else "FAIL")
sys.exit(0 if report["pass"] else 1)
```

### 6c — Pass/fail gates

Hard gates for a model to receive the `deployable` tag:
- `overall_mse` < 0.05 (roughly < 15 px mean error)
- No per-label MSE > 0.2
- Eval runs to completion without crash

### 6d — Unit tests

**File**: `tests/train/test_eval.py`

- Feed `eval.py` a dummy ONNX model that outputs zeros for all inputs
- Assert the report marks it as FAIL (zero velocity = max tracking error)
- Feed a "perfect oracle" dummy that outputs the ground-truth labels from HDF5
- Assert the report marks it as PASS

---

## Step 7 — CI Integration

**File**: `.github/workflows/ci.yml`

### 7a — PR checks (fast, no sim)

Run on every pull request. No Gazebo, no GPU required:

```yaml
on: [pull_request]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - name: Install dependencies
        run: pip install -r requirements-train.txt pytest
      - name: Lint
        run: python -m flake8 train/ ocelot/ sim/ tests/ --max-line-length 100
      - name: Unit tests (no sim, no GPU)
        run: pytest tests/train/ -v --tb=short
        # tests/sim/ require sim container — skip in PR checks
```

### 7b — Model eval gate (on merge to main)

Run on every merge to `main` that changes `train/`, `ocelot/vla_node.py`, or `models/`:

```yaml
on:
  push:
    branches: [main]
    paths: ["train/**", "ocelot/vla_node.py", "models/**"]

jobs:
  eval:
    runs-on: ubuntu-latest  # no GPU needed — onnxruntime CPU is fast enough for eval
    steps:
      - uses: actions/checkout@v4
      - name: Pull dataset (DVC)
        run: |
          pip install dvc h5py onnxruntime numpy
          dvc pull dataset/  # requires DVC remote credentials in CI secrets
      - name: Run offline eval
        run: |
          python3 sim/eval.py \
            --model_path models/vla_v0.1.onnx \
            --dataset_dir dataset/ \
            --output eval_results.json
      - name: Upload eval report
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: eval_results.json
      - name: Check pass/fail
        run: python3 -c "
          import json, sys
          r = json.load(open('eval_results.json'))
          print(f'Overall MSE: {r[\"overall_mse\"]:.4f}')
          sys.exit(0 if r['pass'] else 1)
        "
```

If the eval job fails, the CI marks the commit as failed and blocks any deployment tag.

### 7c — End-to-end test

Validate CI works by pushing a deliberately bad model (all-zero action head weights).
Use `export_onnx.py` to produce the file — do not save a PyTorch state dict with an `.onnx`
extension; `onnxruntime` will reject it immediately.

```bash
# 1. Train (or load) a model, then zero the action head in-place before exporting
python3 -c "
import train.model, train.export_onnx, torch
m = train.model.VLAModel(pretrained=True)
for p in m.action_head.parameters():
    p.data.zero_()
# export_onnx.export(m, 'models/vla_v0.1.onnx')  # call your export helper
"
# 2. Commit the bad ONNX file and push to main
# 3. Confirm CI eval reports FAIL and blocks the commit
# 4. Replace with the real v0.1 ONNX and confirm CI reports PASS
```
Confirm CI reports FAIL and blocks the commit. Then push the real v0.1 model and confirm PASS.

---

## Directory Layout After Phase 3

```
ocelot/
├── ocelot/
│   ├── vla_node.py             # NEW — ONNX inference ROS 2 node
│   └── ... (Phase 1/2 nodes unchanged)
├── train/
│   ├── dataset.py              # NEW — HDF5 DataLoader
│   ├── model.py                # NEW — VLA model (DINOv2 + CLIP + fusion + head)
│   ├── train.py                # NEW — training script + MLflow
│   └── export_onnx.py          # NEW — PyTorch → ONNX export + validation
├── sim/
│   ├── eval.py                 # NEW — offline eval script with pass/fail gate
│   └── ... (Phase 2 files unchanged)
├── models/
│   ├── vla_v0.1.onnx           # NEW — DVC-tracked
│   └── vla_v0.1.onnx.dvc      # NEW — git-tracked DVC pointer
├── tests/
│   ├── train/
│   │   ├── test_dataset.py     # NEW
│   │   ├── test_model.py       # NEW
│   │   ├── test_vla_node.py    # NEW
│   │   └── test_eval.py        # NEW
│   └── sim/ (Phase 2 tests unchanged)
├── requirements-train.txt      # NEW — pinned ML deps
├── .github/
│   └── workflows/
│       └── ci.yml              # NEW — PR lint+test + main eval gate
└── mlruns/                     # MLflow tracking (gitignored)
```

---

## Phase 3 Validation Checklist

Before calling Phase 3 complete:

- [ ] `pytest tests/train/ -v` — all tests pass (no GPU, no ROS, no Gazebo)
- [ ] VLA model v0.1 forward pass: shapes correct, gradients only through trainable params
- [ ] Training run: `val_loss` decreases over 10 epochs on 50k dataset
- [ ] ONNX model validates within 1e-4 of PyTorch model
- [ ] `sim/eval.py` on test split: `overall_mse` < 0.05 → PASS verdict
- [ ] `eval.py` with zero-model → FAIL verdict (gate is meaningful)
- [ ] CI runs lint + unit tests on a test PR → passes
- [ ] CI runs eval on merge to main → PASS for v0.1 model
- [ ] CI runs eval on deliberately bad model → FAIL, commit blocked
- [ ] MLflow dashboard shows v0.1 metrics and sweep comparison
- [ ] `models/vla_v0.1.onnx` tracked by DVC

---

## Key Risks and Mitigations

| Risk | Mitigation |
|---|---|
| DINOv2 features don't generalize to Gazebo-rendered faces | Aggressive domain randomization from Phase 2 is the primary defense. If val_loss is stuck at >0.1 after 10 epochs, add LoRA fine-tuning of DINOv2 layers (Phase 4 option). |
| CLIP text encoder doesn't discriminate well between label types | Check per-label-type MSE in MLflow. If "follow the person on the left" and "follow the person on the right" have identical MSE, the text encoder isn't being used. Add a text-ablation run (random text tokens) to confirm the model uses language. |
| Training bottlenecked by DataLoader I/O | With RTX 2070 the GPU is fast enough that disk I/O may become the bottleneck at 5M frames. Set `num_workers=4` in DataLoader. If still bottlenecked, precompute frozen DINOv2+CLIP encoder outputs for the full dataset once and cache as tensors — the frozen encoders become a one-time offline cost and the training loop only trains the fusion+head. |
| ONNX export fails due to dynamic control flow | Avoid Python conditionals that depend on tensor values inside the model. Use `torch.where` instead of `if`. Test export with `dynamo=True` if standard export fails. |
| Offline eval MSE doesn't correlate with real tracking quality | After Phase 3, verify in Phase 4 by running the ONNX node live in sim and measuring pixel tracking error. If the correlation is low, replace offline eval with online eval (requires headless Gazebo in CI). |

---

## Phase 4 Handoff Criteria

Phase 4 (Hardware Deployment and MLOps) can begin when:

1. `sim/eval.py` reports PASS for vla_v0.1.onnx on the held-out test split
2. CI eval gate is live and blocks on FAIL
3. MLflow tracking shows meaningful per-label-type metrics
4. ONNX model is DVC-tracked and reproducible from `dvc pull`
5. VLA node runs in sim without crashing (confirmed during Step 5 testing)
