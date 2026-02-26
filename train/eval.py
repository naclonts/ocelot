"""train/eval.py — Evaluate a trained VLAModel checkpoint on the val (or test) split.

Reports:
  - Per-axis RMSE and Pearson correlation (pan, tilt)
  - Sign-agreement % (predicted direction matches oracle)
  - Per-label-key breakdown (same keys as training per_label_mse)
  - Fraction of near-zero targets (|target| < 0.05) where model also outputs near-zero

Optionally:
  --plot      scatter plot: predicted vs GT for pan and tilt (requires matplotlib)
  --episodes  time-series plot for N random episodes (requires matplotlib)

Usage:
    source .venv/bin/activate

    # Quick eval on val split:
    python3 train/eval.py \\
        --checkpoint runs/v0.0-smoke/best.pt \\
        --dataset_dir sim/dataset/

    # Full eval with plots:
    python3 train/eval.py \\
        --checkpoint runs/v0.0-smoke/best.pt \\
        --dataset_dir sim/dataset/ \\
        --split val \\
        --plot \\
        --episodes 4

    # Test-split eval:
    python3 train/eval.py \\
        --checkpoint runs/v0.0-smoke/best.pt \\
        --dataset_dir sim/dataset/ \\
        --split test
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import torch
from torch.utils.data import DataLoader

from train.dataset import OcelotDataset
from train.model import VLAModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def sign_agreement(pred: np.ndarray, target: np.ndarray) -> float:
    """% of samples where sign(pred) == sign(target), ignoring near-zero targets."""
    mask = np.abs(target) > 0.01  # skip frames where oracle is essentially stopped
    if mask.sum() == 0:
        return float("nan")
    return float((np.sign(pred[mask]) == np.sign(target[mask])).mean() * 100)


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: VLAModel,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Run inference on a DataLoader. Returns arrays of preds, targets, label_keys."""
    model.eval()

    all_pred   = []
    all_target = []
    all_keys   = []

    for batch in loader:
        frames         = batch["frames"].to(device, non_blocking=True)
        targets        = batch["targets"].to(device, non_blocking=True)
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        label_keys     = batch["label_keys"]

        pred = model(frames, input_ids, attention_mask)

        all_pred.append(pred.cpu().numpy())
        all_target.append(targets.cpu().numpy())
        all_keys.extend(label_keys)

    return {
        "pred":   np.concatenate(all_pred,   axis=0),   # (N, 2)
        "target": np.concatenate(all_target, axis=0),   # (N, 2)
        "keys":   all_keys,                             # list[str], length N
    }


# ---------------------------------------------------------------------------
# Episode time-series data
# ---------------------------------------------------------------------------

def load_episode_series(
    dataset_dir: Path,
    split: str,
    n_episodes: int,
    tokenizer,
    model: VLAModel,
    device: torch.device,
    seed: int = 0,
) -> list[dict]:
    """Load N random full episodes and run inference frame-by-frame.

    Returns list of dicts with keys: ep_id, pred (T,2), target (T,2), cmd.
    """
    import h5py

    rng = np.random.default_rng(seed)

    # Gather all episode h5 paths for this split
    split_files = OcelotDataset._find_split_files(dataset_dir, split)
    all_paths = []
    for sf in split_files:
        shard_dir = sf.parent
        ep_ids = [l.strip() for l in sf.read_text().splitlines() if l.strip()]
        for ep_id in ep_ids:
            h5_path = shard_dir / "episodes" / f"ep_{ep_id}.h5"
            if h5_path.exists():
                all_paths.append((ep_id, h5_path))

    chosen = rng.choice(len(all_paths), size=min(n_episodes, len(all_paths)), replace=False)

    results = []
    for idx in chosen:
        ep_id, h5_path = all_paths[idx]
        with h5py.File(h5_path, "r") as f:
            frames_np = f["frames"][:]          # (T, 224, 224, 3) uint8
            pan_vel   = f["pan_vel"][:]          # (T,) float32
            tilt_vel  = f["tilt_vel"][:]         # (T,) float32
            cmd_bytes = f["cmd"][()]
            cmd = cmd_bytes.decode("utf-8") if isinstance(cmd_bytes, bytes) else str(cmd_bytes)

        # Normalise frames
        from train.dataset import IMAGENET_MEAN, IMAGENET_STD
        frames_f32 = frames_np.astype(np.float32) / 255.0
        frames_f32 = (frames_f32 - IMAGENET_MEAN) / IMAGENET_STD
        frames_t   = torch.from_numpy(frames_f32.transpose(0, 3, 1, 2))  # (T,3,H,W)

        # Tokenise command
        tokens = tokenizer(
            [cmd] * len(frames_t),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )

        # Inference in chunks to avoid OOM
        chunk = 64
        preds = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(frames_t), chunk):
                fr = frames_t[i:i+chunk].to(device)
                ii = tokens["input_ids"][i:i+chunk].to(device)
                am = tokens["attention_mask"][i:i+chunk].to(device)
                preds.append(model(fr, ii, am).cpu().numpy())

        pred_arr   = np.concatenate(preds, axis=0)  # (T, 2)
        target_arr = np.stack([pan_vel, tilt_vel], axis=1)  # (T, 2)

        results.append({
            "ep_id":  ep_id,
            "pred":   pred_arr,
            "target": target_arr,
            "cmd":    cmd,
        })

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: dict) -> None:
    pred   = results["pred"]    # (N, 2)
    target = results["target"]  # (N, 2)
    keys   = results["keys"]

    pan_pred,  tilt_pred  = pred[:, 0],   pred[:, 1]
    pan_tgt,   tilt_tgt   = target[:, 0], target[:, 1]

    pan_rmse  = float(np.sqrt(np.mean((pan_pred  - pan_tgt)  ** 2)))
    tilt_rmse = float(np.sqrt(np.mean((tilt_pred - tilt_tgt) ** 2)))
    pan_r     = pearson(pan_pred, pan_tgt)
    tilt_r    = pearson(tilt_pred, tilt_tgt)
    pan_sign  = sign_agreement(pan_pred,  pan_tgt)
    tilt_sign = sign_agreement(tilt_pred, tilt_tgt)

    # Near-zero quietness: when |target| < 0.05, is |pred| < 0.1?
    quiet_mask = np.abs(pan_tgt) < 0.05
    pan_quiet  = (np.abs(pan_pred[quiet_mask]) < 0.10).mean() * 100 if quiet_mask.any() else float("nan")
    quiet_mask = np.abs(tilt_tgt) < 0.05
    tilt_quiet = (np.abs(tilt_pred[quiet_mask]) < 0.10).mean() * 100 if quiet_mask.any() else float("nan")

    print()
    print("=" * 60)
    print(f"{'OVERALL METRICS':^60}")
    print("=" * 60)
    print(f"  Samples:              {len(pred):>8,}")
    print(f"  {'':30}  {'pan':>8}  {'tilt':>8}")
    print(f"  {'RMSE (rad/s)':30}  {pan_rmse:>8.4f}  {tilt_rmse:>8.4f}")
    print(f"  {'Pearson r':30}  {pan_r:>8.3f}  {tilt_r:>8.3f}")
    print(f"  {'Sign agreement (%)':30}  {pan_sign:>8.1f}  {tilt_sign:>8.1f}")
    print(f"  {'Quiet @ zero (%)':30}  {pan_quiet:>8.1f}  {tilt_quiet:>8.1f}")
    print(f"  {'Pred range':30}  [{pan_pred.min():.3f}, {pan_pred.max():.3f}]  [{tilt_pred.min():.3f}, {tilt_pred.max():.3f}]")
    print(f"  {'GT range':30}  [{pan_tgt.min():.3f}, {pan_tgt.max():.3f}]  [{tilt_tgt.min():.3f}, {tilt_tgt.max():.3f}]")

    # Per-label breakdown
    label_set = sorted(set(keys))
    if len(label_set) > 1:
        print()
        print("-" * 60)
        print(f"  {'PER-LABEL RMSE':30}  {'pan':>8}  {'tilt':>8}  {'N':>6}")
        print("-" * 60)
        for lk in label_set:
            mask = np.array([k == lk for k in keys])
            p_p = pan_pred[mask];  p_t = pan_tgt[mask]
            t_p = tilt_pred[mask]; t_t = tilt_tgt[mask]
            print(
                f"  {lk:30}  "
                f"{np.sqrt(np.mean((p_p-p_t)**2)):>8.4f}  "
                f"{np.sqrt(np.mean((t_p-t_t)**2)):>8.4f}  "
                f"{mask.sum():>6}"
            )

    print("=" * 60)
    print()

    # Interpretation guide
    typical_signal = 0.15  # rad/s typical oracle output magnitude
    pan_pct = pan_rmse / typical_signal * 100
    print("Interpretation:")
    print(f"  Typical oracle signal magnitude: ~{typical_signal:.2f} rad/s")
    print(f"  Pan RMSE as % of signal:  {pan_pct:.0f}%  ", end="")
    if pan_pct < 15:
        print("(good)")
    elif pan_pct < 35:
        print("(acceptable for early training)")
    else:
        print("(high — model not fitting well)")
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def scatter_plot(results: dict, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    pred   = results["pred"]
    target = results["target"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, axis_idx, name in zip(axes, [0, 1], ["pan_vel", "tilt_vel"]):
        p = pred[:, axis_idx]
        t = target[:, axis_idx]
        ax.scatter(t, p, s=1, alpha=0.2, rasterized=True)
        lim = max(abs(t).max(), abs(p).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1, label="ideal")
        ax.set_xlabel(f"GT {name} (rad/s)")
        ax.set_ylabel(f"Pred {name} (rad/s)")
        ax.set_title(f"{name}  r={pearson(p,t):.3f}")
        ax.legend(fontsize=8)
        ax.set_aspect("equal")

    fig.suptitle("Predicted vs Ground-Truth Actions")
    plt.tight_layout()
    out = out_dir / "scatter.png"
    plt.savefig(out, dpi=150)
    log.info("Saved scatter plot → %s", out)
    plt.close()


def episode_plot(episodes: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    n = len(episodes)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n), squeeze=False)

    for row, ep in enumerate(episodes):
        t = np.arange(len(ep["pred"]))
        for col, (axis_idx, name) in enumerate([(0, "pan_vel"), (1, "tilt_vel")]):
            ax = axes[row][col]
            ax.plot(t, ep["target"][:, axis_idx], label="GT",   color="steelblue", linewidth=1)
            ax.plot(t, ep["pred"][:,   axis_idx], label="Pred", color="orangered", linewidth=1, alpha=0.8)
            ax.set_ylabel("rad/s")
            ax.set_xlabel("frame")
            ax.set_title(f"ep {ep['ep_id']}  {name}\n\"{ep['cmd'][:40]}\"")
            ax.legend(fontsize=7, loc="upper right")
            ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

    plt.tight_layout()
    out = out_dir / "episodes.png"
    plt.savefig(out, dpi=150)
    log.info("Saved episode plot → %s", out)
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained VLAModel checkpoint")
    p.add_argument("--checkpoint",   required=True, type=Path, help="Path to best.pt")
    p.add_argument("--dataset_dir",  required=True, type=Path)
    p.add_argument("--split",        default="val", choices=["val", "test"])
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--n_fusion_layers", type=int, default=2)
    p.add_argument("--n_heads",      type=int, default=6)
    p.add_argument("--plot",         action="store_true", help="Save scatter.png")
    p.add_argument("--episodes",     type=int, default=0,
                   help="If > 0, plot N random episodes as time series")
    p.add_argument("--out_dir",      type=Path, default=None,
                   help="Where to save plots (defaults to checkpoint parent dir)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = args.out_dir or args.checkpoint.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    from transformers import CLIPTokenizerFast
    log.info("Loading tokenizer …")
    tokenizer = CLIPTokenizerFast.from_pretrained(VLAModel.CLIP_ID)

    log.info("Loading model from %s …", args.checkpoint)
    model = VLAModel(
        n_fusion_layers=args.n_fusion_layers,
        n_heads=args.n_heads,
        pretrained=True,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    log.info("Checkpoint loaded.")

    # Val/test DataLoader
    collate = lambda b: OcelotDataset.collate_fn(b, tokenizer)  # noqa: E731
    ds      = OcelotDataset(args.split, args.dataset_dir)
    loader  = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    log.info("%s split: %d frames", args.split, len(ds))

    log.info("Running inference …")
    results = run_inference(model, loader, device)

    print_report(results)

    if args.plot:
        scatter_plot(results, out_dir)

    if args.episodes > 0:
        log.info("Loading %d episodes for time-series plot …", args.episodes)
        episodes = load_episode_series(
            args.dataset_dir, args.split, args.episodes, tokenizer, model, device
        )
        episode_plot(episodes, out_dir)


if __name__ == "__main__":
    main()
