"""train/train.py — Behavioral cloning training loop with MLflow tracking.

Loss: MSE on (pan_vel, tilt_vel) — behavioral cloning from oracle labels.
Optimizer: Adam on trainable params only (fusion + head ≈ 1.5-2M params).
Schedule: Cosine annealing over all epochs.
AMP: optional fp16 via --amp (halves VRAM, ~2x faster on RTX 2070).

Usage:
    source .venv/bin/activate

    # Smoke test on the existing 400-episode dataset:
    python3 train/train.py \\
        --dataset_dir sim/dataset/ \\
        --output_dir runs/v0.0-smoke/ \\
        --epochs 3 \\
        --batch_size 32 \\
        --experiment ocelot-smoke

    # Full training with AMP:
    python3 train/train.py \\
        --dataset_dir sim/dataset/ \\
        --output_dir runs/v0.1/ \\
        --epochs 20 \\
        --batch_size 64 \\
        --amp \\
        --experiment ocelot-v0.1

    # Inspect results:
    mlflow ui  # open http://localhost:5000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path when run as a standalone script
# (pytest picks this up via conftest.py; direct invocation needs it here).
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import mlflow
import torch
import torch.nn as nn
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
# DataLoader helpers
# ---------------------------------------------------------------------------

def build_loaders(
    dataset_dir: Path,
    batch_size: int,
    num_workers: int,
    tokenizer,
    max_episodes: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders backed by OcelotDataset."""
    collate = lambda b: OcelotDataset.collate_fn(b, tokenizer)  # noqa: E731

    train_ds = OcelotDataset("train", dataset_dir, max_episodes=max_episodes)
    val_ds   = OcelotDataset("val",   dataset_dir, max_episodes=max_episodes)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training / evaluation steps
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: VLAModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: "torch.cuda.amp.GradScaler | None",
    grad_clip: float = 1.0,
) -> float:
    """Run one training epoch. Returns mean batch loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        frames         = batch["frames"].to(device, non_blocking=True)
        targets        = batch["targets"].to(device, non_blocking=True)
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                pred = model(frames, input_ids, attention_mask)
                loss = criterion(pred, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(frames, input_ids, attention_mask)
            loss = criterion(pred, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else float("nan")


@torch.no_grad()
def evaluate(
    model: VLAModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Evaluate on val/test split.

    Returns:
        mean_loss:     mean batch MSE over the split
        per_label_mse: dict mapping label_key → mean per-sample MSE
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    label_losses: dict[str, list[float]] = {}

    for batch in loader:
        frames         = batch["frames"].to(device, non_blocking=True)
        targets        = batch["targets"].to(device, non_blocking=True)
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        label_keys     = batch["label_keys"]

        pred = model(frames, input_ids, attention_mask)
        loss = criterion(pred, targets)

        total_loss += loss.item()
        n_batches += 1

        # Per-sample MSE for per-label breakdown
        per_sample = ((pred - targets) ** 2).mean(dim=1)  # (B,)
        for key, val in zip(label_keys, per_sample.tolist()):
            label_losses.setdefault(key, []).append(val)

    mean_loss = total_loss / n_batches if n_batches > 0 else float("nan")
    per_label_mse = {k: sum(v) / len(v) for k, v in sorted(label_losses.items())}
    return mean_loss, per_label_mse


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train VLA model on Ocelot dataset with MLflow tracking"
    )
    p.add_argument(
        "--dataset_dir", required=True, type=Path,
        help="Path to dataset/ with shard_*/train.txt layout (or flat single-shard)",
    )
    p.add_argument(
        "--output_dir", required=True, type=Path,
        help="Directory for checkpoints; created if it doesn't exist",
    )
    p.add_argument("--epochs",          type=int,   default=10)
    p.add_argument("--batch_size",      type=int,   default=64)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--n_fusion_layers", type=int,   default=2)
    p.add_argument("--n_heads",         type=int,   default=6)
    p.add_argument("--num_workers",     type=int,   default=4,
                   help="DataLoader worker processes (set 0 to debug in main process)")
    p.add_argument("--amp",             action="store_true",
                   help="Mixed-precision training (fp16); halves VRAM, ~2× faster")
    p.add_argument("--max_episodes",    type=int,   default=None,
                   help="Cap total episodes per split (train and val); useful for fast sweep runs")
    p.add_argument("--experiment",      type=str,   default="ocelot",
                   help="MLflow experiment name")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Tokenizer — same vocabulary as the model's frozen CLIPTextModel
    from transformers import CLIPTokenizerFast
    log.info("Loading CLIP tokenizer …")
    tokenizer = CLIPTokenizerFast.from_pretrained(VLAModel.CLIP_ID)

    # DataLoaders
    log.info("Building DataLoaders from %s …", args.dataset_dir)
    train_loader, val_loader = build_loaders(
        args.dataset_dir, args.batch_size, args.num_workers, tokenizer,
        max_episodes=args.max_episodes,
    )
    log.info(
        "Dataset: %d train frames / %d val frames",
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    # Model
    log.info("Loading VLAModel (pretrained encoders) …")
    model = VLAModel(
        n_fusion_layers=args.n_fusion_layers,
        n_heads=args.n_heads,
        pretrained=True,
    ).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    log.info("Params: %d trainable / %d total", n_trainable, n_total)

    # Loss, optimizer, scheduler
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler("cuda") if args.amp else None
    if args.amp:
        log.info("AMP enabled (GradScaler)")

    # MLflow
    # autolog captures optimizer config, system metrics, and model summary.
    # Epoch-level metrics (train_loss, val_loss, etc.) are logged manually below
    # since autolog's epoch hooks require PyTorch Lightning.
    mlflow.pytorch.autolog(log_every_n_epoch=1, log_models=False)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run():
        mlflow.log_params({
            "epochs":          args.epochs,
            "batch_size":      args.batch_size,
            "lr":              args.lr,
            "n_fusion_layers": args.n_fusion_layers,
            "n_heads":         args.n_heads,
            "amp":             args.amp,
            "dataset_dir":     str(args.dataset_dir),
            "max_episodes":    args.max_episodes,
            "n_trainable":     n_trainable,
        })

        best_val_loss = float("inf")
        best_ckpt = args.output_dir / "best.pt"

        for epoch in range(args.epochs):
            log.info("── Epoch %d/%d ──", epoch + 1, args.epochs)

            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler
            )
            val_loss, per_label_mse = evaluate(
                model, val_loader, criterion, device
            )
            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            metrics: dict[str, float] = {
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "lr":         current_lr,
                **{f"val_mse_{k}": v for k, v in per_label_mse.items()},
            }
            mlflow.log_metrics(metrics, step=epoch)

            log.info(
                "  train_loss=%.5f  val_loss=%.5f  lr=%.2e",
                train_loss, val_loss, current_lr,
            )
            for k, v in per_label_mse.items():
                log.info("  val_mse_%-22s %.5f", k, v)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_ckpt)
                log.info("  → saved best checkpoint (val_loss=%.5f)", val_loss)

        # Log best checkpoint as MLflow artifact
        if best_ckpt.exists():
            mlflow.log_artifact(str(best_ckpt))

        log.info("Done. Best val_loss=%.5f  checkpoint=%s", best_val_loss, best_ckpt)


if __name__ == "__main__":
    main()
