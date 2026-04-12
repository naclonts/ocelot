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
from dataclasses import asdict
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
from train.transforms import DomainRandomizationConfig, DomainRandomizationTransform

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
    shards: list[int] | None = None,
    label_keys: list[str] | None = None,
    *,
    train_augment=None,
) -> tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders backed by OcelotDataset."""
    collate = lambda b: OcelotDataset.collate_fn(b, tokenizer)  # noqa: E731

    train_ds = OcelotDataset(
        "train",
        dataset_dir,
        augment=train_augment,
        max_episodes=max_episodes,
        shards=shards,
        label_keys=label_keys,
    )
    val_ds = OcelotDataset(
        "val",
        dataset_dir,
        max_episodes=max_episodes,
        shards=shards,
        label_keys=label_keys,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
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
    scheduler: "torch.optim.lr_scheduler.LRScheduler | None" = None,
    grad_clip: float = 1.0,
    confidence_weight: float = 1.0,
) -> float:
    """Run one training epoch. Returns mean batch loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        confidence_tgt = torch.tensor(
            [0.0 if key == "no_face" else 1.0 for key in batch["label_keys"]],
            dtype=torch.float32,
            device=device,
        )

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                pred, confidence = model(
                    frames,
                    input_ids,
                    attention_mask,
                    gate_actions=False,
                    return_confidence=True,
                )
                loss = criterion(pred, targets)
                loss = loss + confidence_weight * nn.functional.binary_cross_entropy(
                    confidence, confidence_tgt
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred, confidence = model(
                frames,
                input_ids,
                attention_mask,
                gate_actions=False,
                return_confidence=True,
            )
            loss = criterion(pred, targets)
            loss = loss + confidence_weight * nn.functional.binary_cross_entropy(
                confidence, confidence_tgt
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else float("nan")


@torch.no_grad()
def evaluate(
    model: VLAModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    include_aux: bool = False,
) -> tuple[float, dict[str, float]] | tuple[float, dict[str, float], dict[str, float]]:
    """Evaluate on val/test split.

    Returns:
        mean_loss:     mean batch MSE over the split
        per_label_mse: dict mapping label_key → mean per-sample MSE
        aux_metrics:    optional confidence metrics when include_aux=True
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    label_losses: dict[str, list[float]] = {}
    confidence_hits = 0
    confidence_total = 0

    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        label_keys = batch["label_keys"]

        pred, confidence = model(
            frames,
            input_ids,
            attention_mask,
            return_confidence=True,
        )
        loss = criterion(pred, targets)

        total_loss += loss.item()
        n_batches += 1

        # Per-sample MSE for per-label breakdown
        per_sample = ((pred - targets) ** 2).mean(dim=1)  # (B,)
        for key, val in zip(label_keys, per_sample.tolist()):
            label_losses.setdefault(key, []).append(val)

        if include_aux:
            face_present = torch.tensor(
                [0 if key == "no_face" else 1 for key in label_keys],
                device=device,
                dtype=torch.long,
            )
            confidence_hits += int((confidence >= 0.5).long().eq(face_present).sum().item())
            confidence_total += len(label_keys)

    mean_loss = total_loss / n_batches if n_batches > 0 else float("nan")
    per_label_mse = {k: sum(v) / len(v) for k, v in sorted(label_losses.items())}
    if include_aux:
        aux = {
            "confidence_accuracy": (
                confidence_hits / confidence_total if confidence_total > 0 else float("nan")
            ),
        }
        return mean_loss, per_label_mse, aux
    return mean_loss, per_label_mse


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train VLA model on Ocelot dataset with MLflow tracking"
    )
    p.add_argument(
        "--dataset_dir",
        required=True,
        type=Path,
        help="Path to dataset/ with shard_*/train.txt layout (or flat single-shard)",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory for checkpoints; created if it doesn't exist",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--warmup_steps", type=int, default=500, help="Linear LR warmup steps (0 to disable)"
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="CKPT",
        help="Resume training from checkpoint (best.pt or last.pt)",
    )
    p.add_argument("--n_fusion_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=6)
    p.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="DataLoader worker processes (set 0 to debug in main process)",
    )
    p.add_argument(
        "--amp",
        action="store_true",
        help="Mixed-precision training (fp16); halves VRAM, ~2× faster",
    )
    p.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Cap total episodes per split (train and val); useful for fast sweep runs",
    )
    p.add_argument(
        "--shards",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="Only train on these shard numbers (e.g. --shards 26 27 28 29 30 31)",
    )
    p.add_argument(
        "--label_keys",
        type=str,
        nargs="+",
        default=None,
        metavar="KEY",
        help=("Only train on episodes with these label_key values (e.g. --label_keys track)"),
    )
    p.add_argument(
        "--domain_randomization",
        action="store_true",
        help="Apply training-only visual augmentations before ImageNet normalization",
    )
    p.add_argument("--color_jitter_prob", type=float, default=1.0)
    p.add_argument("--brightness_jitter", type=float, default=0.2)
    p.add_argument("--contrast_jitter", type=float, default=0.2)
    p.add_argument("--saturation_jitter", type=float, default=0.2)
    p.add_argument("--hue_jitter", type=float, default=0.1)
    p.add_argument("--blur_prob", type=float, default=0.3)
    p.add_argument("--blur_kernel_min", type=int, default=3)
    p.add_argument("--blur_kernel_max", type=int, default=7)
    p.add_argument("--blur_sigma_min", type=float, default=0.1)
    p.add_argument("--blur_sigma_max", type=float, default=2.0)
    p.add_argument("--noise_prob", type=float, default=0.3)
    p.add_argument("--noise_sigma_max", type=float, default=0.05)
    p.add_argument("--cutout_prob", type=float, default=0.2)
    p.add_argument("--cutout_min_scale", type=float, default=0.02)
    p.add_argument("--cutout_max_scale", type=float, default=0.08)
    p.add_argument("--cutout_max_patches", type=int, default=1)
    p.add_argument("--gradient_prob", type=float, default=0.3)
    p.add_argument("--gradient_strength_min", type=float, default=0.08)
    p.add_argument("--gradient_strength_max", type=float, default=0.25)
    p.add_argument(
        "--confidence_weight",
        type=float,
        default=1.0,
        help="Weight for the confidence-head binary loss (0 to disable)",
    )
    p.add_argument("--experiment", type=str, default="ocelot", help="MLflow experiment name")
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
    if args.shards is not None:
        log.info("Shards: %s", args.shards)
    if args.label_keys is not None:
        log.info("Label keys: %s", args.label_keys)
    train_augment = None
    if args.domain_randomization:
        aug_cfg = DomainRandomizationConfig(
            color_jitter_prob=args.color_jitter_prob,
            brightness=args.brightness_jitter,
            contrast=args.contrast_jitter,
            saturation=args.saturation_jitter,
            hue=args.hue_jitter,
            blur_prob=args.blur_prob,
            blur_kernel_min=args.blur_kernel_min,
            blur_kernel_max=args.blur_kernel_max,
            blur_sigma_min=args.blur_sigma_min,
            blur_sigma_max=args.blur_sigma_max,
            noise_prob=args.noise_prob,
            noise_sigma_max=args.noise_sigma_max,
            cutout_prob=args.cutout_prob,
            cutout_min_scale=args.cutout_min_scale,
            cutout_max_scale=args.cutout_max_scale,
            cutout_max_patches=args.cutout_max_patches,
            gradient_prob=args.gradient_prob,
            gradient_strength_min=args.gradient_strength_min,
            gradient_strength_max=args.gradient_strength_max,
        )
        train_augment = DomainRandomizationTransform(aug_cfg)
        log.info("Domain randomization enabled: %s", asdict(aug_cfg))
    train_loader, val_loader = build_loaders(
        args.dataset_dir,
        args.batch_size,
        args.num_workers,
        tokenizer,
        train_augment=train_augment,
        max_episodes=args.max_episodes,
        shards=args.shards,
        label_keys=args.label_keys,
    )
    train_ds = train_loader.dataset
    val_ds = val_loader.dataset
    log.info(
        "Dataset: %d episodes / %d train frames — %d episodes / %d val frames",
        train_ds.n_episodes,
        len(train_ds),
        val_ds.n_episodes,
        len(val_ds),
    )

    # Model
    log.info("Loading VLAModel (pretrained encoders) …")
    model = VLAModel(
        n_fusion_layers=args.n_fusion_layers,
        n_heads=args.n_heads,
        pretrained=True,
    ).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log.info("Params: %d trainable / %d total", n_trainable, n_total)

    # Loss, optimizer, scheduler
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - args.warmup_steps,
    )
    if args.warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-6 / args.lr,
            total_iters=args.warmup_steps,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[args.warmup_steps],
        )
    else:
        scheduler = cosine
    log.info(
        "Steps/epoch: %d, total: %d, warmup: %d",
        steps_per_epoch,
        total_steps,
        args.warmup_steps,
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
        mlflow.log_params(
            {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "n_fusion_layers": args.n_fusion_layers,
                "n_heads": args.n_heads,
                "amp": args.amp,
                "dataset_dir": str(args.dataset_dir),
                "max_episodes": args.max_episodes,
                "shards": str(args.shards) if args.shards else "all",
                "label_keys": str(args.label_keys) if args.label_keys else "all",
                "n_trainable": n_trainable,
                "warmup_steps": args.warmup_steps,
                "domain_randomization": args.domain_randomization,
                "confidence_weight": args.confidence_weight,
            }
        )
        if args.domain_randomization:
            mlflow.log_params({f"aug_{k}": v for k, v in asdict(aug_cfg).items()})

        best_val_loss = float("inf")
        start_epoch = 0
        best_ckpt = args.output_dir / "best.pt"
        last_ckpt = args.output_dir / "last.pt"

        if args.resume is not None:
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if scaler is not None and "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            log.info(
                "Resumed from %s (epoch %d, best_val_loss=%.5f)",
                args.resume,
                ckpt["epoch"],
                best_val_loss,
            )

        for epoch in range(start_epoch, args.epochs):
            log.info("── Epoch %d/%d ──", epoch + 1, args.epochs)

            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler,
                scheduler=scheduler,
                confidence_weight=args.confidence_weight,
            )
            val_result = evaluate(model, val_loader, criterion, device, include_aux=True)
            if len(val_result) == 3:
                val_loss, per_label_mse, aux_metrics = val_result
            else:
                val_loss, per_label_mse = val_result
                aux_metrics = {}

            current_lr = scheduler.get_last_lr()[0]
            metrics: dict[str, float] = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
                **{f"val_mse_{k}": v for k, v in per_label_mse.items()},
                **{f"val_{k}": v for k, v in aux_metrics.items()},
            }
            mlflow.log_metrics(metrics, step=epoch)

            log.info(
                "  train_loss=%.5f  val_loss=%.5f  lr=%.2e",
                train_loss,
                val_loss,
                current_lr,
            )
            for k, v in per_label_mse.items():
                log.info("  val_mse_%-22s %.5f", k, v)
            for k, v in aux_metrics.items():
                log.info("  val_%-26s %.5f", k, v)

            ckpt_state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            }
            if scaler is not None:
                ckpt_state["scaler"] = scaler.state_dict()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_state["best_val_loss"] = best_val_loss
                torch.save(ckpt_state, best_ckpt)
                log.info("  → saved best checkpoint (val_loss=%.5f)", val_loss)

            torch.save(ckpt_state, last_ckpt)

        # Log best checkpoint as MLflow artifact
        if best_ckpt.exists():
            mlflow.log_artifact(str(best_ckpt))

        log.info("Done. Best val_loss=%.5f  checkpoint=%s", best_val_loss, best_ckpt)


if __name__ == "__main__":
    main()
