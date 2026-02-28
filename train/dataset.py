"""train/dataset.py — OcelotDataset: HDF5-backed DataLoader for VLA training.

Dataset layout (one or more shards):

    dataset_dir/
        shard_0/
            train.txt       # one 6-digit episode ID per line (e.g. "000042")
            val.txt
            test.txt
            episodes/
                ep_000042.h5
                ...
        shard_1/
            ...

Each HDF5 episode file contains:
    frames    (N, 224, 224, 3) uint8  — RGB camera frames
    pan_vel   (N,)             float32 — oracle /cmd_vel angular.z (rad/s)
    tilt_vel  (N,)             float32 — oracle /cmd_vel angular.y (rad/s)
    cmd       scalar str               — language label (e.g. "track the face")
    label_key scalar str               — label type key (e.g. "basic_track")

Usage:
    from train.dataset import OcelotDataset
    ds = OcelotDataset("train", dataset_dir=Path("sim/dataset"))
    loader = DataLoader(ds, batch_size=64, num_workers=4, collate_fn=ds.collate_fn)
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# ImageNet statistics — DINOv2 expects these
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class OcelotDataset(Dataset):
    """Loads (frame, language_cmd, pan_vel, tilt_vel) tuples from HDF5 episodes.

    One episode = N frames; each frame is one sample. The index is built at
    construction time (a flat list of (h5_path, frame_idx) pairs) so that
    __getitem__ only reads a single frame at a time — no full episode is
    ever loaded into RAM.

    Args:
        split:       "train", "val", or "test"
        dataset_dir: path to the parent directory containing shard_* subdirs
                     (or a single shard directory with train/val/test.txt)
        transform:   optional callable applied to the (3,224,224) float32
                     tensor *after* ImageNet normalisation
    """

    def __init__(
        self,
        split: str,
        dataset_dir: Path,
        transform=None,
        max_episodes: int | None = None,
        shards: list[int] | None = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")

        dataset_dir = Path(dataset_dir)
        self.split = split
        self.transform = transform

        # Build flat index: list of (h5_path, frame_idx)
        self._index: list[tuple[Path, int]] = []

        # Support both a flat layout (single shard) and a sharded layout.
        split_files = self._find_split_files(dataset_dir, split)
        if shards is not None:
            shard_set = set(shards)
            split_files = [
                p for p in split_files
                if p.parent.name.startswith("shard_")
                and int(p.parent.name.split("_", 1)[1]) in shard_set
            ]
        if not split_files:
            raise FileNotFoundError(
                f"No {split}.txt files found under {dataset_dir}"
            )

        # Collect all (episodes_dir, ep_id) pairs across shards, then truncate.
        all_episodes: list[tuple[Path, str]] = []
        for split_file in sorted(split_files):
            shard_dir = split_file.parent
            episodes_dir = shard_dir / "episodes"
            ep_ids = [line.strip() for line in split_file.read_text().splitlines()
                      if line.strip()]
            for ep_id in ep_ids:
                all_episodes.append((episodes_dir, ep_id))

        if max_episodes is not None:
            all_episodes = all_episodes[:max_episodes]

        self.n_episodes = len(all_episodes)

        for episodes_dir, ep_id in all_episodes:
            h5_path = episodes_dir / f"ep_{ep_id}.h5"
            if not h5_path.exists():
                raise FileNotFoundError(f"Episode file not found: {h5_path}")
            with h5py.File(h5_path, "r") as f:
                n_frames = f["frames"].shape[0]
            for frame_idx in range(n_frames):
                self._index.append((h5_path, frame_idx))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_split_files(dataset_dir: Path, split: str) -> list[Path]:
        """Return all split.txt files found under dataset_dir."""
        # Direct layout: dataset_dir/train.txt
        direct = dataset_dir / f"{split}.txt"
        if direct.exists():
            return [direct]
        # Sharded layout: dataset_dir/shard_*/train.txt
        return sorted(dataset_dir.glob(f"shard_*/{split}.txt"))

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        h5_path, frame_idx = self._index[idx]

        with h5py.File(h5_path, "r", swmr=True) as f:
            frame_hwc = f["frames"][frame_idx]          # (224,224,3) uint8
            pan_vel   = float(f["pan_vel"][frame_idx])
            tilt_vel  = float(f["tilt_vel"][frame_idx])
            cmd       = f["cmd"][()].decode("utf-8") if isinstance(f["cmd"][()], bytes) \
                        else str(f["cmd"][()])
            label_key = f["label_key"][()].decode("utf-8") if isinstance(f["label_key"][()], bytes) \
                        else str(f["label_key"][()])

        # Normalise and convert to (3,224,224) float32 tensor
        frame_f32 = frame_hwc.astype(np.float32) / 255.0       # (H,W,3) in [0,1]
        frame_f32 = (frame_f32 - IMAGENET_MEAN) / IMAGENET_STD  # ImageNet normalise
        frame_t   = torch.from_numpy(frame_f32.transpose(2, 0, 1))  # CHW

        if self.transform is not None:
            frame_t = self.transform(frame_t)

        return {
            "frame":     frame_t,                              # (3,224,224) float32
            "cmd":       cmd,                                  # str
            "label_key": label_key,                            # str
            "pan_vel":   torch.tensor(pan_vel,  dtype=torch.float32),
            "tilt_vel":  torch.tensor(tilt_vel, dtype=torch.float32),
        }

    # ------------------------------------------------------------------
    # collate_fn — tokenises language commands for the model
    # ------------------------------------------------------------------

    @staticmethod
    def collate_fn(batch: list[dict], tokenizer=None) -> dict:
        """Collate a list of samples into a batch dict.

        If tokenizer is provided (a CLIP tokenizer), input_ids and
        attention_mask are included in the returned dict.
        Pass this as a lambda to DataLoader:
            collate = lambda b: OcelotDataset.collate_fn(b, tokenizer)
            DataLoader(ds, collate_fn=collate)
        """
        frames    = torch.stack([s["frame"]    for s in batch])   # (B,3,224,224)
        pan_vels  = torch.stack([s["pan_vel"]  for s in batch])   # (B,)
        tilt_vels = torch.stack([s["tilt_vel"] for s in batch])   # (B,)
        targets   = torch.stack([pan_vels, tilt_vels], dim=1)     # (B,2)
        cmds      = [s["cmd"]       for s in batch]
        label_keys = [s["label_key"] for s in batch]

        result = {
            "frames":     frames,
            "targets":    targets,
            "cmds":       cmds,
            "label_keys": label_keys,
        }

        if tokenizer is not None:
            tokens = tokenizer(
                cmds,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            result["input_ids"]      = tokens["input_ids"]
            result["attention_mask"] = tokens["attention_mask"]

        return result
