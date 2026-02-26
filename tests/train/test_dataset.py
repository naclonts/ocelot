"""tests/train/test_dataset.py — unit tests for OcelotDataset.

Uses a tiny synthetic HDF5 fixture (3 episodes × 10 frames) instead of the
real dataset so this suite runs without a GPU, ROS, or Gazebo.
"""

from __future__ import annotations

import struct
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from train.dataset import IMAGENET_MEAN, IMAGENET_STD, OcelotDataset

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

N_EPISODES   = 3
N_FRAMES     = 10
IMG_H, IMG_W = 224, 224

LABEL_KEYS = ["basic_track", "slow_follow", "fast_follow"]
CMDS       = ["track the face", "follow slowly", "follow quickly"]


def _make_fixture(tmp_path: Path) -> Path:
    """Write a minimal sharded dataset under tmp_path and return dataset_dir."""
    dataset_dir = tmp_path / "dataset"

    # Single shard
    shard = dataset_dir / "shard_0"
    episodes_dir = shard / "episodes"
    episodes_dir.mkdir(parents=True)

    ep_ids = [f"{i:06d}" for i in range(N_EPISODES)]

    for i, ep_id in enumerate(ep_ids):
        path = episodes_dir / f"ep_{ep_id}.h5"
        with h5py.File(path, "w") as f:
            # Deterministic pixel values so we can check normalisation
            frames = np.full((N_FRAMES, IMG_H, IMG_W, 3), fill_value=i * 80,
                             dtype=np.uint8)
            f.create_dataset("frames", data=frames, compression="gzip", compression_opts=1)
            f.create_dataset("pan_vel",  data=np.linspace(-1.0, 1.0, N_FRAMES, dtype=np.float32))
            f.create_dataset("tilt_vel", data=np.linspace( 0.5,-0.5, N_FRAMES, dtype=np.float32))
            f["cmd"]       = CMDS[i]
            f["label_key"] = LABEL_KEYS[i]

    # Write split files — episodes 0,1 → train; episode 2 → val and test
    (shard / "train.txt").write_text("\n".join(ep_ids[:2]))
    (shard / "val.txt").write_text(ep_ids[2])
    (shard / "test.txt").write_text(ep_ids[2])

    return dataset_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOcelotDataset:

    def test_len_train(self, tmp_path):
        ds = OcelotDataset("train", _make_fixture(tmp_path))
        # 2 train episodes × 10 frames
        assert len(ds) == 2 * N_FRAMES

    def test_len_val(self, tmp_path):
        ds = OcelotDataset("val", _make_fixture(tmp_path))
        assert len(ds) == 1 * N_FRAMES

    def test_getitem_shapes(self, tmp_path):
        ds = OcelotDataset("train", _make_fixture(tmp_path))
        sample = ds[0]

        assert sample["frame"].shape == (3, IMG_H, IMG_W)
        assert sample["frame"].dtype == torch.float32
        assert isinstance(sample["pan_vel"],  torch.Tensor)
        assert isinstance(sample["tilt_vel"], torch.Tensor)
        assert sample["pan_vel"].shape  == ()
        assert sample["tilt_vel"].shape == ()
        assert isinstance(sample["cmd"],       str)
        assert isinstance(sample["label_key"], str)

    def test_imagenet_normalisation_applied(self, tmp_path):
        """Pixel value ~0 → after norm should be ≈ -mean/std (not ≈ 0)."""
        ds = OcelotDataset("train", _make_fixture(tmp_path))
        # episode 0 has all-zero pixels (fill_value=0)
        sample = ds[0]
        frame = sample["frame"]  # (3,224,224)

        # Channel 0 (R): expected = (0/255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        expected_r = (0.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        np.testing.assert_allclose(
            frame[0, 0, 0].item(), expected_r, atol=1e-5,
            err_msg="ImageNet normalisation not applied to channel 0"
        )

    def test_normalisation_not_zero_mean(self, tmp_path):
        """Sanity: after normalisation, channel mean should not be close to 0
        for an all-zero image (it should be ≈ -mean/std ≈ -2.1)."""
        ds = OcelotDataset("train", _make_fixture(tmp_path))
        sample = ds[0]
        # Episode 0 pixels are 0; mean across spatial dims should be large-magnitude negative
        assert sample["frame"][0].mean().item() < -1.5

    def test_velocities_match_numpy(self, tmp_path):
        """Pan/tilt velocities match what was written to HDF5."""
        ds = OcelotDataset("train", _make_fixture(tmp_path))
        expected_pan_vels = np.linspace(-1.0, 1.0, N_FRAMES, dtype=np.float32)

        for frame_idx in range(N_FRAMES):
            sample = ds[frame_idx]
            np.testing.assert_allclose(
                sample["pan_vel"].item(), expected_pan_vels[frame_idx], atol=1e-6
            )

    def test_cmd_and_label_key_strings(self, tmp_path):
        ds = OcelotDataset("train", _make_fixture(tmp_path))
        # Both episodes in train split; check both cmds appear
        cmds_seen = {ds[i * N_FRAMES]["cmd"] for i in range(2)}
        assert cmds_seen == {CMDS[0], CMDS[1]}

    def test_train_val_splits_disjoint(self, tmp_path):
        """No episode should appear in both train and val splits."""
        fixture = _make_fixture(tmp_path)
        train_ds = OcelotDataset("train", fixture)
        val_ds   = OcelotDataset("val",   fixture)

        train_paths = {h5p for h5p, _ in train_ds._index}
        val_paths   = {h5p for h5p, _ in val_ds._index}
        assert train_paths.isdisjoint(val_paths), \
            f"Train/val overlap: {train_paths & val_paths}"

    def test_invalid_split_raises(self, tmp_path):
        with pytest.raises(ValueError, match="split must be"):
            OcelotDataset("bogus", _make_fixture(tmp_path))

    def test_missing_dataset_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            OcelotDataset("train", tmp_path / "nonexistent")

    def test_dataloader_one_epoch(self, tmp_path):
        """DataLoader should iterate the full epoch without error."""
        ds = OcelotDataset("train", _make_fixture(tmp_path))
        loader = DataLoader(ds, batch_size=4, num_workers=0,
                            collate_fn=lambda b: OcelotDataset.collate_fn(b))
        total = 0
        for batch in loader:
            assert batch["frames"].shape[1:] == (3, IMG_H, IMG_W)
            assert batch["targets"].shape[1] == 2
            total += batch["frames"].shape[0]
        assert total == len(ds)

    def test_collate_fn_shapes(self, tmp_path):
        ds = OcelotDataset("train", _make_fixture(tmp_path))
        batch = [ds[i] for i in range(4)]
        out = OcelotDataset.collate_fn(batch)

        assert out["frames"].shape  == (4, 3, IMG_H, IMG_W)
        assert out["targets"].shape == (4, 2)
        assert len(out["cmds"]) == 4
        assert len(out["label_keys"]) == 4

    def test_multi_shard(self, tmp_path):
        """Two shards should be concatenated correctly."""
        dataset_dir = tmp_path / "multi"

        for shard_idx in range(2):
            shard = dataset_dir / f"shard_{shard_idx}"
            ep_dir = shard / "episodes"
            ep_dir.mkdir(parents=True)

            ep_id = f"{shard_idx:06d}"
            with h5py.File(ep_dir / f"ep_{ep_id}.h5", "w") as f:
                f.create_dataset("frames",
                                 data=np.zeros((N_FRAMES, IMG_H, IMG_W, 3), dtype=np.uint8))
                f.create_dataset("pan_vel",  data=np.zeros(N_FRAMES, dtype=np.float32))
                f.create_dataset("tilt_vel", data=np.zeros(N_FRAMES, dtype=np.float32))
                f["cmd"]       = "track the face"
                f["label_key"] = "basic_track"

            (shard / "train.txt").write_text(ep_id)
            (shard / "val.txt").write_text(ep_id)
            (shard / "test.txt").write_text(ep_id)

        ds = OcelotDataset("train", dataset_dir)
        assert len(ds) == 2 * N_FRAMES
