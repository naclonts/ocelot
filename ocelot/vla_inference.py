#!/usr/bin/env python3
"""Shared VLA inference helpers for local and remote execution paths."""

from __future__ import annotations

import ctypes
import json
import os
import time
from pathlib import Path

import numpy as np

# ImageNet normalisation — must match train/dataset.py
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preload_cuda_runtime_libs() -> None:
    """Load CUDA/cuDNN libs from common local locations before importing ONNX Runtime."""
    repo_root = Path(__file__).resolve().parents[1]
    search_roots: list[Path] = [
        Path(root) for root in os.environ.get("LD_LIBRARY_PATH", "").split(":") if root
    ]

    venv_nvidia_root = repo_root / ".venv" / "lib" / "python3.11" / "site-packages" / "nvidia"
    if venv_nvidia_root.exists():
        search_roots.extend(sorted(venv_nvidia_root.glob("*/lib")))

    search_roots.extend(
        path
        for path in (
            Path("/usr/local/cuda/lib64"),
            Path("/usr/local/cuda-13.1/targets/x86_64-linux/lib"),
            Path("/usr/local/lib/ollama"),
        )
        if path.exists()
    )

    required_libs = (
        "libcudart.so.12",
        "libnvrtc.so.12",
        "libcublas.so.12",
        "libcublasLt.so.12",
        "libcurand.so.10",
        "libcufft.so.11",
        "libcufftw.so.11",
        "libcudnn.so.9",
        "libcudnn_ops.so.9",
        "libcudnn_cnn.so.9",
        "libcudnn_adv.so.9",
        "libcudnn_graph.so.9",
        "libcudnn_heuristic.so.9",
        "libcudnn_engines_runtime_compiled.so.9",
        "libcudnn_engines_precompiled.so.9",
    )

    seen: set[Path] = set()
    for lib_name in required_libs:
        for root in search_roots:
            candidate = root / lib_name
            if not candidate.exists() or candidate in seen:
                continue
            ctypes.CDLL(str(candidate), mode=os.RTLD_GLOBAL)
            seen.add(candidate)
            break


def preprocess_bgr(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 HxWx3 → float32 (1, 3, 224, 224) ImageNet-normalised."""
    import cv2

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (224, 224):
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    frame = rgb.astype(np.float32) / 255.0
    frame = (frame - _IMAGENET_MEAN) / _IMAGENET_STD
    frame = frame.transpose(2, 0, 1)[np.newaxis, ...]
    return np.ascontiguousarray(frame)


def find_best_command(requested: str, cache: dict[str, dict]) -> str:
    """Return the closest command in the cache (exact match first, then substring)."""
    if requested in cache:
        return requested
    req_lower = requested.lower().strip()
    for key in cache:
        if req_lower in key.lower() or key.lower() in req_lower:
            return key
    for fallback in (
        "track the face",
        "look at the person",
        "follow the person",
        "keep the face centered",
        "no face visible",
    ):
        if fallback in cache:
            return fallback
    return next(iter(cache))


class VLAInferenceEngine:
    """Thin ONNX wrapper shared by the Pi-local node and the remote HTTP server."""

    def __init__(
        self,
        checkpoint: str | Path,
        token_cache: str | Path | None = None,
        providers: list[str] | None = None,
    ) -> None:
        _preload_cuda_runtime_libs()
        import onnxruntime as ort

        self.checkpoint = Path(checkpoint)
        if token_cache is None:
            self.token_cache_path = self.checkpoint.with_name(
                self.checkpoint.stem + "_tokens.json"
            )
        else:
            self.token_cache_path = Path(token_cache)

        if not self.token_cache_path.exists():
            raise FileNotFoundError(
                f"Token cache not found: {self.token_cache_path}"
            )

        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(self.checkpoint), providers=providers)
        self.provider = self.session.get_providers()[0]
        self.token_cache: dict[str, dict] = json.loads(self.token_cache_path.read_text())
        self._token_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def resolve_command(self, requested: str) -> str:
        return find_best_command(requested, self.token_cache)

    def _command_arrays(self, command: str) -> tuple[str, np.ndarray, np.ndarray]:
        actual = self.resolve_command(command)
        arrays = self._token_arrays.get(actual)
        if arrays is None:
            tokens = self.token_cache[actual]
            arrays = (
                np.array([tokens["input_ids"]], dtype=np.int64),
                np.array([tokens["attention_mask"]], dtype=np.int64),
            )
            self._token_arrays[actual] = arrays
        return actual, arrays[0], arrays[1]

    def predict_bgr(self, bgr: np.ndarray, command: str) -> dict[str, float | str]:
        actual, input_ids, attention_mask = self._command_arrays(command)
        frame = preprocess_bgr(bgr)

        start = time.perf_counter()
        actions = self.session.run(
            ["actions"],
            {
                "frames": frame,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )[0]
        latency_ms = (time.perf_counter() - start) * 1000.0

        return {
            "command": actual,
            "pan_vel": float(actions[0, 0]),
            "tilt_vel": float(actions[0, 1]),
            "inference_latency_ms": latency_ms,
        }
