#!/usr/bin/env python3
"""Shared VLA inference helpers for local and remote execution paths."""

from __future__ import annotations

import ctypes
import json
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np

# ImageNet normalisation — must match train/dataset.py
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_MAX_LEN = 77  # CLIP max token length — must match training/export.
_CLIP_ID = "openai/clip-vit-base-patch32"


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


def build_tokenize_fn(
    token_cache_path: str | Path | None,
) -> Callable[[str], tuple[np.ndarray, np.ndarray]]:
    """Return a cache-first CLIP tokenization function for runtime commands.

    Cached commands reuse the arrays exported alongside the model. Uncached
    commands are tokenized lazily with the same CLIP tokenizer used in
    training/export and memoized for subsequent calls.
    """
    memo: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    if token_cache_path is not None:
        cache_path = Path(token_cache_path)
        if cache_path.exists():
            raw = json.loads(cache_path.read_text())
            for cmd, entry in raw.items():
                memo[cmd] = (
                    np.array([entry["input_ids"]], dtype=np.int64),
                    np.array([entry["attention_mask"]], dtype=np.int64),
                )

    tokenizer_holder: list[object | None] = [None]

    def _get_tokenizer():
        if tokenizer_holder[0] is None:
            try:
                import transformers
            except ImportError as exc:
                raise RuntimeError(
                    "transformers is required to tokenize uncached commands at runtime."
                ) from exc
            tokenizer_cls = getattr(transformers, "CLIPTokenizerFast", None)
            if tokenizer_cls is None:
                tokenizer_cls = getattr(transformers, "CLIPTokenizer", None)
            if tokenizer_cls is None:
                try:
                    from transformers.models.clip import CLIPTokenizerFast as tokenizer_cls
                except ImportError:
                    try:
                        from transformers.models.clip import CLIPTokenizer as tokenizer_cls
                    except ImportError as exc:
                        raise RuntimeError(
                            "transformers is installed but does not expose a CLIP tokenizer."
                        ) from exc
            tokenizer_holder[0] = tokenizer_cls.from_pretrained(_CLIP_ID)
        return tokenizer_holder[0]

    def tokenize(command: str) -> tuple[np.ndarray, np.ndarray]:
        arrays = memo.get(command)
        if arrays is None:
            tokenizer = _get_tokenizer()
            encoded = tokenizer(
                command,
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=_MAX_LEN,
            )
            arrays = (
                encoded["input_ids"].astype(np.int64),
                encoded["attention_mask"].astype(np.int64),
            )
            memo[command] = arrays
        return arrays

    return tokenize


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
            default_cache = self.checkpoint.with_name(
                self.checkpoint.stem + "_tokens.json"
            )
            fallback_cache = None
            if self.checkpoint.stem.endswith("_int8"):
                fallback_cache = self.checkpoint.with_name(
                    self.checkpoint.stem.removesuffix("_int8") + "_tokens.json"
                )

            if default_cache.exists():
                self.token_cache_path = default_cache
            elif fallback_cache is not None and fallback_cache.exists():
                self.token_cache_path = fallback_cache
            else:
                self.token_cache_path = default_cache
        else:
            self.token_cache_path = Path(token_cache)

        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(self.checkpoint), providers=providers)
        self.provider = self.session.get_providers()[0]
        self._tokenize = build_tokenize_fn(self.token_cache_path)

    def predict_bgr(self, bgr: np.ndarray, command: str) -> dict[str, float | str]:
        input_ids, attention_mask = self._tokenize(command)
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
            "command": command,
            "pan_vel": float(actions[0, 0]),
            "tilt_vel": float(actions[0, 1]),
            "inference_latency_ms": latency_ms,
        }
