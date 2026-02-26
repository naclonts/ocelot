#!/usr/bin/env python3
"""train/verify_env.py — sanity-check the training environment.

Run manually before starting Phase 3 training. Not part of the pytest suite.

    source .venv/bin/activate
    python3 train/verify_env.py
"""

import sys

REQUIRED_VRAM_GB    = 6.0   # RTX 2070 has 8 GB; leave headroom for OS/other processes
MIN_COMPUTE_MAJOR   = 7     # sm_70 = Volta; RTX 2070 is sm_75 (Turing)


def check(label: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    line = f"  [{status}] {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    return passed


def main() -> int:
    failures = 0

    # ------------------------------------------------------------------
    # Python version
    # ------------------------------------------------------------------
    print("\nPython")
    ok = sys.version_info >= (3, 10)
    if not check("version ≥ 3.10", ok, f"{sys.version.split()[0]}"):
        failures += 1

    # ------------------------------------------------------------------
    # PyTorch + CUDA
    # ------------------------------------------------------------------
    print("\nPyTorch")
    try:
        import torch
        check("import torch", True, torch.__version__)

        cuda_available = torch.cuda.is_available()
        if not check("cuda available", cuda_available):
            failures += 1
            print("  (remaining GPU checks skipped)")
        else:
            name   = torch.cuda.get_device_name(0)
            vram   = torch.cuda.get_device_properties(0).total_memory / 1024**3
            props  = torch.cuda.get_device_properties(0)
            major, minor = props.major, props.minor

            check("device name", True, name)
            if not check(f"VRAM ≥ {REQUIRED_VRAM_GB:.0f} GB", vram >= REQUIRED_VRAM_GB,
                         f"{vram:.1f} GB"):
                failures += 1
            # Compute capability: sm_75 → major=7. cu121 wheels need ≥ sm_37.
            if not check(f"compute capability ≥ sm_{MIN_COMPUTE_MAJOR}0",
                         major >= MIN_COMPUTE_MAJOR, f"sm_{major}{minor}"):
                failures += 1
            # CUDA toolkit version (from torch.version.cuda, e.g. "12.1")
            cuda_ver = torch.version.cuda or ""
            cuda_major = int(cuda_ver.split(".")[0]) if cuda_ver else 0
            if not check("CUDA toolkit ≥ 12", cuda_major >= 12, f"CUDA {cuda_ver}"):
                failures += 1

            # Quick tensor round-trip on GPU
            try:
                x = torch.ones(4, 4, device="cuda")
                result = (x @ x).sum().item()
                check("GPU tensor op", result == 64.0, f"matmul sum = {result}")
            except RuntimeError as e:
                msg = str(e).splitlines()[0]
                check("GPU tensor op", False, f"{msg} (device may be busy — retry)")
                failures += 1

            # GPU load test: sustained matmuls + nvidia-smi utilization sample
            try:
                import subprocess
                import threading
                import time

                LOAD_SECONDS = 5.0
                SIZE = 4096

                util_samples: list[int] = []
                stop = threading.Event()

                def _sample():
                    while not stop.is_set():
                        try:
                            raw = subprocess.check_output(
                                ["nvidia-smi",
                                 "--query-gpu=utilization.gpu",
                                 "--format=csv,noheader,nounits"],
                                timeout=1,
                            ).decode().strip().splitlines()[0]
                            util_samples.append(int(raw))
                        except Exception:
                            pass
                        time.sleep(0.1)

                sampler = threading.Thread(target=_sample, daemon=True)
                sampler.start()

                a = torch.randn(SIZE, SIZE, device="cuda")
                t0 = time.time()
                while time.time() - t0 < LOAD_SECONDS:
                    a = a @ a
                    a = a / a.norm()   # keep values finite
                torch.cuda.synchronize()

                stop.set()
                sampler.join(timeout=1.0)

                peak = max(util_samples) if util_samples else 0
                if not check("GPU load test", peak > 0, f"peak utilization = {peak}%"):
                    failures += 1
            except Exception as e:
                check("GPU load test", False, str(e).splitlines()[0])
                failures += 1

    except ImportError as e:
        check("import torch", False, str(e))
        failures += 1

    # ------------------------------------------------------------------
    # Required packages
    # ------------------------------------------------------------------
    print("\nPackages")
    packages = [
        ("transformers", None),
        ("mlflow",       None),
        ("onnx",         None),
        ("onnxruntime",  None),
        ("h5py",         None),
        ("tqdm",         None),
        ("numpy",        None),
    ]
    for pkg, _ in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            check(f"import {pkg}", True, ver)
        except ImportError as e:
            check(f"import {pkg}", False, str(e))
            failures += 1

    # ------------------------------------------------------------------
    # HuggingFace model download (offline check — just confirm cache or network)
    # ------------------------------------------------------------------
    print("\nHuggingFace models (config fetch only)")
    models = [
        ("facebook/dinov2-small",        "DINOv2-small visual encoder"),
        ("openai/clip-vit-base-patch32", "CLIP text encoder"),
    ]
    for model_id, label in models:
        try:
            from transformers import AutoConfig
            AutoConfig.from_pretrained(model_id)
            check(label, True, model_id)
        except Exception as e:
            check(label, False, str(e))
            failures += 1

    # ------------------------------------------------------------------
    # Dataset smoke-check (optional — skip gracefully if not present)
    # ------------------------------------------------------------------
    print("\nDataset")
    from pathlib import Path
    dataset_dir = Path("sim/dataset")
    if not dataset_dir.exists():
        print("  [SKIP] sim/dataset not found — run from project root or collect data first")
    else:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from train.dataset import OcelotDataset
            ds = OcelotDataset("train", dataset_dir)
            check("OcelotDataset loads", True, f"{len(ds):,} train frames")
        except Exception as e:
            check("OcelotDataset loads", False, str(e))
            failures += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    if failures == 0:
        print("All checks passed — environment is ready for Phase 3 training.")
    else:
        print(f"{failures} check(s) FAILED — fix the issues above before training.")

    return failures


if __name__ == "__main__":
    sys.exit(main())
