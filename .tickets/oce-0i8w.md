---
id: oce-0i8w
status: open
deps: []
links: []
created: 2026-04-03T05:36:33Z
type: task
priority: 2
assignee: Nathan Clonts
tags: [dx, tooling]
---
# Migrate from pip/requirements.txt to uv

Replace pip + requirements*.txt with uv for dependency management on the host and in CI. Evaluate Docker containers case-by-case.

## Current state

Three requirements files, two Dockerfiles, and CI all use raw pip:

| File                    | Purpose                                    | Deps  |
|-------------------------|--------------------------------------------|-------|
| requirements.txt        | Host .venv — DVC, celery, misc tooling     | ~121  |
| requirements-train.txt  | Host .venv — torch (cu121), transformers…  | ~18   |
| requirements-worker.txt | Pi capture worker (Python 3.11 in Docker)  | 4     |
| Dockerfile.robot        | pip3 --break-system-packages (Pi libs)     | ~6    |
| Dockerfile.sim          | pip3 --break-system-packages (onnxrt, CUDA)| ~8    |
| ci.yml                  | pip install in GHA (CPU torch, lint, eval) | mixed |

Pain points: slow pip installs (especially torch+transformers), no lockfile (requirements.txt is a full freeze but fragile to edit), separate install step for PyTorch custom index, `setup.py` is the only project metadata file (no pyproject.toml).

## Proposed changes

### 1. Host dev environment (high value)
- Add `pyproject.toml` with project metadata + dependency groups:
  - `[project.dependencies]` — base deps (currently requirements.txt)
  - `[project.optional-dependencies]` or `[dependency-groups]` — `train` group (currently requirements-train.txt)
  - PyTorch custom index via `[tool.uv.sources]` or `[[tool.uv.index]]`
- Generate `uv.lock` (replaces both requirements.txt files as the lockfile)
- Manage `.venv` via `uv sync` / `uv sync --group train`
- Keep `setup.py` + `setup.cfg` for ROS2/colcon compatibility (colcon needs setup.py); pyproject.toml can coexist

### 2. CI (high value)
- Replace `actions/setup-python` + `pip install` with `astral-sh/setup-uv`
- Use `uv sync --frozen` to install from lockfile (deterministic, fast)
- PyTorch CPU index for CI can be handled via `--index-url` override or a CI-specific UV env var
- Removes the grep-out-torch hack in ci.yml

### 3. Docker containers (low value, optional)
- **Dockerfile.sim**: Could replace `pip3 install --break-system-packages` with `uv pip install --system` for faster image rebuilds during dev. Requires adding uv to the image (`COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv`). Marginal gain — only ~3 pip install lines, runs once at build time.
- **Dockerfile.robot**: Same story, plus the dual-Python (3.12 system + 3.11 deadsnakes) and bind-mounted libcamera make it more complex. Low ROI.
- **requirements-worker.txt**: Only 4 deps, used inside Docker for Python 3.11. Leave as-is or convert last.
- **Recommendation**: Skip Docker migration initially. Revisit if image build times become painful.

## Design

## Key decisions

**pyproject.toml + setup.py coexistence**: ROS2/colcon requires setup.py for ament_python packages. pyproject.toml can coexist — uv reads pyproject.toml for deps, colcon reads setup.py for ROS packaging. The pyproject.toml should NOT declare a [build-system] that conflicts with setuptools (or just use setuptools as the build backend, which is what colcon expects).

**PyTorch custom index**: uv supports `[[tool.uv.index]]` for extra indexes. Pin the cu121 index for the train group. For CI, override with `UV_EXTRA_INDEX_URL` pointing to the CPU wheel index, or use a separate `--index-url` flag on `uv sync`.

**Dependency groups vs optional-dependencies**: Use PEP 735 dependency-groups (`[dependency-groups]`) if uv version supports it (uv ≥ 0.4.27). Otherwise fall back to `[project.optional-dependencies]`. Groups are better because they don't become extras of the package itself.

**Lock strategy**: Commit uv.lock. CI uses `--frozen` to enforce the lockfile matches. Developers run `uv lock` when changing deps.

**Docker**: Use multi-stage `COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv` if adding uv to Docker images. But this is deferred — not part of the initial migration.

## Acceptance Criteria

- [ ] pyproject.toml exists with project metadata and all dependency groups
- [ ] uv.lock is committed and tracks all deps (base + train)
- [ ] `uv sync` creates a working .venv for base deps
- [ ] `uv sync --group train` adds training deps (torch cu121, transformers, etc.)
- [ ] CI uses astral-sh/setup-uv and installs from lockfile
- [ ] CI torch-CPU override works without the grep hack
- [ ] requirements.txt and requirements-train.txt are removed (or kept as generated artifacts if needed for compat)
- [ ] setup.py still works for colcon build
- [ ] All 107 existing tests pass
- [ ] Docker containers still build and run (no regressions even if not migrated)

