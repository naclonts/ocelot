SIM_COMPOSE     = deploy/docker/docker-compose.sim.yml
SIM_GPU_COMPOSE = deploy/docker/docker-compose.sim.gpu.yml

# Detect installed NVIDIA driver version so the GPU compose file can bind-mount
# libnvidia-glsi.so.<version> — a dep of libEGL_nvidia.so.0 that nvidia-container-cli
# omits from its mounts because it lives in /usr/lib/x86_64-linux-gnu/ (top-level)
# rather than the nvidia/current/ subdirectory the toolkit scans.
NVIDIA_DRIVER_VERSION := $(shell nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' \n')
export NVIDIA_DRIVER_VERSION

# Inner command run inside the sim container for all launch targets
define SIM_CMD
source /opt/ros/jazzy/setup.bash && cd /ws && \
colcon build --symlink-install --packages-select ocelot --event-handlers console_direct- && \
source /ws/install/setup.bash && \
ros2 launch ocelot sim_launch.py headless:=$(HEADLESS)
endef

VLA_ONNX ?= runs/sweep-v0.0.2-1500-ep/lr1e-4_l2/best.onnx

.PHONY: sim-build sim sim-gui sim-gpu sim-vla sim-vla-eval sim-shell sim-xauth faces backgrounds dvc-push dvc-pull use-model lint hook-install help

help:
	@grep -E '^##' Makefile | sed 's/## //'

## lint        run Ruff lint checks
lint:
	./.venv/bin/python -m ruff check ocelot sim tests train

## hook-install install the tracked git pre-commit hook for auto-fixing staged Python files
hook-install:
	git config core.hooksPath .githooks

## sim-build   build the sim Docker image
sim-build:
	docker compose -f $(SIM_COMPOSE) build

## sim         headless sim — no GUI, fast, works on any machine
sim: HEADLESS=true
sim:
	docker compose -f $(SIM_COMPOSE) run --rm sim bash -c "$(SIM_CMD)"

## sim-gui     sim with Gazebo GUI — software rendering (no GPU required)
sim-gui: HEADLESS=false
sim-gui:
	docker compose -f $(SIM_COMPOSE) run --rm sim bash -c "$(SIM_CMD)"

## sim-gpu     sim with Gazebo GUI — GPU accelerated (requires NVIDIA runtime)
sim-gpu: HEADLESS=false
sim-gpu:
	docker compose -f $(SIM_COMPOSE) -f $(SIM_GPU_COMPOSE) run --rm sim bash -c "$(SIM_CMD)"

## sim-vla     run trained VLA model in sim (GPU). Usage: make sim-vla VLA_ONNX=runs/v0.1/best.onnx
sim-vla:
	docker compose -f $(SIM_COMPOSE) -f $(SIM_GPU_COMPOSE) run --rm sim bash -c " \
	  source /opt/ros/jazzy/setup.bash && cd /ws && \
	  colcon build --symlink-install --packages-select ocelot --event-handlers console_direct- && \
	  source /ws/install/setup.bash && \
	  ros2 launch ocelot sim_launch.py use_vla:=true headless:=true \
	    vla_checkpoint:=/ws/src/ocelot/$(VLA_ONNX)"

SCENARIO_SEED ?= 0
N_SCENARIOS   ?= 5

## sim-vla-eval  eval VLA against N training-distribution scenarios. Usage: make sim-vla-eval VLA_ONNX=runs/v0.1/best.onnx [SCENARIO_SEED=0] [N_SCENARIOS=5]
sim-vla-eval:
	docker compose -f $(SIM_COMPOSE) -f $(SIM_GPU_COMPOSE) run --rm sim bash -c " \
	  source /opt/ros/jazzy/setup.bash && cd /ws && \
	  colcon build --symlink-install --packages-select ocelot --event-handlers console_direct- && \
	  source /ws/install/setup.bash && \
	  ros2 launch ocelot sim_launch.py use_vla:=true world:=scenario_world headless:=true \
	    vla_checkpoint:=/ws/src/ocelot/$(VLA_ONNX) & \
	  python3 /ws/src/ocelot/sim/eval_vla_live.py \
	    --seed $(SCENARIO_SEED) --n-scenarios $(N_SCENARIOS)"

## sim-shell   open an interactive shell in a fresh sim container
sim-shell:
	docker compose -f $(SIM_COMPOSE) run --rm sim bash

## sim-xauth   one-time X11 auth setup (re-run if display session changes)
sim-xauth:
	rm -rf /tmp/.docker.xauth
	touch /tmp/.docker.xauth
	xauth nlist $$DISPLAY | sed -e 's/^..../ffff/' | xauth -f /tmp/.docker.xauth nmerge -

## faces      generate face descriptions + images, then track with DVC and push to S3
faces:
	python3 sim/scenario_generator/face_descriptions.py --count 100 --seed 7 --out sim/scenario_generator/
	python3 sim/scenario_generator/generate_face_images.py --input sim/scenario_generator/face_descriptions.json --out sim/assets/faces/
	dvc add sim/assets
	dvc push

## backgrounds  generate plain background textures (6 solid-color PNGs)
backgrounds:
	python3 sim/scenario_generator/generate_backgrounds.py \
	  --out sim/assets/backgrounds/
	dvc add sim/assets
	dvc push

## use-model  quantize + activate a model for robot deployment. Usage: make use-model RUN=runs/v0.1.1-single-face
RUN ?= runs/v0.1.1-single-face
use-model:
	@if [ ! -f $(RUN)/best_int8.onnx ]; then \
	  echo "Quantizing $(RUN)/best.onnx → $(RUN)/best_int8.onnx ..."; \
	  docker compose -f deploy/docker/docker-compose.yml run --rm --no-deps \
	    --entrypoint "" ocelot \
	    python3 -c "from onnxruntime.quantization import quantize_dynamic, QuantType; \
	      quantize_dynamic('/ws/src/ocelot/$(RUN)/best.onnx', '/ws/src/ocelot/$(RUN)/best_int8.onnx', weight_type=QuantType.QInt8)"; \
	  echo "Quantization done."; \
	else \
	  echo "$(RUN)/best_int8.onnx already exists, skipping quantization."; \
	fi
	ln -sf $(shell realpath --relative-to=models $(RUN)/best_int8.onnx) models/active.onnx
	ln -sf $(shell realpath --relative-to=models $(RUN)/best_tokens.json) models/active_tokens.json
	@echo "Active model → $(RUN)/best_int8.onnx"

## dvc-push   push all DVC-tracked data to S3
dvc-push:
	dvc push

## dvc-pull   fetch all DVC-tracked data from S3
dvc-pull:
	dvc pull

.DEFAULT_GOAL := help
