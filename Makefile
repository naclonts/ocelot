SIM_COMPOSE     = deploy/docker/docker-compose.sim.yml
SIM_GPU_COMPOSE = deploy/docker/docker-compose.sim.gpu.yml

# Inner command run inside the sim container for all launch targets
define SIM_CMD
source /opt/ros/jazzy/setup.bash && cd /ws && \
colcon build --symlink-install --packages-select ocelot --event-handlers console_direct- && \
source /ws/install/setup.bash && \
ros2 launch ocelot sim_launch.py headless:=$(HEADLESS)
endef

.PHONY: sim-build sim sim-gui sim-gpu sim-shell sim-xauth help

help:
	@grep -E '^##' Makefile | sed 's/## //'

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

## sim-shell   open an interactive shell in a fresh sim container
sim-shell:
	docker compose -f $(SIM_COMPOSE) run --rm sim bash

## sim-xauth   one-time X11 auth setup (re-run if display session changes)
sim-xauth:
	touch /tmp/.docker.xauth
	xauth nlist $$DISPLAY | sed -e 's/^..../ffff/' | xauth -f /tmp/.docker.xauth nmerge -

.DEFAULT_GOAL := help
