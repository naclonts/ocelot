# Sim

## Quickstart

```bash
make sim-build   # build the sim image (once, or after Dockerfile changes)
make sim         # headless — no GUI, fast, works on any machine
make sim-gui     # Gazebo GUI — software rendering (no GPU required)
make sim-gpu     # Gazebo GUI — GPU accelerated (requires NVIDIA runtime)
make sim-xauth   # one-time X11 auth setup (re-run if display session changes)
make sim-shell   # interactive shell in a fresh sim container
```

The colcon build is fast on repeat runs — named volumes (`sim_build`, `sim_install`) cache artifacts between container invocations.

After ~15 seconds the sim is fully up: the face billboard starts oscillating in both pan (Y) and tilt (Z), and the tracker follows it automatically. No manual steps needed.

Verify tracking is working from a second shell in the container:

```bash
ros2 topic echo /joint_states --field position   # pan/tilt positions should change
```

Training-related sim workflows such as `make sim-vla` and `make sim-vla-eval` are documented in [TRAINING.md](TRAINING.md).

## Episode runner (scenario generator)

The episode runner generates randomized scenarios — face textures, background, lighting, motion
patterns, language labels — and drives them in a live Gazebo session. Use it to smoke-test the
scenario generator before running full data collection.

Episodes are deterministic by seed: re-running with the same seed reproduces the same scenario.
These runs use `scenario_world`, the clean simulation world used for collection and scenario tests.

**Prerequisites** (assets must exist before running):

```bash
# Face description JSONs (git-tracked — present after clone if committed)
ls sim/scenario_generator/face_descriptions*.json

# Face PNGs in sim/assets/faces/ and background PNGs in sim/assets/backgrounds/
# Pull from DVC if available:
dvc pull
# Or regenerate locally (backgrounds take seconds; faces require an AI image API):
make backgrounds                          # generates 6 plain-color PNGs; no API required
# make faces                             # generates descriptions + calls image API
```

## Run a single episode

```bash
make sim-shell   # open an interactive shell in a fresh sim container

# Inside the container — build, start sim in background, then run one episode
colcon build --symlink-install --packages-select ocelot --event-handlers console_direct-
source /ws/install/setup.bash
ros2 launch ocelot sim_launch.py world:=scenario_world headless:=true use_oracle:=true &
sleep 15   # wait for Gazebo + ros2_control to finish starting

python3 /ws/src/ocelot/sim/scenario_generator/run_one_episode.py --seed 42 --duration 10
```

Exit code 0 means the episode completed without error. The script prints the full scenario config,
face positions every second, and final positions at teardown.

## Run 10 sequential episodes (entity leak check)

```bash
for i in $(seq 0 9); do
    python3 /ws/src/ocelot/sim/scenario_generator/run_one_episode.py --seed $i --duration 5
done
# After all episodes: verify no leaked entities
gz model --list   # should show only: ground_plane, ocelot
gz light --list   # should be empty
```
