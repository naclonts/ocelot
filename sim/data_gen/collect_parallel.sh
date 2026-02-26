#!/usr/bin/env bash
# collect_parallel.sh — spin up N sim stacks and collect episodes in parallel.
#
# Each shard runs its own Gazebo instance, isolated by GZ_PARTITION + ROS_DOMAIN_ID.
# Results land in <output>/shard_N/episodes/ and are merged at the end.
#
# Usage (from the repo root on the host):
#
#   bash sim/data_gen/collect_parallel.sh [--shards N] [--episodes M] [--output PATH]
#
# Defaults: 4 shards, 25000 episodes each, output = sim/dataset

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
N_SHARDS=4
START_SHARD=""   # auto-detected from output dir if not set
N_EPISODES=25000
OUTPUT=/ws/src/ocelot/sim/dataset
COMPOSE_FILE=deploy/docker/docker-compose.sim.yml
IMAGE_TAG=sim   # service name in docker-compose.sim.yml

# ── Arg parse ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --shards)       N_SHARDS=$2;    shift 2 ;;
        --start-shard)  START_SHARD=$2; shift 2 ;;
        --episodes)     N_EPISODES=$2;  shift 2 ;;
        --output)       OUTPUT=$2;      shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Auto-detect start shard ───────────────────────────────────────────────────
# OUTPUT is the container-internal path (/ws/src/ocelot/...).  For shard
# detection we need the host-side equivalent, derived by stripping the
# container mount prefix and prepending the repo root.
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
HOST_OUTPUT="${REPO_ROOT}/sim/dataset"
echo "HOST_OUTPUT directory: $HOST_OUTPUT"

if [[ -z "$START_SHARD" ]]; then
    START_SHARD=0
    if [[ -d "$HOST_OUTPUT" ]]; then
        for d in "$HOST_OUTPUT"/shard_*/; do
            [[ -d "$d" ]] || continue
            n="${d%/}"; n="${n##*shard_}"
            (( n + 1 > START_SHARD )) && START_SHARD=$(( n + 1 ))
        done
    fi
    echo "Auto-detected start shard: ${START_SHARD}"
fi

END_SHARD=$(( START_SHARD + N_SHARDS - 1 ))
echo "Starting ${N_SHARDS} shards × ${N_EPISODES} episodes (shards ${START_SHARD}–${END_SHARD}) → ${OUTPUT}"

# ── Start sim stacks ───────────────────────────────────────────────────────────
for i in $(seq $START_SHARD $END_SHARD); do
    echo "Starting sim stack shard $i ..."
    docker compose -f "$COMPOSE_FILE" run --rm \
        --name "ocelot-sim-$i" \
        -e GZ_PARTITION="$i" \
        -e ROS_DOMAIN_ID="$i" \
        "$IMAGE_TAG" bash -c "
          source /opt/ros/jazzy/setup.bash && cd /ws &&
          colcon build --symlink-install --packages-select ocelot --event-handlers console_direct- &&
          source /ws/install/setup.bash &&
          ros2 launch ocelot sim_launch.py world:=scenario_world use_oracle:=true headless:=true" &
done

echo "Waiting 30 s for sim stacks to finish starting ..."
sleep 30

# ── Start collectors ──────────────────────────────────────────────────────────
for i in $(seq $START_SHARD $END_SHARD); do
    echo "Starting collector shard $i ..."
    docker exec -e ROS_DOMAIN_ID="$i" -e GZ_PARTITION="$i" "ocelot-sim-$i" bash -c "
      source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash &&
      python3 /ws/src/ocelot/sim/data_gen/collect_data.py \
        --n_episodes ${N_EPISODES} \
        --shard ${i} \
        --output ${OUTPUT}" &
done

echo "Collecting — waiting for all shards to finish ..."
wait

# ── Merge (runs on host, not in container) ────────────────────────────────────
# merge_shards.py only needs h5py, which is in .venv.  Running on the host
# avoids the container path / bind-mount confusion that arises when OUTPUT
# differs from the default /ws/src/ocelot/sim/dataset.
echo "Merging shards ..."
source "${REPO_ROOT}/.venv/bin/activate"
python3 "${REPO_ROOT}/sim/data_gen/merge_shards.py" \
    --parent "${HOST_OUTPUT}" \
    --output "${HOST_OUTPUT}/merged"

echo "Done. Dataset at ${HOST_OUTPUT}/merged"
