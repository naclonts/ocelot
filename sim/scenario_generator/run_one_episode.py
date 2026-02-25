#!/usr/bin/env python3
"""
Smoke-test script for EpisodeRunner (Step G acceptance test).

Requires a running Gazebo instance with scenario_world loaded:
    ros2 launch ocelot sim_launch.py world:=scenario_world headless:=true use_oracle:=true

Run inside the sim container:
    python3 /ws/src/ocelot/sim/scenario_generator/run_one_episode.py --seed 42 --duration 10

Exit 0 = episode completed without error.  Exit 1 = error.

To run 10 sequential episodes and verify no entity leaks:
    for i in $(seq 0 9); do
        python3 /ws/src/ocelot/sim/scenario_generator/run_one_episode.py --seed $i --duration 5
    done
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root on sys.path so `sim.*` imports work regardless of cwd.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sim.scenario_generator.scenario import ScenarioGenerator
from sim.scenario_generator.gazebo_bridge import GazeboBridge
from sim.scenario_generator.episode_runner import EpisodeRunner

FACES_DIR = _PROJECT_ROOT / "sim" / "scenario_generator"
BACKGROUNDS_DIR = _PROJECT_ROOT / "sim" / "assets" / "backgrounds"
WORLD = "scenario_world"
TICK_RATE_HZ = 10


def _print_config(config) -> None:
    print("── Scenario Config ─────────────────────────────────────────")
    print(f"  scenario_id : {config.scenario_id}")
    print(f"  seed        : {config.seed}")
    print(f"  label       : [{config.label_key}] {config.language_label!r}")
    print(f"  faces       : {len(config.faces)}")
    for i, f in enumerate(config.faces):
        target = "  ← target" if i == config.target_face_idx else ""
        print(
            f"    [{i}] {f.face_id:<12}  "
            f"pos=({f.initial_x:.2f}, {f.initial_y:.2f}, {f.initial_z:.2f})  "
            f"motion={f.motion}  speed={f.speed:.2f}{target}"
        )
    print(f"  background  : {config.background_id}")
    print(
        f"  lighting    : az={config.lighting_azimuth_deg:.1f}°  "
        f"el={config.lighting_elevation_deg:.1f}°  "
        f"intensity={config.lighting_intensity:.2f}"
    )
    print(f"  distractors : {config.distractor_count}")
    for i, d in enumerate(config.distractors):
        print(
            f"    [{i}] {d.shape:<6}  "
            f"pos=({d.initial_x:.2f}, {d.initial_y:.2f}, {d.initial_z:.2f})  "
            f"speed={d.speed:.3f}"
        )
    print(
        f"  cam noise   : sigma={config.camera_noise_sigma:.4f}  "
        f"brightness_offset={config.camera_brightness_offset:+.1f}"
    )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a single randomized episode in Gazebo (Step G smoke test)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Scenario seed (default: 42)")
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Episode duration in seconds (default: 10)"
    )
    parser.add_argument(
        "--world", default=WORLD,
        help=f"Gazebo world name (default: {WORLD})"
    )
    args = parser.parse_args()

    # ── Generate config ────────────────────────────────────────────────
    try:
        generator = ScenarioGenerator(
            faces_dir=FACES_DIR,
            backgrounds_dir=BACKGROUNDS_DIR,
        )
    except ValueError as exc:
        print(f"ERROR: Could not create ScenarioGenerator: {exc}", file=sys.stderr)
        return 1

    config = generator.sample(args.seed)
    _print_config(config)

    # ── Setup ──────────────────────────────────────────────────────────
    bridge = GazeboBridge(world=args.world)
    runner = EpisodeRunner(bridge)

    print("── Setting up episode ──────────────────────────────────────")
    try:
        runner.setup(config)
    except Exception as exc:
        print(f"ERROR during setup: {exc}", file=sys.stderr)
        return 1
    print("   OK")
    print()

    # ── Tick loop ──────────────────────────────────────────────────────
    n_ticks = int(args.duration * TICK_RATE_HZ)
    tick_dt = 1.0 / TICK_RATE_HZ
    print(
        f"── Running {args.duration:.0f} s at {TICK_RATE_HZ} Hz "
        f"({n_ticks} ticks) ────────────────────"
    )

    last_positions: dict = {}
    t = 0.0
    for tick in range(n_ticks):
        t_wall_start = time.monotonic()
        try:
            last_positions = runner.step(t)
        except Exception as exc:
            print(f"ERROR during step {tick} (t={t:.2f}): {exc}", file=sys.stderr)
            runner.teardown()
            return 1
        t += tick_dt

        # Rate-limit to TICK_RATE_HZ (best-effort; gz calls dominate latency).
        elapsed = time.monotonic() - t_wall_start
        spare = tick_dt - elapsed
        if spare > 0:
            time.sleep(spare)

        # Progress dot every second.
        if (tick + 1) % TICK_RATE_HZ == 0:
            elapsed_s = (tick + 1) * tick_dt
            print(f"   t={elapsed_s:5.1f}s", end="  ", flush=True)
            if len(last_positions) > 0:
                target_name = f"face_{config.target_face_idx}"
                if target_name in last_positions:
                    p = last_positions[target_name]
                    print(f"target @ ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})", flush=True)
                else:
                    print(flush=True)

    print()

    # ── Final positions ────────────────────────────────────────────────
    print("── Final positions ─────────────────────────────────────────")
    target_name = f"face_{config.target_face_idx}"
    for name, pos in sorted(last_positions.items()):
        marker = "  ← target" if name == target_name else ""
        print(f"  {name:<20}  ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}){marker}")
    print()

    # ── Teardown ───────────────────────────────────────────────────────
    print("── Teardown ────────────────────────────────────────────────")
    try:
        runner.teardown()
    except Exception as exc:
        print(f"ERROR during teardown: {exc}", file=sys.stderr)
        return 1
    print("   OK")
    print()
    print(f"Episode seed={args.seed} complete ({args.duration:.0f} s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
