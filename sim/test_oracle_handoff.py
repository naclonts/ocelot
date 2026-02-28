#!/usr/bin/env python3
"""Manual validation: oracle switches targets when faces swap left/right.

Run inside the sim container with scenario_world + oracle already running:

    # Terminal 1 — start the sim:
    ros2 launch ocelot sim_launch.py world:=scenario_world use_oracle:=true headless:=true

    # Terminal 2 — run this script:
    python3 /ws/src/ocelot/sim/test_oracle_handoff.py

    # Terminal 3 — watch the oracle log (grep for target= lines):
    ros2 node info /oracle_node   # confirm it's running
    # The sim launch terminal already shows oracle logs with target= field.

What you should see:
  Phase 1 (0–10 s):  face_0 on LEFT  → oracle log shows  target=face_0
  Phase 2 (10–20 s): swap in progress (faces cross)
  Phase 3 (20–30 s): face_1 on LEFT  → oracle log shows  target=face_1
  Phase 4 (30–40 s): swap back
  Phase 5 (40–50 s): face_0 on LEFT  → oracle log shows  target=face_0
"""

import subprocess
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.scenario_generator.gazebo_bridge import GazeboBridge

# Positions: robot at origin, faces at x=2.5 m, z=0.9 m
X = 2.5
Z = 0.9
Y_LEFT  =  0.6   # world +Y = left in camera frame when pan≈0
Y_RIGHT = -0.6

SWAP_SECS  = 10.0   # time for faces to cross (linear interpolation)
HOLD_SECS  = 10.0   # time to hold each configuration
STEP_SECS  = 0.1    # update interval


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def set_oracle_params(label_key: str, num_faces: int) -> None:
    for param, val in [("label_key", label_key), ("num_faces", str(num_faces))]:
        subprocess.run(
            ["ros2", "param", "set", "/oracle_node", param, val],
            capture_output=True, timeout=3.0,
        )


def main() -> None:
    bridge = GazeboBridge(world="scenario_world")

    # Use a blank face texture if no real assets are available.
    assets_dir = Path(__file__).resolve().parent / "assets" / "faces"
    pngs = sorted(assets_dir.glob("*.png"))
    if len(pngs) < 2:
        print("ERROR: need at least 2 face PNGs in sim/assets/faces/. Aborting.")
        sys.exit(1)

    tex0 = str(pngs[0])
    tex1 = str(pngs[1])

    print(f"Spawning face_0 ({pngs[0].stem}) and face_1 ({pngs[1].stem})")
    bridge.spawn_face("face_0", (X, Y_LEFT,  Z), tex0)
    bridge.spawn_face("face_1", (X, Y_RIGHT, Z), tex1)
    time.sleep(1.0)

    print("Configuring oracle: label_key=multi_left, num_faces=2")
    set_oracle_params("multi_left", 2)
    time.sleep(0.5)

    def phase(label: str, y0: float, y1: float, duration: float) -> None:
        """Hold or animate face positions for `duration` seconds."""
        t_start = time.monotonic()
        while True:
            elapsed = time.monotonic() - t_start
            if elapsed >= duration:
                break
            frac = min(elapsed / duration, 1.0) if duration > 0 else 1.0
            cy0 = lerp(Y_LEFT if y0 > 0 else Y_RIGHT, y0, frac) if duration > 0 else y0
            cy1 = lerp(Y_RIGHT if y1 < 0 else Y_LEFT, y1, frac) if duration > 0 else y1
            bridge.set_pose("face_0", X, y0, Z)
            bridge.set_pose("face_1", X, y1, Z)
            target = "face_0" if y0 >= y1 else "face_1"
            bar = "█" * int(elapsed) + "░" * int(duration - elapsed)
            print(
                f"\r[{bar}] {label:30s}  "
                f"face_0.y={y0:+.2f}  face_1.y={y1:+.2f}  "
                f"expected target → {target}    ",
                end="", flush=True,
            )
            time.sleep(STEP_SECS)
        print()

    try:
        print("\n── Phase 1: face_0 LEFT, face_1 RIGHT (hold 10 s) ─────────────────")
        phase("face_0 LEFT / face_1 RIGHT", Y_LEFT, Y_RIGHT, HOLD_SECS)

        print("\n── Phase 2: swap — face_0 crosses to RIGHT (10 s) ─────────────────")
        steps = int(SWAP_SECS / STEP_SECS)
        t_start = time.monotonic()
        for i in range(steps):
            frac = i / steps
            y0 = lerp(Y_LEFT,  Y_RIGHT, frac)
            y1 = lerp(Y_RIGHT, Y_LEFT,  frac)
            bridge.set_pose("face_0", X, y0, Z)
            bridge.set_pose("face_1", X, y1, Z)
            target = "face_0" if y0 >= y1 else "face_1"
            print(
                f"\r  swapping... face_0.y={y0:+.2f}  face_1.y={y1:+.2f}  "
                f"expected target → {target}    ",
                end="", flush=True,
            )
            time.sleep(STEP_SECS)
        print()

        print("\n── Phase 3: face_1 LEFT, face_0 RIGHT (hold 10 s) ─────────────────")
        phase("face_1 LEFT / face_0 RIGHT", Y_RIGHT, Y_LEFT, HOLD_SECS)

        print("\n── Phase 4: swap back (10 s) ────────────────────────────────────────")
        for i in range(steps):
            frac = i / steps
            y0 = lerp(Y_RIGHT, Y_LEFT,  frac)
            y1 = lerp(Y_LEFT,  Y_RIGHT, frac)
            bridge.set_pose("face_0", X, y0, Z)
            bridge.set_pose("face_1", X, y1, Z)
            target = "face_0" if y0 >= y1 else "face_1"
            print(
                f"\r  swapping back... face_0.y={y0:+.2f}  face_1.y={y1:+.2f}  "
                f"expected target → {target}    ",
                end="", flush=True,
            )
            time.sleep(STEP_SECS)
        print()

        print("\n── Phase 5: face_0 LEFT again (hold 10 s) ──────────────────────────")
        phase("face_0 LEFT / face_1 RIGHT", Y_LEFT, Y_RIGHT, HOLD_SECS)

    finally:
        print("\nCleaning up...")
        bridge.despawn("face_0")
        bridge.despawn("face_1")
        set_oracle_params("track", 1)
        print("Done. Oracle reset to label_key=track, num_faces=1.")


if __name__ == "__main__":
    main()
