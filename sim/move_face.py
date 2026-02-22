#!/usr/bin/env python3
"""Slowly oscillate the face_0 billboard in the Gazebo sim (Step 4 parity check).

Moves the billboard sinusoidally in Y (pan axis) while holding X=2.0, Z=0.5.
This keeps the face billboard continuously off-centre so the tracker's steady
state can be observed rather than just a transient centering manoeuvre.

Usage (inside the sim container, after the sim is running):
    python3 sim/move_face.py

Optional args:
    --amp   float   Amplitude in metres (default 0.6 — ±0.6 m in Y)
    --period float  Oscillation period in seconds (default 12.0)
    --z     float   Billboard height in metres (default 0.5)

The world name is tracker_world (must match the world name in tracker_world.sdf).
"""

import argparse
import math
import subprocess
import time


WORLD = 'tracker_world'
MODEL = 'face_0'
X = 2.0           # fixed distance in front of robot (metres)


def set_pose(x: float, y: float, z: float) -> bool:
    """Call gz service to teleport the face billboard to (x, y, z)."""
    req = f'name: "{MODEL}" position: {{x: {x:.4f}, y: {y:.4f}, z: {z:.4f}}}'
    result = subprocess.run(
        [
            'gz', 'service',
            '-s', f'/world/{WORLD}/set_pose',
            '--reqtype', 'gz.msgs.Pose',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '500',
            '--req', req,
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description='Oscillate face billboard for parity check')
    parser.add_argument('--amp',    type=float, default=0.6,
                        help='Y oscillation amplitude in metres (default 0.6)')
    parser.add_argument('--period', type=float, default=12.0,
                        help='Oscillation period in seconds (default 12.0)')
    parser.add_argument('--z',      type=float, default=0.5,
                        help='Billboard height in metres (default 0.5)')
    args = parser.parse_args()

    print(f'Moving {MODEL} in world {WORLD}: amp={args.amp} m, period={args.period} s')
    print('Press Ctrl-C to stop.')

    t0 = time.monotonic()
    hz = 10.0         # update rate (10 Hz is plenty for a slow sinusoid)
    dt = 1.0 / hz

    try:
        while True:
            t = time.monotonic() - t0
            y = args.amp * math.sin(2 * math.pi * t / args.period)
            ok = set_pose(X, y, args.z)
            if not ok:
                print(f'  [warn] set_pose failed at t={t:.1f}s — is Gazebo running?')
            else:
                print(f'  face_0 → y={y:+.3f} m  (t={t:.1f}s)', end='\r')
            time.sleep(dt)
    except KeyboardInterrupt:
        print('\nStopped.')
        # Park face at a slightly off-centre position so tracker stays active
        set_pose(X, 0.3, args.z)


if __name__ == '__main__':
    main()
