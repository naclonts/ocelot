#!/usr/bin/env python3
"""Oscillate the face_0 billboard in Y (pan) and Z (tilt) for sim tracking tests.

Y oscillates sinusoidally at the given period; Z oscillates at 1.7× the period
(incommensurate, so the path doesn't repeat quickly) with a smaller amplitude.
This exercises both pan and tilt joints continuously.

Usage (inside the sim container, after the sim is running):
    python3 sim/move_face.py

Optional args:
    --amp    float  Y oscillation amplitude in metres (default 0.6)
    --z-amp  float  Z oscillation amplitude in metres (default 0.15)
    --period float  Pan oscillation period in seconds (default 12.0)
    --z      float  Billboard centre height in metres (default 0.5)

The world name is tracker_world (must match the world name in tracker_world.sdf).
"""

import argparse
import math
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor


WORLD = 'tracker_world'
MODEL = 'face_0'
X = 2.0           # fixed distance in front of robot (metres)

# Each gz service call blocks until the response arrives.  Running it in a
# thread pool keeps the 20 Hz main loop non-blocking while still keeping the
# ZMQ endpoint alive long enough for Gazebo to deliver the reply.  Without
# this, Popen() exits before Gazebo responds → "Host unreachable" errors.
_pool = ThreadPoolExecutor(max_workers=4)


def set_pose(x: float, y: float, z: float) -> None:
    """Submit a gz service call to the thread pool (non-blocking)."""
    req = f'name: "{MODEL}" position: {{x: {x:.4f}, y: {y:.4f}, z: {z:.4f}}}'
    _pool.submit(
        subprocess.run,
        [
            'gz', 'service',
            '-s', f'/world/{WORLD}/set_pose',
            '--reqtype', 'gz.msgs.Pose',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '200',
            '--req', req,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Oscillate face billboard for parity check')
    parser.add_argument('--amp',    type=float, default=0.6,
                        help='Y oscillation amplitude in metres (default 0.6)')
    parser.add_argument('--z-amp', type=float, default=0.15,
                        help='Z oscillation amplitude in metres (default 0.15)')
    parser.add_argument('--period', type=float, default=12.0,
                        help='Pan oscillation period in seconds (default 12.0)')
    parser.add_argument('--z',      type=float, default=0.5,
                        help='Billboard centre height in metres (default 0.5)')
    args = parser.parse_args()

    print(f'Moving {MODEL} in world {WORLD}: '
          f'Y amp={args.amp} m, Z amp={args.z_amp} m, period={args.period} s')
    print('Press Ctrl-C to stop.')

    t0 = time.monotonic()
    hz = 20.0         # 20 Hz — thread pool keeps the main loop non-blocking
    dt = 1.0 / hz

    try:
        while True:
            t = time.monotonic() - t0
            y = args.amp * math.sin(2 * math.pi * t / args.period)
            # Tilt at 1.7× the pan period — incommensurate so the path doesn't repeat
            z = args.z + args.z_amp * math.sin(2 * math.pi * t / (args.period * 1.7))
            set_pose(X, y, z)
            print(f'  face_0 → y={y:+.3f} z={z:.3f}  (t={t:.1f}s)', end='\r')
            time.sleep(dt)
    except KeyboardInterrupt:
        print('\nStopped.')
        # Park face at a slightly off-centre position so tracker stays active
        set_pose(X, 0.3, args.z)



if __name__ == '__main__':
    main()
