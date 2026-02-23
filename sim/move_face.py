#!/usr/bin/env python3
"""Oscillate the face_0 billboard in Y (pan) and Z (tilt) for sim tracking tests.

Y oscillates sinusoidally at the given period; Z oscillates at 1.7× the period
(incommensurate, so the path doesn't repeat quickly) with a smaller amplitude.
This exercises both pan and tilt joints continuously.

Uses gz.transport13 Python API directly (no subprocesses) so the ZMQ node
stays alive for the lifetime of the script.  Ephemeral gz service subprocesses
trigger gz-transport's BYE/cleanup protocol on exit, causing Gazebo to log
'NodeShared::RecvSrvRequest() error sending response: Host unreachable' on
every call.  A persistent Node avoids that entirely.

Requires: python3-gz-transport13  python3-gz-msgs10
  (both in deploy/docker/Dockerfile.sim — rebuild image if missing)

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
import sys
import time

try:
    from gz.transport13 import Node
    from gz.msgs10.pose_pb2 import Pose
    from gz.msgs10.boolean_pb2 import Boolean
except ImportError as exc:
    sys.exit(
        f'gz-transport Python bindings not available: {exc}\n'
        'Fix: add python3-gz-transport13 and python3-gz-msgs10 to\n'
        '     deploy/docker/Dockerfile.sim and rebuild the image.'
    )


WORLD = 'tracker_world'
MODEL = 'face_0'
X = 2.0           # fixed distance in front of robot (metres)
SERVICE = f'/world/{WORLD}/set_pose'

# Single persistent node — ZMQ connection stays open, no per-call teardown.
_node = Node()

_last_result: bool = True   # track to avoid spamming warnings


def set_pose(x: float, y: float, z: float) -> None:
    """Call the Gazebo set_pose service via the persistent gz-transport node."""
    global _last_result
    req = Pose()
    req.name = MODEL
    req.position.x = x
    req.position.y = y
    req.position.z = z
    result, _rep = _node.request(SERVICE, req, Pose, Boolean, 500)
    if result != _last_result:
        if result:
            print(f'\n[move_face] service OK — pose updates resuming')
        else:
            print(f'\n[move_face] WARNING: {SERVICE} returned False (timeout or service unreachable)')
        _last_result = result


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
    print(f'Service: {SERVICE}')
    print('Press Ctrl-C to stop.')

    # Probe: list available services and warn if ours is absent.
    services = _node.service_list()
    if SERVICE not in services:
        print(f'[move_face] WARNING: {SERVICE!r} not in service list.')
        print(f'  Available services: {services if services else "(none — gz transport not connected?)"}')
        print('  Is the sim running? Will keep trying...')

    t0 = time.monotonic()
    hz = 10.0
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
