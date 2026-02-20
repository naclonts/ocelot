#!/usr/bin/env python3
"""Camera capture worker — must run under Python 3.11 for picamera2/libcamera.

Spawned as a subprocess by camera_node.py. Captures frames and writes them
to stdout as length-prefixed raw RGB bytes (4-byte big-endian length + data).
"""
import signal
import struct
import sys

from picamera2 import Picamera2


def main():
    width = int(sys.argv[1]) if len(sys.argv) > 1 else 640
    height = int(sys.argv[2]) if len(sys.argv) > 2 else 480

    cam = Picamera2()
    config = cam.create_video_configuration(
        main={'size': (width, height), 'format': 'RGB888'}
    )
    cam.configure(config)
    cam.start()

    out = sys.stdout.buffer

    def shutdown(sig, _frame):
        cam.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    while True:
        data = cam.capture_array().tobytes()
        out.write(struct.pack('>I', len(data)) + data)
        out.flush()


if __name__ == '__main__':
    main()
