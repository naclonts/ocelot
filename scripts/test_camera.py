#!/usr/bin/env python3
"""Camera validation script for Pi Camera V2 via picamera2.

Captures a test frame, saves as JPEG, and optionally measures FPS
by capturing a burst of frames.
"""

import time
import os

from picamera2 import Picamera2


def capture_single(cam: Picamera2, output_dir: str = "test_output"):
    """Capture a single frame and save as JPEG."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "test_frame.jpg")

    frame = cam.capture_array()
    height, width = frame.shape[:2]
    print(f"Captured frame: {width}x{height}, dtype={frame.dtype}")

    cam.capture_file(path)
    size_kb = os.path.getsize(path) / 1024
    print(f"Saved to {path} ({size_kb:.1f} KB)")
    return frame


def measure_fps(cam: Picamera2, num_frames: int = 30):
    """Capture a burst of frames and measure actual FPS."""
    print(f"\nCapturing {num_frames} frames to measure FPS...")

    # Warm up
    cam.capture_array()

    start = time.monotonic()
    for _ in range(num_frames):
        cam.capture_array()
    elapsed = time.monotonic() - start

    fps = num_frames / elapsed
    print(f"Captured {num_frames} frames in {elapsed:.2f}s = {fps:.1f} FPS")
    return fps


def main():
    print("=== Ocelot Camera Validation ===\n")

    cam = Picamera2()

    # Configure for 640x480 — matches what we'll use for face tracking
    config = cam.create_still_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    cam.configure(config)
    cam.start()
    time.sleep(1)  # Let auto-exposure settle

    print(f"Camera model: {cam.camera_properties.get('Model', 'unknown')}\n")

    # Single capture
    capture_single(cam)

    # FPS measurement
    measure_fps(cam, num_frames=30)

    cam.stop()
    print("\nCamera stopped. Done.")


if __name__ == "__main__":
    main()
