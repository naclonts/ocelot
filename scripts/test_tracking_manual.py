#!/usr/bin/env python3
"""Manual tracking test — camera feed with keyboard-controlled servos.

Captures frames from the Pi Camera, saves them with a crosshair overlay,
and lets you steer the pan-tilt with keyboard commands. This validates
the full camera+servo pipeline before automated face tracking.

Since we're headless (no display), this works in terminal mode:
frames are saved periodically and you type commands to move.

Hardware:
  PCA9685 at 0x40, pan=ch0 (0-180, center=90), tilt=ch1 (90-180, forward=180, up=90)
"""

import time
import os

from picamera2 import Picamera2
import cv2
import numpy as np
from adafruit_servokit import ServoKit

# PCA9685 channels
PAN_CH = 0
TILT_CH = 1

# Servo limits
PAN_MIN, PAN_MAX, PAN_CENTER = 0, 180, 90
TILT_MIN, TILT_MAX, TILT_CENTER = 90, 180, 180  # 180=forward, 90=up

# Movement step size in degrees
STEP = 5

SERVO_MIN_PULSE = 500
SERVO_MAX_PULSE = 2500


def make_kit() -> ServoKit:
    kit = ServoKit(channels=16)
    kit.servo[PAN_CH].set_pulse_width_range(SERVO_MIN_PULSE, SERVO_MAX_PULSE)
    kit.servo[TILT_CH].set_pulse_width_range(SERVO_MIN_PULSE, SERVO_MAX_PULSE)
    return kit


def draw_crosshair(frame: np.ndarray) -> np.ndarray:
    """Draw a crosshair at the center of the frame."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    color = (0, 255, 0)
    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), color, 2)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), color, 2)
    return frame


def save_snapshot(cam: Picamera2, pan_pos: int, tilt_pos: int, output_dir: str = "test_output"):
    """Capture a frame, draw crosshair and position info, save it."""
    os.makedirs(output_dir, exist_ok=True)

    frame = cam.capture_array()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    draw_crosshair(frame_bgr)

    label = f"Pan:{pan_pos} Tilt:{tilt_pos}"
    cv2.putText(frame_bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    path = os.path.join(output_dir, "manual_view.jpg")
    cv2.imwrite(path, frame_bgr)
    return path


def main():
    print("=== Ocelot Manual Tracking Test ===\n")

    kit = make_kit()

    pan_pos = PAN_CENTER
    tilt_pos = TILT_CENTER
    kit.servo[PAN_CH].angle = pan_pos
    kit.servo[TILT_CH].angle = tilt_pos

    cam = Picamera2()
    config = cam.create_still_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    cam.configure(config)
    cam.start()
    time.sleep(1)

    print("Controls:")
    print("  w/s  — tilt up/down")
    print("  a/d  — pan left/right")
    print("  f    — save snapshot to test_output/manual_view.jpg")
    print("  c    — center both servos")
    print("  q    — quit")
    print(f"  Step size: {STEP}°")
    print(f"  Pan range: {PAN_MIN}-{PAN_MAX} (center {PAN_CENTER})")
    print(f"  Tilt range: {TILT_MIN}-{TILT_MAX} (forward {TILT_CENTER})")
    print()

    path = save_snapshot(cam, pan_pos, tilt_pos)
    print(f"Initial snapshot saved to {path}")

    while True:
        try:
            cmd = input(f"[pan={pan_pos} tilt={tilt_pos}] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if not cmd:
            continue

        if cmd == "q":
            break
        elif cmd == "c":
            pan_pos = PAN_CENTER
            tilt_pos = TILT_CENTER
            kit.servo[PAN_CH].angle = pan_pos
            kit.servo[TILT_CH].angle = tilt_pos
            print("  Centered.")
        elif cmd == "a":
            pan_pos = max(PAN_MIN, pan_pos - STEP)
            kit.servo[PAN_CH].angle = pan_pos
        elif cmd == "d":
            pan_pos = min(PAN_MAX, pan_pos + STEP)
            kit.servo[PAN_CH].angle = pan_pos
        elif cmd == "w":
            # Tilt up = toward 90
            tilt_pos = max(TILT_MIN, tilt_pos - STEP)
            kit.servo[TILT_CH].angle = tilt_pos
        elif cmd == "s":
            # Tilt down = toward 180
            tilt_pos = min(TILT_MAX, tilt_pos + STEP)
            kit.servo[TILT_CH].angle = tilt_pos
        elif cmd == "f":
            path = save_snapshot(cam, pan_pos, tilt_pos)
            print(f"  Snapshot saved to {path}")
        else:
            print("  Unknown command. Use w/a/s/d/f/c/q.")

        if cmd in ("w", "a", "s", "d", "c"):
            save_snapshot(cam, pan_pos, tilt_pos)

    kit.servo[PAN_CH].angle = PAN_CENTER
    kit.servo[TILT_CH].angle = TILT_CENTER
    cam.stop()
    print("\nCentered and stopped. Done.")


if __name__ == "__main__":
    main()
