#!/usr/bin/env python3
"""Servo validation script for PCA9685 + Adafruit pan-tilt bracket.

Sweeps pan and tilt through their range, then enters interactive mode
for manual positioning. Use this to find mechanical limits on your bracket.

PCA9685 at I2C address 0x40. SG90 servos on channels 0 (pan) and 1 (tilt).
Servo angle range: 0-180 degrees (90 = center).
"""

import time

from adafruit_servokit import ServoKit

# PCA9685 channels
PAN_CH = 0
TILT_CH = 1

# SG90 pulse range (microseconds) — default works for most SG90s
SERVO_MIN_PULSE = 500
SERVO_MAX_PULSE = 2500


def make_kit() -> ServoKit:
    kit = ServoKit(channels=16)
    kit.servo[PAN_CH].set_pulse_width_range(SERVO_MIN_PULSE, SERVO_MAX_PULSE)
    kit.servo[TILT_CH].set_pulse_width_range(SERVO_MIN_PULSE, SERVO_MAX_PULSE)
    return kit


def sweep_axis(kit: ServoKit, name: str, channel: int, start: int = 0, end: int = 180, step: int = 10):
    """Sweep one axis and report positions."""
    print(f"\n--- Sweeping {name}: {start}° to {end}° (step {step}) ---")
    print("Watch the servo. Note where it hits mechanical limits.")
    print("Press Ctrl+C to skip.\n")

    try:
        for angle in range(start, end + 1, step):
            kit.servo[channel].angle = angle
            print(f"  {name} = {angle:3d}°", flush=True)
            time.sleep(0.4)

        # Return to center
        kit.servo[channel].angle = 90
        print(f"  {name} centered at 90°")
    except KeyboardInterrupt:
        kit.servo[channel].angle = 90
        print(f"\n  Skipped. {name} centered.")


def interactive_mode(kit: ServoKit):
    """Interactive positioning via typed commands."""
    print("\n--- Interactive Mode ---")
    print("Commands:")
    print("  p <angle>   — set pan  (0-180, 90=center)")
    print("  t <angle>   — set tilt (0-180, 90=center)")
    print("  c           — center both (90)")
    print("  q           — quit")
    print()

    pan_pos = 90
    tilt_pos = 90

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
            kit.servo[PAN_CH].angle = 90
            kit.servo[TILT_CH].angle = 90
            pan_pos = tilt_pos = 90
            print("  Centered.")
        elif cmd.startswith("p "):
            try:
                angle = int(cmd[2:])
                angle = max(0, min(180, angle))
                kit.servo[PAN_CH].angle = angle
                pan_pos = angle
                print(f"  Pan set to {angle}°")
            except ValueError:
                print("  Invalid angle. Use: p <number>")
        elif cmd.startswith("t "):
            try:
                angle = int(cmd[2:])
                angle = max(0, min(180, angle))
                kit.servo[TILT_CH].angle = angle
                tilt_pos = angle
                print(f"  Tilt set to {angle}°")
            except ValueError:
                print("  Invalid angle. Use: t <number>")
        else:
            print("  Unknown command. Use p/t/c/q.")


def main():
    print("=== Ocelot Servo Validation ===")
    print("Using PCA9685 at 0x40, channels 0 (pan) and 1 (tilt)\n")

    kit = make_kit()

    # Center first
    kit.servo[PAN_CH].angle = 90
    kit.servo[TILT_CH].angle = 90
    time.sleep(0.5)
    print("Servos centered at 90°.")

    # Sweep pan
    sweep_axis(kit, "Pan", PAN_CH, 0, 180, 10)
    time.sleep(0.5)

    # Sweep tilt
    sweep_axis(kit, "Tilt", TILT_CH, 0, 180, 10)
    time.sleep(0.5)

    # Interactive
    interactive_mode(kit)

    # Clean up — center on exit
    kit.servo[PAN_CH].angle = 90
    kit.servo[TILT_CH].angle = 90
    print("\nServos centered. Done.")


if __name__ == "__main__":
    main()
