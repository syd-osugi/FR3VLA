"""
Intrinsic Calibration Checker
-----------------------------
Verifies that saved intrinsic calibration files still match the connected D435
and D405 cameras at the configured stream resolution.

Run this before extrinsic calibration. If either camera fails, re-run
calibrate_intrinsics.py for that camera/setup.
"""

import os
import sys
import time

WORKING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKING_DIR not in sys.path:
    sys.path.insert(0, WORKING_DIR)

import config as cfg
from camera_calibration.intrinsics_math import validate_saved_intrinsics
from hardware.camera import RealSense


def _is_configured_serial(serial_number):
    return bool(serial_number) and not serial_number.startswith("YOUR_")


def _camera_resolution(camera_name):
    return cfg.D435_RESOLUTION if camera_name == "D435" else cfg.D405_RESOLUTION


def check_single_camera(camera_name, serial_number, json_path):
    """Checks one camera and returns True only when its intrinsics are usable."""
    print("\n" + "-" * 60)
    print(f"Checking {camera_name} intrinsics")
    print(f"Serial: {serial_number}")
    print(f"File:   {json_path}")
    print("-" * 60)

    if not os.path.exists(json_path):
        print("FAIL: calibration file not found")
        print("ACTION: run calibrate_intrinsics.py")
        return False

    try:
        camera = RealSense(
            serial_number=serial_number,
            resolution=_camera_resolution(camera_name),
            fps=cfg.CAMERA_FPS,
        )
    except Exception as exc:
        print(f"FAIL: could not initialize camera: {exc}")
        return False

    try:
        time.sleep(cfg.CAMERA_WARMUP_SECONDS)
        is_valid, reason = validate_saved_intrinsics(camera.pipeline, json_path)
    except Exception as exc:
        print(f"FAIL: validation error: {exc}")
        return False
    finally:
        camera.stop()

    if is_valid:
        print(f"PASS: {reason}")
        return True

    print(f"FAIL: {reason}")
    print("ACTION: delete the stale JSON file and run calibrate_intrinsics.py")
    return False


def main():
    print("=" * 60)
    print("Intrinsic Calibration Check")
    print("=" * 60)

    checks = [
        ("D435", cfg.D435_SERIAL, cfg.INTRINSICS_D435_PATH),
        ("D405", cfg.D405_SERIAL, cfg.INTRINSICS_D405_PATH),
    ]

    results = {}
    for camera_name, serial_number, json_path in checks:
        if not _is_configured_serial(serial_number):
            print(f"\nFAIL: {camera_name} serial number is not configured in config.py")
            results[camera_name] = False
            continue
        results[camera_name] = check_single_camera(camera_name, serial_number, json_path)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for camera_name in ("D435", "D405"):
        print(f"{camera_name}: {'PASS' if results.get(camera_name) else 'FAIL'}")

    if all(results.values()):
        print("\nAll camera intrinsics validated. Ready for extrinsic calibration.")
        return 0

    print("\nOne or more intrinsic checks failed. Recalibrate before continuing.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
