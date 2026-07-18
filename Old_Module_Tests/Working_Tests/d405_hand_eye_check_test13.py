"""
Test 13: D405 Hand-Eye Check
----------------------------
Verifies the D405 eye-in-hand extrinsic calibration file is valid and
mathematically sound.
"""
from _working_test_utils import add_working_to_path

add_working_to_path()

import config as cfg
from _extrinsic_check_utils import validate_rigid_transform
from camera_calibration.hand_eye_math import load_hand_eye


def main():
    print("--- Testing D405 Hand-Eye Extrinsics ---")
    print(f"File: {cfg.HAND_EYE_D405_PATH}")

    try:
        matrix = load_hand_eye(cfg.HAND_EYE_D405_PATH)
    except Exception as exc:
        print(f"FAIL: Could not load D405 hand-eye file: {exc}")
        print("ACTION: Run Working/camera_calibration/run_extrinsics_d405_hand_eye.py.")
        return 1

    if matrix is None:
        print(f"FAIL: File not found at {cfg.HAND_EYE_D405_PATH}")
        print("ACTION: Run Working/camera_calibration/run_extrinsics_d405_hand_eye.py.")
        return 1

    is_valid, reason = validate_rigid_transform(matrix)
    if not is_valid:
        print(f"FAIL: {reason}")
        return 1

    print(f"PASS: {reason}")
    print(f"D405 camera-to-wrist translation (X, Y, Z in meters): {matrix[:3, 3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
