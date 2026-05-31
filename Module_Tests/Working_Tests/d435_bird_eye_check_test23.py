"""
Test 23: D435 Bird-Eye Check
----------------------------
Verifies the D435 fixed overhead extrinsic calibration file is valid and
mathematically sound.
"""
from _working_test_utils import add_working_to_path

add_working_to_path()

import config as cfg
from _extrinsic_check_utils import load_matrix_from_json, validate_rigid_transform


def main():
    print("--- Testing D435 Bird-Eye Extrinsics ---")
    print(f"File: {cfg.BIRD_EYE_D435_PATH}")

    matrix, error = load_matrix_from_json(cfg.BIRD_EYE_D435_PATH)
    if error:
        print(f"FAIL: {error}")
        print("ACTION: Run Working/camera_calibration/run_extrinsics_d435_bird_eye.py.")
        return 1

    is_valid, reason = validate_rigid_transform(matrix)
    if not is_valid:
        print(f"FAIL: {reason}")
        return 1

    print(f"PASS: {reason}")
    print(f"D435 camera-to-robot-base translation (X, Y, Z in meters): {matrix[:3, 3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
