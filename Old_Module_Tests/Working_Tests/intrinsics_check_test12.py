"""
Test 12: Intrinsic Validation Check (D435 & D405)
--------------------------------------------------
Ensures that saved calibration files exactly match the plugged-in hardware.

This is a compatibility wrapper around the current calibration checker in
Working/camera_calibration. It reads the saved D435 and D405 intrinsic JSON
files, checks the configured camera serial numbers, and verifies the saved
intrinsics against the connected hardware.
"""
from _working_test_utils import add_working_to_path

add_working_to_path()

from camera_calibration.intrinsics_check import main


if __name__ == "__main__":
    raise SystemExit(main())
