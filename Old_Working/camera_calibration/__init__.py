"""
Camera Calibration Package
--------------------------
Interactive scripts and reusable math helpers for camera intrinsics, D435
bird's-eye extrinsics, and D405 eye-in-hand extrinsics.

User-run calibration scripts:
    intrinsics_check.py
        Verifies existing D435/D405 intrinsic JSON files.

    calibrate_intrinsics.py
        Captures ChArUco images and writes:
            calibration_data/d435_intrinsics.json
            calibration_data/d405_intrinsics.json

    run_extrinsics_d405_hand_eye.py
        Uses a fixed ChArUco board and robot wrist motion to write:
            calibration_data/d405_to_wrist.json

    run_extrinsics_d435_bird_eye.py
        Uses an end-effector-mounted ChArUco board and fixed overhead D435 to write:
            calibration_data/d435_to_robot_base.json

Helper-only modules:
    intrinsics_math.py, hand_eye_math.py, bird_eye_math.py, charuco_utils.py

Runtime use:
    robot/trajectory.py loads the saved extrinsic JSON files to convert D435 and
    D405 depth points into robot-base coordinates. It does not recalibrate.
"""
