"""
Test 14: Calibration math helpers without cameras or OpenCV.
"""

from __future__ import annotations

import numpy as np

from _working_test_utils import homogeneous, require, require_close, run_tests

from camera_calibration import bird_eye_math, hand_eye_math, intrinsics_math


def test_intrinsics_matrix_conversion_and_validation():
    factory = {
        "width": 640,
        "height": 480,
        "fx": 600.0,
        "fy": 610.0,
        "ppx": 320.0,
        "ppy": 240.0,
        "distortion": [0.1, 0.2, 0.0, 0.0, 0.0],
    }
    camera_matrix, dist = intrinsics_math.get_opencv_matrices(factory)
    require_close(camera_matrix, [[600.0, 0.0, 320.0], [0.0, 610.0, 240.0], [0.0, 0.0, 1.0]], "factory matrix conversion failed")
    require_close(dist, [0.1, 0.2, 0.0, 0.0, 0.0], "factory distortion conversion failed")

    image_calibration = {
        "camera_matrix": [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
        "reprojection_error": 0.25,
    }
    ok, reason = intrinsics_math._validate_image_calibration(image_calibration)
    require(ok is True, f"good image calibration should validate: {reason}")

    bad = dict(image_calibration)
    bad["reprojection_error"] = 2.0
    ok, reason = intrinsics_math._validate_image_calibration(bad)
    require(ok is False and "too high" in reason, "high reprojection error should fail")


def test_bird_eye_transform_chain_and_average():
    rotation = bird_eye_math.rotation_matrix_from_rpy_deg(0, 0, 90)
    require_close(rotation @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], "90 degree yaw failed")

    T_board_to_cam = homogeneous((0.0, 0.0, 1.0))
    T_board_to_ee = bird_eye_math.make_board_to_ee_transform((0.1, 0.0, 0.0), (0.0, 0.0, 0.0))
    T_ee_to_base = homogeneous((1.0, 2.0, 3.0))
    actual = bird_eye_math.camera_to_robot_from_pose(T_board_to_cam, T_board_to_ee, T_ee_to_base)
    expected = T_ee_to_base @ T_board_to_ee @ np.linalg.inv(T_board_to_cam)
    require_close(actual, expected, "camera-to-robot transform chain failed")

    transforms = [homogeneous((0.0, 0.0, 0.0)), homogeneous((2.0, 0.0, 0.0))]
    averaged = bird_eye_math.average_transforms(transforms)
    require_close(averaged[:3, 3], [1.0, 0.0, 0.0], "translation average failed")
    report = bird_eye_math.validate_transform_set(transforms, averaged)
    require(report["pose_count"] == 2, "validation should report pose count")
    require(report["max_translation_error_m"] == 1.0, "validation translation spread unexpected")


def test_hand_eye_consistency_validation():
    T_cam_to_gripper = homogeneous((0.1, 0.2, 0.3))
    T_target_to_base = homogeneous((1.0, 2.0, 3.0))
    robot_poses = [
        homogeneous((0.0, 0.0, 0.0)),
        homogeneous((0.2, 0.1, 0.0)),
        homogeneous((-0.1, 0.3, 0.2)),
    ]
    camera_poses = [
        np.linalg.inv(T_cam_to_gripper) @ np.linalg.inv(T_gripper_to_base) @ T_target_to_base
        for T_gripper_to_base in robot_poses
    ]

    ok, rot_error, trans_error = hand_eye_math.validate_hand_eye_result(
        T_cam_to_gripper,
        robot_poses,
        camera_poses,
        tolerance_deg=0.001,
        tolerance_m=0.001,
    )
    require(ok is True, f"consistent hand-eye set should validate, got rot={rot_error}, trans={trans_error}")

    camera_poses[-1] = camera_poses[-1].copy()
    camera_poses[-1][:3, 3] += np.array([0.1, 0.0, 0.0])
    ok, _, _ = hand_eye_math.validate_hand_eye_result(
        T_cam_to_gripper,
        robot_poses,
        camera_poses,
        tolerance_deg=0.001,
        tolerance_m=0.001,
    )
    require(ok is False, "inconsistent hand-eye set should fail strict validation")


if __name__ == "__main__":
    raise SystemExit(
        run_tests(
            [
                ("intrinsics matrix conversion and validation", test_intrinsics_matrix_conversion_and_validation),
                ("bird-eye transform chain and average", test_bird_eye_transform_chain_and_average),
                ("hand-eye consistency validation", test_hand_eye_consistency_validation),
            ]
        )
    )
