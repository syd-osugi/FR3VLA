"""
Test 17: Robot trajectory and transform math with temporary calibration files.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from _working_test_utils import homogeneous, patched_attr, require, require_close, require_raises, run_tests, write_transform_json

from robot import trajectory


def test_translate_point_to_robot_frame_with_temp_calibrations():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        d435_path = write_transform_json(tmp_path / "d435_to_base.json", homogeneous((1.0, 2.0, 3.0)))
        d405_path = write_transform_json(tmp_path / "d405_to_ee.json", homogeneous((0.1, 0.2, 0.3)))

        with patched_attr(trajectory.cfg, "BIRD_EYE_D435_PATH", str(d435_path)), patched_attr(
            trajectory.cfg,
            "HAND_EYE_D405_PATH",
            str(d405_path),
        ):
            d435 = trajectory.translate_point_to_robot_frame([0.5, 0.0, 0.0], "d435")
            require(d435["valid"] is True, f"D435 transform failed: {d435}")
            require_close(d435["xyz"], [1.5, 2.0, 3.0], "D435 camera-to-base translation failed")

            missing_pose = trajectory.translate_point_to_robot_frame([0.0, 0.0, 0.0], "d405")
            require(missing_pose["valid"] is False and "robot_ee_pose required" in missing_pose["reason"], "D405 should require EE pose")

            T_ee_to_base = homogeneous((10.0, 0.0, 0.0))
            d405 = trajectory.translate_point_to_robot_frame([0.5, 0.0, 0.0], "d405", robot_ee_pose=T_ee_to_base)
            require(d405["valid"] is True, f"D405 transform failed: {d405}")
            require_close(d405["xyz"], [10.6, 0.2, 0.3], "D405 camera-to-EE-to-base chain failed")

            bad_camera = trajectory.translate_point_to_robot_frame([0.0, 0.0, 0.0], "not-a-camera")
            require(bad_camera["valid"] is False and "Unknown camera" in bad_camera["reason"], "unknown camera should fail")


def test_fusion_logic_and_quality_warning():
    d435_points = [
        {"xyz": [1.0, 1.0, 1.0], "valid": True},
        {"xyz": [3.0, 1.0, 1.0], "valid": True},
    ]
    single = trajectory.translate_points_fused(d435_points, [])
    require(single["valid"] is True, "single-camera fusion should be valid")
    require_close(single["xyz"], [2.0, 1.0, 1.0], "D435 valid points should average")

    both = trajectory.translate_points_fused(
        [{"xyz": [0.0, 0.0, 0.0], "valid": True}],
        [{"xyz": [0.2, 0.0, 0.0], "valid": True}],
    )
    require(both["valid"] is True, "two-camera fusion should be valid")
    require_close(both["xyz"], [0.1, 0.0, 0.0], "two-camera fusion average failed")
    require("quality_warning" in both, "large camera disagreement should produce warning")

    none = trajectory.translate_points_fused([{"valid": False}], [])
    require(none["valid"] is False and "No valid estimates" in none["reason"], "no valid points should fail")


def test_waypoint_generation_with_and_without_gripper_offset():
    with patched_attr(trajectory.cfg, "GRIPPER_TCP_IN_EE_TRANSLATION_M", (0.0, 0.0, 0.0)), patched_attr(
        trajectory.cfg,
        "GRIPPER_TCP_IN_EE_RPY_DEG",
        (0.0, 0.0, 0.0),
    ):
        waypoints, metadata = trajectory.get_robot_trajectory_to_point(
            [0.2, 0.3, 0.4],
            approach_height=0.1,
            approach_direction="z",
            return_metadata=True,
        )
        require_close(waypoints, [[0.2, 0.3, 0.5], [0.2, 0.3, 0.4]], "zero-offset z approach failed")
        require(metadata["waypoints_are"] == "end_effector_origin_positions_in_robot_base", "metadata label changed")

    Rz90 = trajectory.rotation_matrix_from_rpy_deg(0.0, 0.0, 90.0)
    T_ee_to_base = homogeneous((0.0, 0.0, 0.0), Rz90)
    with patched_attr(trajectory.cfg, "GRIPPER_TCP_IN_EE_TRANSLATION_M", (0.1, 0.0, 0.0)), patched_attr(
        trajectory.cfg,
        "GRIPPER_TCP_IN_EE_RPY_DEG",
        (0.0, 0.0, 0.0),
    ):
        waypoints = trajectory.get_robot_trajectory_to_point(
            [1.0, 1.0, 1.0],
            approach_height=0.05,
            approach_direction="z",
            robot_ee_pose=T_ee_to_base,
        )
        require_close(waypoints, [[1.0, 0.9, 1.05], [1.0, 0.9, 1.0]], "rotated gripper offset should subtract in base frame")
        require_raises(
            ValueError,
            lambda: trajectory.get_robot_trajectory_to_point([1.0, 1.0, 1.0], approach_height=0.0),
            "zero approach height should raise",
        )


if __name__ == "__main__":
    raise SystemExit(
        run_tests(
            [
                ("translate point with temp calibrations", test_translate_point_to_robot_frame_with_temp_calibrations),
                ("fusion logic and quality warning", test_fusion_logic_and_quality_warning),
                ("waypoint generation with and without gripper offset", test_waypoint_generation_with_and_without_gripper_offset),
            ]
        )
    )
