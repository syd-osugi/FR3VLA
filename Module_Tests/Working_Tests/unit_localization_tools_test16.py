"""
Test 16: Pixel localization helper behavior without cameras.
"""

from __future__ import annotations

from _working_test_utils import patched_attr, require, require_close, run_tests

from vision.tools import localization


def test_process_pixel_rejects_bad_inputs_before_depth_math():
    called = {"pixel_to_xyz": False}

    def should_not_be_called(*args, **kwargs):
        called["pixel_to_xyz"] = True
        raise AssertionError("pixel_to_xyz should not be called for invalid inputs")

    with patched_attr(localization, "pixel_to_xyz", should_not_be_called):
        none_point = localization.process_pixel(None, (100, 80), object(), 0.001, "d435")
        require(none_point["status"] == "invalid", "None point should be invalid")

        bad_values = localization.process_pixel(["x", 5], (100, 80), object(), 0.001, "d435")
        require(bad_values["status"] == "invalid" and "must be integers" in bad_values["reason"], "string pixel should be invalid")

        out_of_bounds = localization.process_pixel([100, 0], (100, 80), object(), 0.001, "d435")
        require(out_of_bounds["status"] == "invalid" and "out of bounds" in out_of_bounds["reason"], "bounds check failed")

    require(called["pixel_to_xyz"] is False, "invalid inputs should not reach depth math")


def test_process_pixel_depth_and_transform_failures():
    def invalid_depth(*args, **kwargs):
        return {"x": 0.0, "y": 0.0, "z": 0.0, "valid": False}

    with patched_attr(localization, "pixel_to_xyz", invalid_depth):
        result = localization.process_pixel([5, 6], (100, 80), object(), 0.001, "d435")
        require(result["status"] == "invalid" and "No valid depth" in result["reason"], "invalid depth should be reported")

    def valid_depth(*args, **kwargs):
        return {"x": 1.0, "y": 2.0, "z": 3.0, "valid": True}

    def bad_transform(point, source_camera, robot_ee_pose=None):
        return {"xyz": None, "valid": False, "reason": "fake missing calibration"}

    with patched_attr(localization, "pixel_to_xyz", valid_depth), patched_attr(
        localization,
        "translate_point_to_robot_frame",
        bad_transform,
    ):
        result = localization.process_pixel([5, 6], (100, 80), object(), 0.001, "d435")
        require(result["status"] == "invalid" and "Transform failed" in result["reason"], "transform failure should be reported")


def test_process_pixel_success_and_filtering():
    def valid_depth(u, v, depth_rs, depth_scale):
        return {"x": float(u), "y": float(v), "z": 0.5, "valid": True}

    def good_transform(point, source_camera, robot_ee_pose=None):
        return {"xyz": [point[0] + 10.0, point[1] + 20.0, point[2] + 30.0], "valid": True}

    with patched_attr(localization, "pixel_to_xyz", valid_depth), patched_attr(
        localization,
        "translate_point_to_robot_frame",
        good_transform,
    ):
        result = localization.process_pixel([5, 6], (100, 80), object(), 0.001, "d405", robot_ee_pose="pose")
        require(result["status"] == "ok", f"expected ok result, got {result}")
        require_close(result["xyz_camera"], [5.0, 6.0, 0.5], "camera xyz wrong")
        require_close(result["xyz_robot"], [15.0, 26.0, 30.5], "robot xyz wrong")

    points = localization.valid_robot_points(
        [
            {"status": "ok", "xyz_robot": [1.0, 2.0, 3.0]},
            {"status": "invalid", "xyz_robot": [9.0, 9.0, 9.0]},
        ],
        "d435",
    )
    require(points == [{"xyz": [1.0, 2.0, 3.0], "valid": True, "source": "d435"}], "valid point filtering failed")


if __name__ == "__main__":
    raise SystemExit(
        run_tests(
            [
                ("process pixel rejects bad inputs before depth math", test_process_pixel_rejects_bad_inputs_before_depth_math),
                ("process pixel depth and transform failures", test_process_pixel_depth_and_transform_failures),
                ("process pixel success and filtering", test_process_pixel_success_and_filtering),
            ]
        )
    )
