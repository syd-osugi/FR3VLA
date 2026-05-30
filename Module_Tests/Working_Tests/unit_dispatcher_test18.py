"""
Test 18: Tool dispatcher validation and trajectory routing without cameras.
"""

from __future__ import annotations

import json

from _working_test_utils import patched_attr, require, run_tests

from vision.tools import dispatcher


def test_dispatcher_rejects_bad_tool_calls_without_hardware():
    text, image = dispatcher.dispatch("missing_tool", {}, None, None)
    require(image is None and "Unknown tool" in text, "unknown tool should return text error")

    text, _ = dispatcher.dispatch("get_xyz_d435", {"coords": []}, None, None)
    require("'coords' must be a non-empty list" in text, "empty coords should be rejected before capture")

    text, _ = dispatcher.dispatch("get_xyz_fused", {}, None, None)
    require("provide coordinates from at least one camera" in text, "empty fused coords should be rejected before capture")

    text, _ = dispatcher.dispatch("get_xyz_fused", {"d435_coords": "not-list"}, None, None)
    require("'d435_coords' must be a list" in text, "bad optional coords should be rejected")

    text, _ = dispatcher.dispatch("get_xyz_d435", [], None, None)
    require("Tool arguments must be a JSON object" in text, "non-dict args should be rejected")


def test_plan_robot_trajectory_routes_parsed_arguments():
    seen = {}

    def fake_planner(target_xyz, approach_height=0.10, approach_direction="z", robot_ee_pose=None, return_metadata=False):
        seen["target_xyz"] = target_xyz
        seen["approach_height"] = approach_height
        seen["approach_direction"] = approach_direction
        seen["robot_ee_pose"] = robot_ee_pose
        seen["return_metadata"] = return_metadata
        return [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], {"planner": "fake"}

    robot_pose = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    with patched_attr(dispatcher, "get_robot_trajectory_to_point", fake_planner):
        text, image = dispatcher.dispatch(
            "plan_robot_trajectory",
            {"target_xyz": [0.1, 0.2, 0.3], "approach_direction": "x", "approach_height_m": 0.05},
            None,
            None,
            robot_ee_pose=robot_pose,
        )

    require(image is None, "trajectory planning should not return an image")
    data = json.loads(text)
    require(data["waypoints"] == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "dispatcher did not return planner waypoints")
    require(seen["target_xyz"] == [0.1, 0.2, 0.3], "target parse failed")
    require(seen["approach_height"] == 0.05, "approach height parse failed")
    require(seen["approach_direction"] == "x", "approach direction parse failed")
    require(seen["robot_ee_pose"] == robot_pose, "robot pose was not forwarded")
    require(seen["return_metadata"] is True, "dispatcher should request metadata")


def test_plan_robot_trajectory_reports_parse_errors():
    text, image = dispatcher.dispatch("plan_robot_trajectory", {"target_xyz": [1.0, 2.0]}, None, None)
    require(image is None and "target_xyz must be" in text, "bad target should return parse error")


if __name__ == "__main__":
    raise SystemExit(
        run_tests(
            [
                ("dispatcher rejects bad tool calls without hardware", test_dispatcher_rejects_bad_tool_calls_without_hardware),
                ("plan robot trajectory routes parsed arguments", test_plan_robot_trajectory_routes_parsed_arguments),
                ("plan robot trajectory reports parse errors", test_plan_robot_trajectory_reports_parse_errors),
            ]
        )
    )
