"""
Test 27: Camera viewer annotation parsing without GUI or hardware.
"""

from __future__ import annotations

import json

from _working_test_utils import require, run_tests

from vision.camera_viewer import CameraViewer


def test_single_camera_localization_updates_annotation_state():
    viewer = CameraViewer(None, None)
    viewer.set_prompt("pick up the red cup")

    result_text = json.dumps({
        "units": "meters",
        "source_camera": "d435",
        "points": [
            {
                "pixel": [320, 240],
                "status": "ok",
                "xyz_robot": [0.1, 0.2, 0.3],
                "depth_m": 0.72,
            }
        ],
        "successful_count": 1,
    })
    viewer.handle_tool_result("get_xyz_d435", {"coords": [[320, 240]]}, result_text)

    state = viewer.snapshot_state()
    annotations = state["annotations"]["d435"]
    require(len(annotations) == 1, "expected one D435 annotation")
    require(annotations[0]["pixel"] == (320, 240), "pixel was not preserved")
    require(annotations[0]["status"] == "ok", "status was not preserved")
    require(annotations[0]["xyz_robot"] == [0.1, 0.2, 0.3], "xyz was not preserved")
    require("D435 identified target" in state["status"], "status line should summarize identification")


def test_fused_localization_updates_both_camera_annotations():
    viewer = CameraViewer(None, None)
    viewer.set_prompt("move to the blue block")

    result_text = json.dumps({
        "units": "meters",
        "fused_result": {
            "xyz": [0.4, 0.5, 0.6],
            "valid": True,
            "sources_used": ["d435", "d405"],
        },
        "d435": {
            "results": [
                {
                    "pixel": [101, 202],
                    "status": "ok",
                    "xyz_robot": [0.39, 0.49, 0.61],
                    "depth_m": 0.81,
                }
            ]
        },
        "d405": {
            "results": [
                {
                    "pixel": [303, 404],
                    "status": "ok",
                    "xyz_robot": [0.41, 0.51, 0.59],
                    "depth_m": 0.28,
                }
            ]
        },
    })
    viewer.handle_tool_result("get_xyz_fused", {}, result_text)

    state = viewer.snapshot_state()
    require(len(state["annotations"]["d435"]) == 1, "expected one D435 fused annotation")
    require(len(state["annotations"]["d405"]) == 1, "expected one D405 fused annotation")
    require("Fused target identified" in state["status"], "fused status should summarize identification")


def test_new_prompt_clears_old_annotations():
    viewer = CameraViewer(None, None)
    viewer.set_prompt("find the first target")
    viewer.handle_tool_result(
        "get_xyz_d435",
        {},
        json.dumps({
            "source_camera": "d435",
            "points": [{"pixel": [1, 2], "status": "invalid", "reason": "No depth"}],
        }),
    )
    require(len(viewer.snapshot_state()["annotations"]["d435"]) == 1, "setup annotation missing")

    viewer.set_prompt("find the next target")
    state = viewer.snapshot_state()
    require(state["annotations"]["d435"] == [], "D435 annotations should clear on new prompt")
    require(state["annotations"]["d405"] == [], "D405 annotations should clear on new prompt")
    require(state["prompt"] == "find the next target", "prompt should update")


if __name__ == "__main__":
    raise SystemExit(
        run_tests(
            [
                (
                    "single-camera localization updates annotation state",
                    test_single_camera_localization_updates_annotation_state,
                ),
                (
                    "fused localization updates both camera annotations",
                    test_fused_localization_updates_both_camera_annotations,
                ),
                ("new prompt clears old annotations", test_new_prompt_clears_old_annotations),
            ]
        )
    )
