"""
Test 12: Config parsing and import safety.

This script checks small, no-hardware pieces first: config parsers, environment
overrides, and importability of modules that should not open cameras on import.
"""

from __future__ import annotations

import importlib

from _working_test_utils import require, run_tests, temp_environ

import config as cfg


def test_config_parser_helpers():
    require(cfg._parse_tuple("640,480", (1, 2)) == (640, 480), "resolution tuple should parse")
    require(cfg._parse_tuple("bad", (1, 2)) == (1, 2), "bad tuple should fall back")
    require(cfg._parse_tuple("640,-1", (1, 2)) == (1, 2), "negative tuple should fall back")

    require(cfg._parse_int("12", 1, min_value=5, max_value=10) == 10, "int should clamp high")
    require(cfg._parse_int("bad", 7, min_value=5) == 7, "bad int should fall back")

    require(cfg._parse_float("0.25", 1.0, min_value=0.5) == 0.5, "float should clamp low")
    require(cfg._parse_float("bad", 1.25) == 1.25, "bad float should fall back")

    require(cfg._parse_float_tuple("0.1,-0.2,0.3", (0, 0, 0), 3) == (0.1, -0.2, 0.3), "float tuple should parse")
    require(cfg._parse_float_tuple("0.1,0.2", (1, 2, 3), 3) == (1, 2, 3), "wrong tuple length should fall back")

    require(cfg._parse_bool("yes", False) is True, "yes should parse true")
    require(cfg._parse_bool("OFF", True) is False, "OFF should parse false")
    require(cfg._parse_bool("maybe", True) is True, "unknown bool should fall back")


def test_environment_overrides_reload():
    with temp_environ(
        {
            "D435_RESOLUTION": "800,600",
            "CAMERA_FPS": "120",
            "LLM_TEMPERATURE": "bad",
            "POLICY_REQUIRE_ROBOT_STATE": "yes",
            "GRIPPER_TCP_IN_EE_TRANSLATION_M": "0.1,-0.2,0.3",
        }
    ):
        fresh = importlib.reload(cfg)
        require(fresh.D435_RESOLUTION == (800, 600), "D435 resolution env override failed")
        require(fresh.CAMERA_FPS == 90, "CAMERA_FPS should be clamped to max 90")
        require(fresh.LLM_TEMPERATURE == 0.1, "bad temperature should fall back")
        require(fresh.POLICY_REQUIRE_ROBOT_STATE is True, "bool env override failed")
        require(
            fresh.GRIPPER_TCP_IN_EE_TRANSLATION_M == (0.1, -0.2, 0.3),
            "gripper TCP env override failed",
        )

    importlib.reload(cfg)


def test_low_level_imports_do_not_touch_hardware():
    module_names = [
        "config",
        "utilities.coordinates",
        "camera_calibration.charuco_utils",
        "camera_calibration.intrinsics_math",
        "camera_calibration.hand_eye_math",
        "camera_calibration.bird_eye_math",
        "hardware.camera",
        "robot.trajectory",
        "robot.safety",
        "robot.robot_interface",
        "policy.actions",
        "policy.observation",
        "policy.base",
        "policy.scripted",
        "policy.inference",
        "vision.tools.camera_frames",
        "vision.tools.dispatcher",
        "vision.tools.localization",
        "vision.tools.schemas",
        "vision.llm_interface",
    ]
    for module_name in module_names:
        module = importlib.import_module(module_name)
        require(module is not None, f"failed to import {module_name}")


if __name__ == "__main__":
    raise SystemExit(
        run_tests(
            [
                ("config parser helpers", test_config_parser_helpers),
                ("environment overrides reload", test_environment_overrides_reload),
                ("low-level imports do not touch hardware", test_low_level_imports_do_not_touch_hardware),
            ]
        )
    )
