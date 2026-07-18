"""
Test 31: Physical Extrinsic Validation
--------------------------------------
Checks whether the saved D405 and D435 extrinsic matrices agree with fresh
camera observations and robot poses.

This is stronger than d405_hand_eye_check_test13.py and
d435_bird_eye_check_test23.py. Those scripts verify that the saved JSON files
contain mathematically valid rigid transforms. This script validates the
physical calibration by re-observing a ChArUco board at held-out poses.

Modes:
  d405:
    Board is fixed in the workspace. The wrist-mounted D405 observes it from
    several robot poses. A correct T_d405_to_ee should predict the same
    board -> robot-base pose every time:

        T_board_to_base = T_ee_to_base @ T_d405_to_ee @ T_board_to_d405

  d435:
    Board is rigidly mounted to the end effector. The fixed overhead D435
    observes it from several robot poses. A correct T_d435_to_base should agree
    with the robot/mount prediction:

        T_board_to_base_from_camera = T_d435_to_base @ T_board_to_d435
        T_board_to_base_from_robot  = T_ee_to_base @ T_board_to_ee

Use poses that were not used for calibration, and vary wrist orientation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from _extrinsic_check_utils import load_matrix_from_json, validate_rigid_transform
from _working_test_utils import TEST_OUTPUTS_DIR, add_working_to_path, unique_output_path

add_working_to_path()

import config as cfg
from camera_calibration.bird_eye_math import average_transforms, make_board_to_ee_transform
from camera_calibration.charuco_utils import (
    create_charuco_board,
    detect_charuco_board_pose,
    draw_charuco_detection,
    draw_pose_axes,
    get_aruco_dictionary,
    require_charuco_support,
)
from camera_calibration.intrinsics_math import get_opencv_matrices, load_intrinsics
from robot.franka_setup import (
    LOAD_PROFILE_D405_CALIBRATION,
    LOAD_PROFILE_D435_CALIBRATION,
    apply_franka_control_config,
)

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

try:
    import pyrealsense2 as rs
except ModuleNotFoundError:
    rs = None


SCRIPT_NAME = Path(__file__).stem
DEFAULT_OUTPUT_DIR = TEST_OUTPUTS_DIR / SCRIPT_NAME


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate saved D405/D435 extrinsics with fresh ChArUco observations."
    )
    parser.add_argument(
        "--mode",
        choices=("d405", "d435", "both"),
        default="d405",
        help=(
            "Which extrinsic calibration to validate. Default: d405. "
            "Use separate d405/d435 runs when the payload setups differ."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Target number of fresh validation captures per mode. Default: 5.",
    )
    parser.add_argument(
        "--min-corners",
        type=int,
        default=15,
        help="Minimum ChArUco corners required before a capture is accepted. Default: 15.",
    )
    parser.add_argument(
        "--translation-tolerance-mm",
        type=float,
        default=20.0,
        help="Maximum allowed translation residual in millimeters. Default: 20.",
    )
    parser.add_argument(
        "--rotation-tolerance-deg",
        type=float,
        default=5.0,
        help="Maximum allowed rotation residual in degrees. Default: 5.",
    )
    parser.add_argument(
        "--assume-mount-ok",
        action="store_true",
        help="Skip the D435 board-to-EE mount confirmation prompt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for JSON reports. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def require_runtime_dependencies():
    require_charuco_support(cv2)
    if rs is None:
        raise RuntimeError("pyrealsense2 is required for RealSense validation.")


def validate_args(args):
    if args.samples < 3:
        raise ValueError("--samples must be at least 3")
    if args.min_corners < 6:
        raise ValueError("--min-corners must be at least 6 for solvePnP")
    if args.translation_tolerance_mm <= 0:
        raise ValueError("--translation-tolerance-mm must be positive")
    if args.rotation_tolerance_deg <= 0:
        raise ValueError("--rotation-tolerance-deg must be positive")


def load_profile_for_mode(mode):
    if mode == "d405":
        return LOAD_PROFILE_D405_CALIBRATION
    if mode == "d435":
        return LOAD_PROFILE_D435_CALIBRATION
    raise ValueError(
        "--mode both cannot use separate D405/D435 payload setups in one robot "
        "connection. Run --mode d405 and --mode d435 as separate validation runs."
    )


def connect_robot(load_profile):
    try:
        from pylibfranka import Robot
    except ImportError as exc:
        raise RuntimeError("pylibfranka is required to read robot poses.") from exc

    robot = Robot(cfg.FRANKA_IP)
    apply_franka_control_config(robot, load_profile=load_profile)
    return robot


def get_robot_ee_pose(robot):
    state = robot.read_once()
    return np.array(state.O_T_EE).reshape((4, 4), order="F")


def load_saved_transform(label, filepath):
    matrix, error = load_matrix_from_json(filepath)
    if error:
        raise RuntimeError(error)

    ok, reason = validate_rigid_transform(matrix)
    if not ok:
        raise RuntimeError(f"{label} transform is invalid: {reason}")

    print(f"  {label}: {reason}")
    return matrix


def load_camera_detector(camera_name):
    if camera_name == "d405":
        intrinsics_path = cfg.INTRINSICS_D405_PATH
        board_corners = cfg.HAND_EYE_BOARD_CORNERS
        square_size = cfg.HAND_EYE_SQUARE_SIZE
        marker_size = cfg.HAND_EYE_MARKER_SIZE
        aruco_name = cfg.HAND_EYE_ARUCO_DICT_NAME
        legacy_pattern = cfg.HAND_EYE_CHARUCO_LEGACY_PATTERN
        axis_length = 0.05
    elif camera_name == "d435":
        intrinsics_path = cfg.INTRINSICS_D435_PATH
        board_corners = cfg.BIRD_EYE_BOARD_CORNERS
        square_size = cfg.BIRD_EYE_SQUARE_SIZE
        marker_size = cfg.BIRD_EYE_MARKER_SIZE
        aruco_name = cfg.BIRD_EYE_ARUCO_DICT_NAME
        legacy_pattern = cfg.BIRD_EYE_CHARUCO_LEGACY_PATTERN
        axis_length = 0.08
    else:
        raise ValueError(f"Unknown camera: {camera_name}")

    intrinsics = load_intrinsics(intrinsics_path)
    if intrinsics is None:
        raise RuntimeError(f"{camera_name.upper()} intrinsics not found at {intrinsics_path}")

    camera_matrix, dist_coeffs = get_opencv_matrices(intrinsics)
    aruco_dict = get_aruco_dictionary(cv2, aruco_name)
    charuco_board = create_charuco_board(
        cv2,
        board_corners,
        square_size,
        marker_size,
        aruco_dict,
        legacy_pattern=legacy_pattern,
    )

    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "aruco_dict": aruco_dict,
        "charuco_board": charuco_board,
        "axis_length": axis_length,
        "intrinsics_path": intrinsics_path,
    }


def start_camera(camera_name):
    if camera_name == "d405":
        serial = cfg.D405_SERIAL
        resolution = cfg.D405_RESOLUTION
    elif camera_name == "d435":
        serial = cfg.D435_SERIAL
        resolution = cfg.D435_RESOLUTION
    else:
        raise ValueError(f"Unknown camera: {camera_name}")

    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_device(serial)
    rs_config.enable_stream(
        rs.stream.depth,
        resolution[0],
        resolution[1],
        rs.format.z16,
        cfg.CAMERA_FPS,
    )
    rs_config.enable_stream(
        rs.stream.color,
        resolution[0],
        resolution[1],
        rs.format.bgr8,
        cfg.CAMERA_FPS,
    )
    pipeline.start(rs_config)
    time.sleep(cfg.CAMERA_WARMUP_SECONDS)
    return pipeline


def rotation_error_deg(reference_rotation, measured_rotation):
    delta = reference_rotation.T @ measured_rotation
    cosine = (np.trace(delta) - 1.0) / 2.0
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def transform_error(reference, measured):
    return {
        "translation_error_m": float(np.linalg.norm(measured[:3, 3] - reference[:3, 3])),
        "rotation_error_deg": rotation_error_deg(reference[:3, :3], measured[:3, :3]),
    }


def summarize_errors(records):
    translation_errors = [record["translation_error_m"] for record in records]
    rotation_errors = [record["rotation_error_deg"] for record in records]
    return {
        "sample_count": len(records),
        "mean_translation_error_m": float(np.mean(translation_errors)),
        "max_translation_error_m": float(np.max(translation_errors)),
        "mean_rotation_error_deg": float(np.mean(rotation_errors)),
        "max_rotation_error_deg": float(np.max(rotation_errors)),
    }


def passes_thresholds(summary, translation_tolerance_m, rotation_tolerance_deg):
    return (
        summary["max_translation_error_m"] <= translation_tolerance_m
        and summary["max_rotation_error_deg"] <= rotation_tolerance_deg
    )


def print_summary(title, summary, translation_tolerance_m, rotation_tolerance_deg):
    print(f"\n  {title}")
    print(
        "    Max translation residual: "
        f"{summary['max_translation_error_m'] * 1000:.2f} mm "
        f"(limit {translation_tolerance_m * 1000:.2f} mm)"
    )
    print(
        "    Mean translation residual: "
        f"{summary['mean_translation_error_m'] * 1000:.2f} mm"
    )
    print(
        "    Max rotation residual: "
        f"{summary['max_rotation_error_deg']:.3f} deg "
        f"(limit {rotation_tolerance_deg:.3f} deg)"
    )
    print(
        "    Mean rotation residual: "
        f"{summary['mean_rotation_error_deg']:.3f} deg"
    )


def write_report(output_dir, mode, payload):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = unique_output_path(output_dir / f"{mode}_extrinsic_validation.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    print(f"  Report saved: {path}")


def draw_capture_overlay(display, mode_label, detection, captured_count, target_count):
    marker_count = detection.get("marker_count", 0)
    charuco_count = detection.get("charuco_count", 0)

    if detection["success"]:
        status = f"CHARUCO READY: {charuco_count} corners"
        status_color = (0, 255, 0)
        prompt = "Press 's' to save this validation pose"
    elif marker_count:
        status = f"PARTIAL BOARD: {marker_count} markers, {charuco_count} corners"
        status_color = (0, 220, 255)
        prompt = "Move until the board status turns green"
    else:
        status = "Searching for ChArUco board"
        status_color = (0, 0, 255)
        prompt = "Move until the board is visible"

    cv2.putText(
        display,
        mode_label,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        display,
        status,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2,
    )
    cv2.putText(
        display,
        prompt,
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        status_color,
        2,
    )
    cv2.putText(
        display,
        f"Captured: {captured_count}/{target_count}   q: finish/quit",
        (20, display.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )


def capture_charuco_samples(camera_name, mode_label, detector, robot, sample_count, min_corners):
    pipeline = start_camera(camera_name)
    samples = []
    window_name = f"{camera_name.upper()} Extrinsic Validation"

    try:
        while len(samples) < sample_count:
            frames = pipeline.wait_for_frames(timeout_ms=cfg.CALIBRATION_FRAME_TIMEOUT_MS)
            color_frame = frames.get_color_frame()
            if not color_frame:
                print("  Warning: dropped color frame, retrying...")
                continue

            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            detection = detect_charuco_board_pose(
                cv2,
                gray,
                detector["aruco_dict"],
                detector["charuco_board"],
                detector["camera_matrix"],
                detector["dist_coeffs"],
                min_corners=min_corners,
            )

            display = color_image.copy()
            draw_charuco_detection(cv2, display, detection)
            if detection["success"]:
                draw_pose_axes(
                    cv2,
                    display,
                    detection["T_board_to_cam"],
                    detector["camera_matrix"],
                    detector["dist_coeffs"],
                    axis_length=detector["axis_length"],
                )
            draw_capture_overlay(display, mode_label, detection, len(samples), sample_count)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                if not detection["success"]:
                    print("  ChArUco board is not detected well enough; capture skipped.")
                    continue

                T_ee_to_base = get_robot_ee_pose(robot)
                samples.append(
                    {
                        "T_ee_to_base": T_ee_to_base,
                        "T_board_to_cam": detection["T_board_to_cam"],
                        "charuco_count": int(detection.get("charuco_count", 0)),
                        "marker_count": int(detection.get("marker_count", 0)),
                    }
                )
                print(f"  Saved validation sample {len(samples)}/{sample_count}")

            elif key == ord("q"):
                print("  Finishing early...")
                break

    finally:
        pipeline.stop()
        cv2.destroyWindow(window_name)

    if len(samples) < 3:
        raise RuntimeError(f"Need at least 3 validation samples, captured {len(samples)}")

    return samples


def confirm_d435_mount(args):
    translation_m = tuple(cfg.BIRD_EYE_BOARD_TO_EE_TRANSLATION_M)
    rpy_deg = tuple(cfg.BIRD_EYE_BOARD_TO_EE_RPY_DEG)
    T_board_to_ee = make_board_to_ee_transform(translation_m, rpy_deg)

    print("\n  D435 mounted-board transform from config.py:")
    print(f"    BIRD_EYE_BOARD_TO_EE_TRANSLATION_M = {translation_m}")
    print(f"    BIRD_EYE_BOARD_TO_EE_RPY_DEG       = {rpy_deg}")
    for row in T_board_to_ee:
        print(f"    [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f} {row[3]:8.4f}]")

    if args.assume_mount_ok:
        return T_board_to_ee

    choice = input("  Are these board-to-EE mount values correct? [y/N]: ")
    if choice.strip().lower() not in ("y", "yes"):
        raise RuntimeError(
            "Update BIRD_EYE_BOARD_TO_EE_TRANSLATION_M and "
            "BIRD_EYE_BOARD_TO_EE_RPY_DEG before validating D435."
        )
    return T_board_to_ee


def validate_d405(robot, args):
    print("\n" + "=" * 70)
    print("  D405 HAND-EYE PHYSICAL VALIDATION")
    print("=" * 70)
    print(
        """
  SETUP:
    - Keep the ChArUco board fixed in the workspace, usually flat on the table.
    - Do not move the board during this validation.
    - Move the wrist-mounted D405 through held-out poses.
    - Use poses that were not used to calibrate d405_to_wrist.json.
    - Vary wrist orientation, not just XYZ position.
"""
    )
    input("  Press ENTER when the fixed-board D405 setup is ready...")

    T_d405_to_ee = load_saved_transform("D405 camera -> end-effector", cfg.HAND_EYE_D405_PATH)
    detector = load_camera_detector("d405")
    samples = capture_charuco_samples(
        "d405",
        "D405 hand-eye validation",
        detector,
        robot,
        args.samples,
        args.min_corners,
    )

    board_to_base_estimates = [
        sample["T_ee_to_base"] @ T_d405_to_ee @ sample["T_board_to_cam"]
        for sample in samples
    ]
    reference = average_transforms(board_to_base_estimates)

    records = []
    for index, (sample, estimate) in enumerate(zip(samples, board_to_base_estimates), start=1):
        errors = transform_error(reference, estimate)
        records.append(
            {
                "index": index,
                "translation_error_m": errors["translation_error_m"],
                "rotation_error_deg": errors["rotation_error_deg"],
                "charuco_count": sample["charuco_count"],
                "marker_count": sample["marker_count"],
                "T_board_to_base": estimate.tolist(),
            }
        )

    translation_tolerance_m = args.translation_tolerance_mm / 1000.0
    summary = summarize_errors(records)
    passed = passes_thresholds(
        summary,
        translation_tolerance_m,
        args.rotation_tolerance_deg,
    )

    print_summary(
        "Fixed-board consistency across D405 poses:",
        summary,
        translation_tolerance_m,
        args.rotation_tolerance_deg,
    )
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    write_report(
        args.output_dir,
        "d405",
        {
            "mode": "d405",
            "passed": passed,
            "test_meaning": "Fixed board should resolve to one stable robot-base pose.",
            "calibration_path": cfg.HAND_EYE_D405_PATH,
            "reference_T_board_to_base": reference.tolist(),
            "summary": summary,
            "records": records,
        },
    )
    return passed


def validate_d435(robot, args):
    print("\n" + "=" * 70)
    print("  D435 BIRD'S-EYE PHYSICAL VALIDATION")
    print("=" * 70)
    print(
        """
  SETUP:
    - Keep the D435 fixed above the workspace.
    - Rigidly mount the ChArUco board to the end effector.
    - The board-to-EE values in config.py must match the physical mount.
    - Move through held-out poses where the D435 sees the board clearly.
    - Vary pose and orientation across the normal workspace.
"""
    )
    input("  Press ENTER when the mounted-board D435 setup is ready...")

    T_d435_to_base = load_saved_transform("D435 camera -> robot base", cfg.BIRD_EYE_D435_PATH)
    T_board_to_ee = confirm_d435_mount(args)
    detector = load_camera_detector("d435")
    samples = capture_charuco_samples(
        "d435",
        "D435 bird's-eye validation",
        detector,
        robot,
        args.samples,
        args.min_corners,
    )

    records = []
    for index, sample in enumerate(samples, start=1):
        T_board_to_base_from_camera = T_d435_to_base @ sample["T_board_to_cam"]
        T_board_to_base_from_robot = sample["T_ee_to_base"] @ T_board_to_ee
        errors = transform_error(T_board_to_base_from_robot, T_board_to_base_from_camera)
        records.append(
            {
                "index": index,
                "translation_error_m": errors["translation_error_m"],
                "rotation_error_deg": errors["rotation_error_deg"],
                "charuco_count": sample["charuco_count"],
                "marker_count": sample["marker_count"],
                "T_board_to_base_from_camera": T_board_to_base_from_camera.tolist(),
                "T_board_to_base_from_robot": T_board_to_base_from_robot.tolist(),
            }
        )

    translation_tolerance_m = args.translation_tolerance_mm / 1000.0
    summary = summarize_errors(records)
    passed = passes_thresholds(
        summary,
        translation_tolerance_m,
        args.rotation_tolerance_deg,
    )

    print_summary(
        "Camera-predicted board pose vs robot/mount board pose:",
        summary,
        translation_tolerance_m,
        args.rotation_tolerance_deg,
    )
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    write_report(
        args.output_dir,
        "d435",
        {
            "mode": "d435",
            "passed": passed,
            "test_meaning": "Saved D435 transform should agree with robot pose plus board mount.",
            "calibration_path": cfg.BIRD_EYE_D435_PATH,
            "board_to_ee_translation_m": list(cfg.BIRD_EYE_BOARD_TO_EE_TRANSLATION_M),
            "board_to_ee_rpy_deg": list(cfg.BIRD_EYE_BOARD_TO_EE_RPY_DEG),
            "T_board_to_ee": T_board_to_ee.tolist(),
            "summary": summary,
            "records": records,
        },
    )
    return passed


def main():
    args = parse_args()

    try:
        require_runtime_dependencies()
        validate_args(args)
        load_profile = load_profile_for_mode(args.mode)
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("=" * 70)
    print("  PHYSICAL EXTRINSIC VALIDATION")
    print("=" * 70)
    print(f"  Mode: {args.mode}")
    print(f"  Samples per mode: {args.samples}")
    print(f"  Translation tolerance: {args.translation_tolerance_mm:.2f} mm")
    print(f"  Rotation tolerance: {args.rotation_tolerance_deg:.3f} deg")
    print(f"  Franka load profile: {load_profile}")

    try:
        print(f"\n[1/2] Connecting to Franka robot at {cfg.FRANKA_IP}...")
        robot = connect_robot(load_profile)
        print("      Robot connected")

        results = []
        if args.mode in ("d405", "both"):
            results.append(validate_d405(robot, args))
        if args.mode in ("d435", "both"):
            results.append(validate_d435(robot, args))

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 1
    except Exception as exc:
        print(f"\nERROR: {exc}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        if cv2 is not None:
            cv2.destroyAllWindows()

    if all(results):
        print("\nPASS: requested extrinsic validation checks passed.")
        return 0

    print("\nFAIL: at least one extrinsic validation check exceeded tolerance.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
