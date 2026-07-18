#!/usr/bin/env python3
"""
Script: D435 Eye-to-Hand Extrinsic Calibration (Bird's-Eye)
============================================================
Calibrates a fixed overhead D435 camera using a ChArUco board mounted rigidly
to the robot end-effector.

PHYSICAL SETUP:
  - D435 rigidly mounted to the table/ceiling, looking down at workspace
  - ChArUco board rigidly mounted to the robot end-effector
  - Robot moves the board so the fixed D435 sees it from many viewpoints

SUPPORTED CAMERAS:
  This script calibrates ONE camera at a time via the --camera argument:
    --camera left   : calibrates the left D435 (d435_left_serial)
    --camera right  : calibrates the right D435 (d435_right_serial)

  Each camera saves to its own output file:
    left  -> calibration_data/d435_left_to_base.json
    right -> calibration_data/d435_right_to_base.json

CALIBRATED TRANSFORM:
  T_d435_to_base: maps a point from D435 optical frame -> robot base frame
    p_base = T_d435_to_base @ p_d435

  This is computed per pose as:
    T_cam_to_base = T_ee_to_base @ T_board_to_ee @ inv(T_board_to_cam)
  then averaged across all poses.

WORKFLOW:
  python calibrate_d435_bird_eye.py --camera left
  python calibrate_d435_bird_eye.py --camera right

  Move the robot through varied poses while the selected D435 sees the board.
  Press 's' at each good pose until the required count is reached.

OUTPUT:
  JSON file containing:
    - The averaged 4x4 T_cam_to_base matrix
    - Validation metrics (per-pose agreement)
    - Camera serial number
    - Board-to-EE mount transform used
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
WORKING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKING_DIR not in sys.path:
    sys.path.insert(0, WORKING_DIR)

import numpy as np
import yaml

# Import calibration math module
from calibration_math import (
    average_transforms,
    create_charuco_board,
    detect_charuco_board_pose,
    draw_charuco_detection,
    draw_pose_axes,
    get_aruco_dictionary,
    get_opencv_matrices,
    load_intrinsics,
    make_board_to_ee_transform,
    require_charuco_support,
    save_fixed_camera_transform,
    validate_transform_set,
)

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

try:
    import pyrealsense2 as rs
except ModuleNotFoundError:
    rs = None

try:
    import pylibfranka
except ModuleNotFoundError:
    pylibfranka = None


def load_config(config_path):
    """Load calibration_config.yaml and return the dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command-line arguments. Requires --camera left|right."""
    parser = argparse.ArgumentParser(
        description="Calibrate a fixed D435 eye-to-hand camera using a ChArUco board on the end-effector."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to calibration_config.yaml. Default: calibration_config.yaml in this directory.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        required=True,
        choices=["left", "right"],
        help="Which D435 camera to calibrate: 'left' or 'right'.",
    )
    parser.add_argument(
        "--min-corners",
        type=int,
        default=15,
        help="Minimum ChArUco corners required before capture. Default: 15.",
    )
    return parser.parse_args()


def connect_robot(config):
    """Connect to the Franka robot and apply conservative collision thresholds."""
    if pylibfranka is None:
        raise RuntimeError("pylibfranka is required. Install with: pip install pylibfranka")

    realtime_config = getattr(getattr(pylibfranka, "RealtimeConfig", None), "kIgnore", None)
    if realtime_config is not None:
        robot = pylibfranka.Robot(config["franka_ip"], realtime_config)
    else:
        robot = pylibfranka.Robot(config["franka_ip"])

    robot.set_collision_behavior(
        [config["franka_collision_force_n"]] * 7,
        [config["franka_collision_force_n"]] * 7,
        [config["franka_collision_force_n"]] * 6,
        [config["franka_collision_force_n"]] * 6,
    )
    return robot


def get_robot_ee_pose(robot):
    """Read end-effector -> robot base pose from the Franka as a 4x4 matrix."""
    state = robot.read_once()
    return np.array(state.O_T_EE).reshape((4, 4), order="F")


def confirm_board_to_ee_transform(config):
    """Confirm the board-to-EE mount transform from config and build the matrix."""
    translation_m = tuple(config["bird_eye_d435"]["board_to_ee_translation_m"])
    rpy_deg = tuple(config["bird_eye_d435"]["board_to_ee_rpy_deg"])

    print("\n" + "=" * 70)
    print("  MOUNTED BOARD TO END-EFFECTOR TRANSFORM")
    print("=" * 70)
    print("""
  The D435 sees board -> camera. The robot reports wrist -> base.
  To connect them, we need board -> wrist (T_board_to_ee).

  These values are read from config.py. Use all zeros only if the board
  coordinate frame is intentionally identical to the robot end-effector frame.
""")

    T_board_to_ee = make_board_to_ee_transform(translation_m, rpy_deg)
    identity = all(abs(v) < 1e-12 for v in translation_m + rpy_deg)

    print("  Config values:")
    print(f"    board_to_ee_translation_m = {translation_m} meters")
    print(f"    board_to_ee_rpy_deg       = {rpy_deg} degrees")
    print("\n  Board -> end-effector transform:")
    for row in T_board_to_ee:
        print(f"    [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f} {row[3]:8.4f}]")

    while True:
        choice = input("  Are these config values correct? [y/n]: ").strip().lower()
        if choice in ("y", "yes"):
            break
        if choice in ("n", "no"):
            raise RuntimeError(
                "Update bird_eye_d435.board_to_ee_translation_m and "
                "board_to_ee_rpy_deg in calibration_config.yaml before calibrating."
            )
        print("  Please answer y or n.")

    mount_metadata = {
        "source": "calibration_config.yaml",
        "translation_m": list(translation_m),
        "rpy_deg": list(rpy_deg),
        "identity_asserted": identity,
    }
    return T_board_to_ee, mount_metadata


def draw_text(image, text, position, color, scale=0.65, thickness=2):
    """Draw readable overlay text with a black outline for camera previews."""
    cv2.putText(image, text, (position[0] + 1, position[1] + 1),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(image, text, position,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_preview(display, detection, captured, required):
    """Draw operator status overlay on the camera preview."""
    detected = detection["success"]
    marker_count = detection.get("marker_count", 0)
    charuco_count = detection.get("charuco_count", 0)

    if detected:
        status = f"CHARUCO READY: {charuco_count} corners"
        color = (0, 255, 0)
    elif marker_count:
        status = f"PARTIAL BOARD: {marker_count} markers, {charuco_count} corners"
        color = (0, 220, 255)
    else:
        status = "Searching for mounted ChArUco board..."
        color = (0, 0, 255)

    draw_text(display, status, (20, 40), color, scale=0.7)
    draw_text(display, "Press 's' to capture this pose", (20, 75), (0, 255, 0), scale=0.65)
    draw_text(display, f"Captured: {captured}/{required}",
              (20, display.shape[0] - 20), (255, 255, 255), scale=0.65)


def run_calibration(config, camera_side):
    """Run the D435 bird's-eye calibration for the specified camera."""
    # Get per-camera config
    camera_config = config["bird_eye_d435"][camera_side]
    cam_serial = camera_config["serial"]
    output_path = camera_config["output_path"]
    poses_required = camera_config["poses_required"]
    min_corners = camera_config["min_charuco_corners"]

    # Validate config
    if not cam_serial or cam_serial == "":
        print(f"ERROR: D435 '{camera_side}' serial not configured. "
              f"Set bird_eye_d435.{camera_side}.serial in calibration_config.yaml")
        return 1

    print("=" * 70)
    print(f"  D435 BIRD'S-EYE EXTRINSIC CALIBRATION ({camera_side.upper()})")
    print("=" * 70)
    print(f"  Camera:         D435 ({camera_side})")
    print(f"  Serial:         {cam_serial}")
    print(f"  Resolution:     {config['d435_resolution'][0]}x{config['d435_resolution'][1]}")
    print(f"  Poses required: {poses_required}")
    print(f"  Min corners:    {min_corners}")

    # Load intrinsics
    intrinsics = load_intrinsics(config["intrinsics_d435_path"])
    if intrinsics is None:
        print(f"ERROR: D435 intrinsics not found at {config['intrinsics_d435_path']}")
        print("Run intrinsic calibration first.")
        return 1
    camera_matrix, dist_coeffs = get_opencv_matrices(intrinsics)
    print(f"  Intrinsics loaded from {config['intrinsics_d435_path']}")

    # Confirm board-to-EE mount
    try:
        T_board_to_ee, mount_metadata = confirm_board_to_ee_transform(config)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    # Connect to robot
    print("\nConnecting to Franka robot...")
    robot = connect_robot(config)
    print("Robot connected.")

    # Start D435 camera
    print("Starting D435 camera...")
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_device(cam_serial)
    res = tuple(config["d435_resolution"])
    rs_config.enable_stream(rs.stream.depth, res[0], res[1], rs.format.z16, config["camera_fps"])
    rs_config.enable_stream(rs.stream.color, res[0], res[1], rs.format.bgr8, config["camera_fps"])
    pipeline.start(rs_config)
    time.sleep(config["camera_warmup_seconds"])
    print(f"D435 started at {res[0]}x{res[1]}")

    # Prepare ChArUco board
    aruco_dict = get_aruco_dictionary(cv2, config["charuco_aruco_dict_name"])
    charuco_board = create_charuco_board(
        cv2,
        tuple(config["charuco_board_inner_corners"]),
        config["charuco_square_size_m"],
        config["charuco_marker_size_m"],
        aruco_dict,
        legacy_pattern=config["charuco_legacy_pattern"],
    )

    camera_to_base_estimates = []
    captured_count = 0
    window_name = f"D435 Bird's-Eye Calibration ({camera_side})"

    print("\n" + "=" * 70)
    print("  CAPTURE PHASE")
    print("=" * 70)
    print("""
  SETUP:
    - The D435 is fixed above the workspace, looking down.
    - The ChArUco board is rigidly mounted to the end-effector.
    - Move the robot so the D435 can see the board clearly from many angles.

  INSTRUCTIONS:
    - Move the robot to varied positions and wrist orientations.
    - Watch the preview: green means the mounted board is recognized.
    - Press 's' when the board is detected and the robot is still.
    - Capture {} poses.
    - Press 'q' to quit early after at least 3 good poses.

  TIPS:
    - Use poses across the workspace the D435 will actually observe.
    - Avoid only translating in a straight line; vary orientation too.
    - Discard any capture where the board is blurred or partly hidden.
""".format(poses_required))
    print("=" * 70 + "\n")

    try:
        while captured_count < poses_required:
            frames = pipeline.wait_for_frames(timeout_ms=config["camera_frame_timeout_ms"])
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # Detect ChArUco board
            detection = detect_charuco_board_pose(
                cv2, gray, aruco_dict, charuco_board,
                camera_matrix, dist_coeffs,
                min_corners=min_corners,
            )
            detected = detection["success"]
            T_board_to_cam = detection["T_board_to_cam"]

            # Draw preview
            display = color_image.copy()
            draw_charuco_detection(cv2, display, detection)
            if detected:
                draw_pose_axes(cv2, display, T_board_to_cam, camera_matrix, dist_coeffs, 0.08)
            draw_preview(display, detection, captured_count, poses_required)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            # Capture pose
            if key == ord("s"):
                if not detected:
                    print("  ChArUco board not detected; capture skipped.")
                    continue

                print(f"  [{captured_count + 1}/{poses_required}] Capturing pose...")

                # Record synchronized pose pair
                T_ee_to_base = get_robot_ee_pose(robot)
                T_cam_to_board = np.linalg.inv(T_board_to_cam)
                T_cam_to_base = T_ee_to_base @ T_board_to_ee @ T_cam_to_board
                camera_to_base_estimates.append(T_cam_to_base)
                captured_count += 1
                print("      Pose saved")

            elif key == ord("q"):
                print("\n  Quitting early...")
                break

        if captured_count < 3:
            print(f"\nERROR: Need at least 3 poses, only captured {captured_count}.")
            return 1

        # Compute averaged transform
        print(f"\nComputing fixed-camera transform from {captured_count} poses...")
        T_cam_to_base = average_transforms(camera_to_base_estimates)
        validation = validate_transform_set(camera_to_base_estimates, T_cam_to_base)

        print("\n  Transform matrix (D435 camera -> robot base):")
        for row in T_cam_to_base:
            print(f"    [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f} {row[3]:8.4f}]")

        print("\n  Per-pose agreement:")
        print(f"    Max translation spread: {validation['max_translation_error_m'] * 1000:.2f} mm")
        print(f"    Mean translation spread: {validation['mean_translation_error_m'] * 1000:.2f} mm")
        print(f"    Max rotation spread:    {validation['max_rotation_error_deg']:.3f} deg")
        print(f"    Mean rotation spread:   {validation['mean_rotation_error_deg']:.3f} deg")

        # Save result
        output_path = Path(output_path)
        save_fixed_camera_transform(
            str(output_path),
            T_cam_to_base,
            cam_serial,
            board_to_ee=T_board_to_ee,
            mount_metadata=mount_metadata,
            validation_results=validation,
        )

        print("\n" + "=" * 70)
        print("  CALIBRATION COMPLETE")
        print("=" * 70)
        print(f"\n  File saved: {output_path}")
        print("  Runtime D435 depth points can now transform directly to robot base.")

    except Exception as exc:
        print(f"\nERROR: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        robot.stop()
        print("\nCleaned up camera and robot resources.")

    return 0


def main():
    config_path = Path(__file__).parent / "calibration_config.yaml"
    config = load_config(config_path)
    args = parse_args()
    return run_calibration(config, args.camera)


if __name__ == "__main__":
    sys.exit(main())
