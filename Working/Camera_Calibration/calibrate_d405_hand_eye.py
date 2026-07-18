#!/usr/bin/env python3
"""
Script: D405 Eye-in-Hand Extrinsic Calibration
================================================
Finds the rigid transform between the wrist-mounted D405 camera and the
Franka end-effector frame using a ChArUco board fixed on the table.

PHYSICAL SETUP:
  - D405 rigidly mounted to the robot end-effector
  - ChArUco board flat on the table in the workspace (does not move)
  - Robot moves the D405 around the board

CALIBRATED TRANSFORM:
  T_d405_to_ee: maps a point from D405 optical frame -> end-effector frame
    p_ee = T_d405_to_ee @ p_d405

WORKFLOW:
  Manual mode (default):
    python calibrate_d405_hand_eye.py
    Move the robot by hand or command, press 's' at each good pose.

  Planned-pose mode:
    python calibrate_d405_hand_eye.py --pose-plan /path/to/pose_plan.json
    Robot auto-moves through saved poses; press 's' at each green view.

RUNTIME USE:
  After calibration, any D405 depth point p_d405 can be converted to the
  robot base frame via:
    p_base = T_ee_to_base @ T_d405_to_ee @ p_d405

OUTPUT:
  JSON file saved to calibration_data/d405_to_ee.json containing:
    - The 4x4 T_d405_to_ee matrix
    - Validation metrics (rotation/translation error across poses)
    - Camera serial number for future reference
"""

import argparse
import json
import os
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
    calibrate_hand_eye,
    create_charuco_board,
    detect_charuco_board_pose,
    draw_charuco_detection,
    draw_pose_axes,
    get_aruco_dictionary,
    get_opencv_matrices,
    load_intrinsics,
    require_charuco_support,
    save_hand_eye,
    validate_hand_eye_result,
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
    """Parse command-line arguments for manual or planned-pose calibration."""
    parser = argparse.ArgumentParser(
        description="Calibrate the D405 eye-in-hand camera extrinsics using a ChArUco board."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to calibration_config.yaml. Default: calibration_config.yaml in this directory.",
    )
    parser.add_argument(
        "--pose-plan",
        nargs="?",
        const=None,
        default=None,
        metavar="PATH",
        help=(
            "Path to a JSON pose plan file. If no path given, uses "
            "calibration_data/d405_hand_eye_pose_plan.json in this directory."
        ),
    )
    parser.add_argument(
        "--min-corners",
        type=int,
        default=15,
        help="Minimum ChArUco corners required before capture is allowed. Default: 15.",
    )
    parser.add_argument(
        "--move-speed-mps",
        type=float,
        default=0.025,
        help="Cartesian translation speed for planned-pose moves. Default: 0.025.",
    )
    parser.add_argument(
        "--move-rot-speed-radps",
        type=float,
        default=0.25,
        help="Approximate rotation speed for planned-pose moves. Default: 0.25.",
    )
    parser.add_argument(
        "--settle-s",
        type=float,
        default=2.0,
        help="Seconds to wait after each planned-pose move. Default: 2.0.",
    )
    parser.add_argument(
        "--skip-motion-confirmation",
        action="store_true",
        help="Do not require typing 'move' before each planned-pose move.",
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

    # Apply collision behavior
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


def matrix_to_franka_pose(matrix):
    """Convert a 4x4 matrix to Franka's flat column-major pose format."""
    return np.array(matrix, dtype=float).reshape(16, order="F").tolist()


def duration_to_seconds(duration):
    """Convert pylibfranka duration objects or numeric values to seconds."""
    to_sec = getattr(duration, "to_sec", None)
    if callable(to_sec):
        return float(to_sec())
    return float(duration)


def smoothstep(alpha):
    """Smooth 0-to-1 interpolation curve for gentle robot motion."""
    bounded = max(0.0, min(1.0, float(alpha)))
    return bounded * bounded * (3.0 - 2.0 * bounded)


def rotation_angle(rotation):
    """Return the rotation magnitude in radians for a 3x3 rotation matrix."""
    trace = np.trace(rotation)
    return float(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))


def interpolate_rotation(start_rotation, target_rotation, alpha):
    """Interpolate orientation between two rotations using Rodrigues vectors."""
    rotation_delta = start_rotation.T @ target_rotation
    rvec_delta, _ = cv2.Rodrigues(rotation_delta)
    partial_delta, _ = cv2.Rodrigues(rvec_delta * float(alpha))
    return start_rotation @ partial_delta


def move_to_full_pose(robot, target_pose, move_speed_mps, move_rot_speed_radps):
    """Command the robot to a saved full T_ee_to_base pose with eased motion."""
    start_pose = get_robot_ee_pose(robot)
    translation_distance = float(np.linalg.norm(target_pose[:3, 3] - start_pose[:3, 3]))
    angle_rad = rotation_angle(start_pose[:3, :3].T @ target_pose[:3, :3])
    duration_s = max(
        translation_distance / move_speed_mps,
        angle_rad / move_rot_speed_radps,
        0.5,
    )

    active_control = robot.start_cartesian_pose_control(
        pylibfranka.ControllerMode.JointImpedance
    )
    elapsed_s = 0.0
    motion_finished = False

    while not motion_finished:
        _, duration = active_control.readOnce()
        elapsed_s += max(0.0, duration_to_seconds(duration))
        alpha = min(1.0, elapsed_s / duration_s)
        eased = smoothstep(alpha)

        command_pose = np.eye(4, dtype=float)
        command_pose[:3, :3] = interpolate_rotation(
            start_pose[:3, :3], target_pose[:3, :3], eased
        )
        command_pose[:3, 3] = (
            start_pose[:3, 3] + eased * (target_pose[:3, 3] - start_pose[:3, 3])
        )

        cartesian_pose = pylibfranka.CartesianPose(matrix_to_franka_pose(command_pose))
        if alpha >= 1.0:
            cartesian_pose.motion_finished = True
            motion_finished = True

        active_control.writeOnce(cartesian_pose)


def draw_text(image, text, position, color, scale=0.65, thickness=2):
    """Draw readable overlay text with a black outline for camera previews."""
    cv2.putText(image, text, (position[0] + 1, position[1] + 1),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(image, text, position,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_preview(display, detection, robot_pose, captured, required, planned_index=None, planned_total=None):
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
        status = "Searching for ChArUco board..."
        color = (0, 0, 255)

    xyz = robot_pose[:3, 3]
    draw_text(display, status, (20, 40), color, scale=0.7)
    draw_text(display, f"EE xyz [m]: x={xyz[0]:+.3f} y={xyz[1]:+.3f} z={xyz[2]:+.3f}",
              (20, 75), (255, 255, 255))
    draw_text(display, f"Captured: {captured}/{required}",
              (20, display.shape[0] - 20), (255, 255, 255), scale=0.65)

    if planned_index is not None and planned_total is not None:
        draw_text(display, f"Planned pose: {planned_index + 1}/{planned_total}  |  n: skip",
                  (20, display.shape[0] - 55), (255, 255, 255), scale=0.6)


def run_calibration(config, args):
    """Run the D405 eye-in-hand calibration loop."""
    d405_config = config["hand_eye_d405"]
    cam_config = config["d405_serial"]
    resolution = tuple(config["d405_resolution"])

    # Validate config
    if not cam_config or cam_config == "":
        print("ERROR: D405 serial not configured. Set d405_serial in calibration_config.yaml")
        return 1

    print("=" * 70)
    print("  D405 EYE-IN-HAND EXTRINSIC CALIBRATION")
    print("=" * 70)
    print(f"  D405 serial:  {cam_config}")
    print(f"  Resolution:   {resolution[0]}x{resolution[1]}")
    print(f"  Min corners:  {args.min_corners}")

    # Load intrinsics
    intrinsics = load_intrinsics(config["intrinsics_d405_path"])
    if intrinsics is None:
        print(f"ERROR: D405 intrinsics not found at {config['intrinsics_d405_path']}")
        print("Run intrinsic calibration first.")
        return 1
    camera_matrix, dist_coeffs = get_opencv_matrices(intrinsics)
    print(f"  Intrinsics loaded from {config['intrinsics_d405_path']}")

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

    # Connect to robot
    print("\nConnecting to Franka robot...")
    robot = connect_robot(config)
    print("Robot connected.")

    # Start D405 camera
    print("Starting D405 camera...")
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_device(cam_config)
    rs_config.enable_stream(rs.stream.depth, resolution[0], resolution[1], rs.format.z16, config["camera_fps"])
    rs_config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, config["camera_fps"])
    pipeline.start(rs_config)
    time.sleep(config["camera_warmup_seconds"])
    print(f"D405 started at {resolution[0]}x{resolution[1]}")

    # Pose plan mode
    planned_poses = None
    pose_plan_path = None
    if args.pose_plan is not None:
        plan_path = Path(args.pose_plan)
        if not plan_path.exists():
            # Try relative to this script's directory
            plan_path = Path(__file__).parent / args.pose_plan
        if not plan_path.exists():
            print(f"ERROR: Pose plan not found at {plan_path}")
            return 1
        with open(plan_path, "r") as f:
            plan_data = json.load(f)
        planned_poses = [np.array(p["T_ee_to_base"], dtype=float) for p in plan_data["poses"]]
        pose_plan_path = str(plan_path)
        print(f"  Pose plan loaded: {len(planned_poses)} poses from {pose_plan_path}")

    poses_required = len(planned_poses) if planned_poses else d405_config["poses_required"]

    robot_poses = []
    camera_poses = []
    captured_count = 0
    planned_index = 0
    planned_pose_ready = planned_poses is None

    window_name = "D405 Eye-in-Hand Calibration"

    print("\n" + "=" * 70)
    print("  CAPTURE PHASE")
    print("=" * 70)
    print("""
  SETUP:
    - ChArUco board is fixed on the table in the workspace.
    - The D405 is mounted on the robot end-effector and moves with it.
    - Board inner corners: {}
    - Square size: {:.4f} m, Marker size: {:.4f} m

  INSTRUCTIONS:
    - Move the robot to varied positions and wrist orientations.
    - Keep the ChArUco board visible in the D405 image.
    - Green status = board detected well enough.
    - Press 's' when the board is detected and the robot is still.
    - Capture {} poses.
    - Press 'q' to quit early after at least 3 good poses.

  TIPS:
    - Use strong variation in rotation and translation.
    - Avoid only sliding the wrist in a straight line.
    - Discard blurred or partially hidden detections.
""".format(
        config["charuco_board_inner_corners"],
        config["charuco_square_size_m"],
        config["charuco_marker_size_m"],
        poses_required,
    ))

    if planned_poses is not None:
        print("""
  PLANNED-POSE MODE:
    - The robot will move through your saved full EE poses.
    - Type 'move' in the terminal before each planned motion.
    - After each move, use the D405 preview and press 's' when green.
    - Press 'n' in the preview to skip a planned pose.
""")
    print("=" * 70 + "\n")

    try:
        while captured_count < poses_required:
            # Move to next planned pose if in planned mode
            if planned_poses is not None and not planned_pose_ready:
                if planned_index >= len(planned_poses):
                    print("\nNo more planned poses.")
                    break
                target_pose = planned_poses[planned_index]
                current_pose = get_robot_ee_pose(robot)
                translation_distance = float(np.linalg.norm(target_pose[:3, 3] - current_pose[:3, 3]))
                angle_deg = np.degrees(rotation_angle(current_pose[:3, :3].T @ target_pose[:3, :3]))

                if not args.skip_motion_confirmation:
                    print(f"\nMoving to planned pose {planned_index + 1}/{len(planned_poses)}")
                    print(f"  Translation: {translation_distance:.4f} m, Rotation: {angle_deg:.2f} deg")
                    answer = input("  Type 'move' to execute: ").strip().lower()
                    if answer != "move":
                        planned_index += 1
                        planned_pose_ready = False
                        continue

                print(f"  Moving to pose {planned_index + 1}/{len(planned_poses)}...")
                move_to_full_pose(robot, target_pose, args.move_speed_mps, args.move_rot_speed_radps)
                if args.settle_s > 0.0:
                    print(f"  Waiting {args.settle_s:.2f}s for settling...")
                    time.sleep(args.settle_s)
                print("  Move complete. Press 's' when ChArUco is green.")
                planned_pose_ready = True

            # Wait for frame
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
                min_corners=args.min_corners,
            )
            detected = detection["success"]
            T_board_to_cam = detection["T_board_to_cam"]

            # Draw preview
            display = color_image.copy()
            draw_charuco_detection(cv2, display, detection)
            if detected:
                draw_pose_axes(cv2, display, T_board_to_cam, camera_matrix, dist_coeffs, 0.05)
            draw_preview(display, detection, get_robot_ee_pose(robot),
                         captured_count, poses_required,
                         planned_index if planned_poses else None,
                         len(planned_poses) if planned_poses else None)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            # Capture pose
            if key == ord("s"):
                if not detected:
                    print("  ChArUco not detected; capture skipped.")
                    continue
                print(f"  [{captured_count + 1}/{poses_required}] Capturing pose...")
                T_ee_to_base = get_robot_ee_pose(robot)
                robot_poses.append(T_ee_to_base)
                camera_poses.append(T_board_to_cam)
                captured_count += 1
                print("      Pose saved")
                if planned_poses is not None:
                    planned_index += 1
                    planned_pose_ready = False

            # Skip planned pose
            elif key == ord("n") and planned_poses is not None:
                print(f"  Skipping planned pose {planned_index + 1}/{len(planned_poses)}")
                planned_index += 1
                planned_pose_ready = False

            # Quit
            elif key in {ord("q"), 27}:
                print("\n  Quitting early...")
                break

        if captured_count < 3:
            print(f"\nERROR: Need at least 3 poses, only captured {captured_count}.")
            return 1

        # Compute calibration
        print("\n" + "=" * 70)
        print("  COMPUTING CALIBRATION")
        print("=" * 70)
        print(f"  Using {captured_count} poses...")

        T_cam_to_ee = calibrate_hand_eye(robot_poses, camera_poses)
        is_valid, rot_err, trans_err = validate_hand_eye_result(T_cam_to_ee, robot_poses, camera_poses)

        print("\n  Results:")
        print(f"    Max rotation error:    {rot_err:.3f} deg")
        print(f"    Max translation error: {trans_err * 1000:.2f} mm")
        print(f"    Validation:            {'PASS' if is_valid else 'FAIL'}")

        print("\n  Transform matrix (D405 camera -> end-effector):")
        for row in T_cam_to_ee:
            print(f"    [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f} {row[3]:8.4f}]")

        # Save result
        output_path = Path(config["hand_eye_d405"]["output_path"])
        save_hand_eye(str(output_path), T_cam_to_ee, cam_config, (is_valid, rot_err, trans_err))

        print("\n" + "=" * 70)
        print("  CALIBRATION COMPLETE")
        print("=" * 70)
        print(f"\n  File saved: {output_path}")
        if not is_valid:
            print("\n  WARNING: Validation failed. Consider recapturing with more varied views.")

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
    return run_calibration(config, args)


if __name__ == "__main__":
    sys.exit(main())
