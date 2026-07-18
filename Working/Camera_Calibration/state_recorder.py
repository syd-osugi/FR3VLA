#!/usr/bin/env python3
"""
State Recorder: Save and Recall Franka Robot States During Calibration
======================================================================

Interactive tool for recording Franka robot joint states and end-effector poses
while viewing a live camera preview. Used during extrinsic calibration to:

  1. Position the robot at viewpoints where the ChArUco board is clearly visible
  2. Save those states for later recall or export
  3. Recall saved states to return to known-good viewpoints
  4. Export saved poses to JSON for use with calibrate_d405_hand_eye.py --pose-plan

Workflow:
  python state_recorder.py --camera d405
  python state_recorder.py --camera left
  python state_recorder.py --camera right

Keyboard controls in the camera preview window:
  s: Save current robot state (prompts for label)
  r: Recall a saved pose to the robot (prompts for index)
  d: Delete a saved pose (prompts for index)
  e: Export saved poses to JSON file
  q: Quit and save the pose list to disk

Per-camera output files (saved to calibration_data/):
  d405  -> d405_poses.json
  left  -> d435_left_poses.json
  right -> d435_right_poses.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
WORKING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKING_DIR not in sys.path:
    sys.path.insert(0, WORKING_DIR)

import numpy as np
import yaml

# Import calibration math module
from calibration_math import (
    create_charuco_board,
    detect_charuco_board_pose,
    draw_charuco_detection,
    draw_pose_axes,
    get_aruco_dictionary,
    get_opencv_matrices,
    load_intrinsics,
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


# Pose file naming per camera
POSE_FILE_MAP = {
    "d405": "d405_poses.json",
    "left": "d435_left_poses.json",
    "right": "d435_right_poses.json",
}


def load_config(config_path):
    """Load calibration_config.yaml and return the dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Save and recall Franka robot states during camera calibration."
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
        choices=["d405", "left", "right"],
        help="Which camera to use for preview: 'd405', 'left', or 'right'.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save/load poses JSON file. Default: per-camera default in calibration_data/.",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Load an existing pose file instead of starting empty.",
    )
    parser.add_argument(
        "--move-speed-mps",
        type=float,
        default=0.025,
        help="Cartesian translation speed for recalling poses. Default: 0.025.",
    )
    parser.add_argument(
        "--move-rot-speed-radps",
        type=float,
        default=0.25,
        help="Approximate rotation speed for recalling poses. Default: 0.25.",
    )
    parser.add_argument(
        "--settle-s",
        type=float,
        default=2.0,
        help="Seconds to wait after recalling a pose. Default: 2.0.",
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


def get_robot_joint_state(robot):
    """Read current joint positions from the Franka."""
    state = robot.read_once()
    return list(state.q)


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


def move_to_pose(robot, target_pose, move_speed_mps, move_rot_speed_radps):
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


def draw_preview(display, detection, robot_pose, pose_count):
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
    draw_text(display, f"Saved poses: {pose_count}  |  s:save r:recall d:delete e:export q:quit",
              (20, display.shape[0] - 20), (255, 255, 255), scale=0.55)


def save_pose_list(poses, output_path):
    """Save the current pose list to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "Franka robot poses recorded during camera calibration",
        "saved_at": datetime.now().isoformat(),
        "pose_count": len(poses),
        "poses": poses,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=4)
    print(f"  Poses saved to {output_path}")


def load_pose_list(path):
    """Load a pose list from a JSON file. Returns list of pose dicts."""
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("poses", [])


def run_state_recorder(config, camera, output_path, load_path, move_speed_mps, move_rot_speed_radps, settle_s):
    """Run the interactive state recorder loop."""
    # Determine camera config
    if camera == "d405":
        cam_serial = config["d405_serial"]
        resolution = tuple(config["d405_resolution"])
        intrinsics_path = config["intrinsics_d405_path"]
    elif camera == "left":
        cam_serial = config["bird_eye_d435"]["left"]["serial"]
        resolution = tuple(config["d435_resolution"])
        intrinsics_path = config["intrinsics_d435_path"]
    elif camera == "right":
        cam_serial = config["bird_eye_d435"]["right"]["serial"]
        resolution = tuple(config["d435_resolution"])
        intrinsics_path = config["intrinsics_d435_path"]

    if not cam_serial or cam_serial == "":
        print(f"ERROR: {camera.upper()} camera serial not configured.")
        return 1

    # Default output path
    if output_path is None:
        output_path = Path(__file__).parent / "calibration_data" / POSE_FILE_MAP[camera]
    else:
        output_path = Path(output_path)

    print("=" * 70)
    print(f"  STATE RECORDER ({camera.upper()})")
    print("=" * 70)
    print(f"  Camera:         {camera.upper()}")
    print(f"  Serial:         {cam_serial}")
    print(f"  Resolution:     {resolution[0]}x{resolution[1]}")
    print(f"  Output file:    {output_path}")

    # Load intrinsics
    intrinsics = load_intrinsics(intrinsics_path)
    if intrinsics is None:
        print(f"ERROR: Intrinsics not found at {intrinsics_path}")
        print("Run intrinsic calibration first.")
        return 1
    camera_matrix, dist_coeffs = get_opencv_matrices(intrinsics)
    print(f"  Intrinsics loaded from {intrinsics_path}")

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

    # Start camera
    print("Starting camera...")
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_device(cam_serial)
    rs_config.enable_stream(rs.stream.depth, resolution[0], resolution[1], rs.format.z16, config["camera_fps"])
    rs_config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, config["camera_fps"])
    pipeline.start(rs_config)
    time.sleep(config["camera_warmup_seconds"])
    print(f"Camera started at {resolution[0]}x{resolution[1]}")

    # Load existing poses or start empty
    poses = []
    if load_path is not None:
        poses = load_pose_list(load_path)
        print(f"  Loaded {len(poses)} poses from {load_path}")
    elif output_path.exists():
        poses = load_pose_list(output_path)
        print(f"  Loaded {len(poses)} existing poses from {output_path}")

    window_name = f"State Recorder ({camera.upper()})"
    selected_index = None

    print("\n" + "=" * 70)
    print("  STATE RECORDER CONTROLS")
    print("=" * 70)
    print("""
  Move the robot to viewpoints where the ChArUco board is clearly visible.
  Use the live camera preview to verify each viewpoint.

  Keyboard controls (in camera window):
    s: Save current robot state (prompts for label)
    r: Recall a saved pose to the robot (prompts for index)
    d: Delete a saved pose (prompts for index)
    e: Export saved poses to JSON file
    q: Quit and save the pose list

  Tips:
    - Save poses with descriptive labels (e.g., "top_view", "side_view")
    - Verify ChArUco is green before saving
    - Recall poses to re-check viewpoints
""")
    print("=" * 70 + "\n")

    try:
        while True:
            # Wait for frame
            frames = pipeline.wait_for_frames(timeout_ms=config["camera_frame_timeout_ms"])
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            robot_pose = get_robot_ee_pose(robot)

            # Detect ChArUco board
            detection = detect_charuco_board_pose(
                cv2, gray, aruco_dict, charuco_board,
                camera_matrix, dist_coeffs,
                min_corners=6,
            )

            # Draw preview
            display = color_image.copy()
            draw_charuco_detection(cv2, display, detection)
            if detection["success"]:
                draw_pose_axes(cv2, display, detection["T_board_to_cam"], camera_matrix, dist_coeffs, 0.05)
            draw_preview(display, detection, robot_pose, len(poses))

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            # Save pose
            if key == ord("s"):
                label = input("  Enter pose label: ").strip()
                if not label:
                    label = f"pose_{len(poses) + 1}"

                joint_state = get_robot_joint_state(robot)
                pose_record = {
                    "label": label,
                    "timestamp": time.time(),
                    "q": joint_state,
                    "O_T_EE": robot_pose.tolist(),
                    "ee_xyz": robot_pose[:3, 3].tolist(),
                }
                poses.append(pose_record)
                print(f"  Saved pose '{label}' ({len(poses)} total)")

            # Recall pose
            elif key == ord("r"):
                if not poses:
                    print("  No saved poses to recall.")
                    continue
                print(f"  Saved poses ({len(poses)} total):")
                for i, p in enumerate(poses):
                    marker = " > " if i == selected_index else "   "
                    print(f"    {marker}[{i}] {p['label']}: xyz={p['ee_xyz']}")

                idx_str = input("  Enter pose index to recall (or 'c' to cancel): ").strip()
                if idx_str.lower() == "c":
                    continue
                try:
                    idx = int(idx_str)
                    if 0 <= idx < len(poses):
                        target_pose = np.array(poses[idx]["O_T_EE"], dtype=float)
                        current_xyz = robot_pose[:3, 3].tolist()
                        target_xyz = target_pose[:3, 3].tolist()
                        dist = float(np.linalg.norm(np.array(target_xyz) - np.array(current_xyz)))
                        print(f"  Moving to '{poses[idx]['label']}' (xyz distance: {dist:.4f} m)...")
                        move_to_pose(robot, target_pose, move_speed_mps, move_rot_speed_radps)
                        if settle_s > 0.0:
                            print(f"  Waiting {settle_s:.2f}s for settling...")
                            time.sleep(settle_s)
                        print("  Move complete.")
                    else:
                        print(f"  Invalid index. Enter 0-{len(poses) - 1}.")
                except ValueError:
                    print("  Invalid input.")

            # Delete pose
            elif key == ord("d"):
                if not poses:
                    print("  No saved poses to delete.")
                    continue
                print(f"  Saved poses ({len(poses)} total):")
                for i, p in enumerate(poses):
                    marker = " > " if i == selected_index else "   "
                    print(f"    {marker}[{i}] {p['label']}: xyz={p['ee_xyz']}")

                idx_str = input("  Enter pose index to delete (or 'c' to cancel): ").strip()
                if idx_str.lower() == "c":
                    continue
                try:
                    idx = int(idx_str)
                    if 0 <= idx < len(poses):
                        removed = poses.pop(idx)
                        if selected_index is not None and selected_index >= len(poses):
                            selected_index = len(poses) - 1 if poses else None
                        print(f"  Deleted pose '{removed['label']}' ({len(poses)} remaining)")
                    else:
                        print(f"  Invalid index. Enter 0-{len(poses) - 1}.")
                except ValueError:
                    print("  Invalid input.")

            # Export poses
            elif key == ord("e"):
                save_pose_list(poses, output_path)

            # Quit
            elif key in {ord("q"), 27}:
                save_pose_list(poses, output_path)
                print(f"\n  Quitting. {len(poses)} poses saved to {output_path}")
                break

    except KeyboardInterrupt:
        print("\n  Interrupted.")
        save_pose_list(poses, output_path)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        import traceback
        traceback.print_exc()
        save_pose_list(poses, output_path)
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
    return run_state_recorder(
        config,
        args.camera,
        args.output,
        args.load,
        args.move_speed_mps,
        args.move_rot_speed_radps,
        args.settle_s,
    )


if __name__ == "__main__":
    sys.exit(main())
