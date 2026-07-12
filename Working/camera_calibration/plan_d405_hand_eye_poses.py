#!/usr/bin/env python3
"""
Script: D405 Hand-Eye Pose Planner
----------------------------------
Shows the same D405 ChArUco preview used during hand-eye calibration while you
build a reusable list of Franka end-effector poses.

WHY THIS SCRIPT EXISTS:
=======================
Hand-guiding the Franka during D405 hand-eye calibration can make it hard to
capture repeatable, settled poses. This helper lets you separate the process into
two calmer steps:

  1. Plan/save calibration poses while watching the D405 ChArUco preview.
  2. Re-run the actual extrinsic calibration using those saved poses as
     commanded robot targets.

The saved poses are full T_ee_to_base matrices, not just XYZ positions, because
D405 hand-eye calibration needs varied wrist orientations. If you replay only
XYZ positions while preserving one wrist orientation, the hand-eye solve can be
weak or degenerate.

HOW TO CREATE A POSE PLAN:
==========================
Run:

    python Working/camera_calibration/plan_d405_hand_eye_poses.py

Then use the camera window:

  s: save current full EE pose when the ChArUco status is green
  p: force-save current full EE pose even when the status is not green
  n: select the next saved pose
  b: select the previous saved pose
  g: move the robot to the selected saved full pose
  u: undo the most recently saved pose
  x: delete the selected saved pose
  q: save the JSON file and quit

By default, the JSON is saved to:

    Working/camera_calibration/calibration_data/d405_hand_eye_pose_plan.json

You can choose a different path:

    python Working/camera_calibration/plan_d405_hand_eye_poses.py \
        --output /path/to/my_pose_plan.json

You can clear an old plan and start fresh:

    python Working/camera_calibration/plan_d405_hand_eye_poses.py --replace

HOW TO USE THE SAVED PLAN FOR CALIBRATION:
==========================================
After saving the plan, run:

    python Working/camera_calibration/run_extrinsics_d405_hand_eye.py --pose-plan

or, for a custom path:

    python Working/camera_calibration/run_extrinsics_d405_hand_eye.py \
        --pose-plan /path/to/my_pose_plan.json

The calibration script will move to each saved full pose, wait for settling,
show the same D405 ChArUco preview, and let you press 's' when the board view is
stable and green.

WHAT A GOOD PLAN LOOKS LIKE:
============================
Use about 12-20 poses for real calibration. Six is only a quick diagnostic.
Good pose sets include:

  - different wrist pitch/yaw/roll angles,
  - different heights above the board,
  - different XY positions,
  - sharp views where the board is not near the edge of the image,
  - many detected ChArUco corners.

Keep the ChArUco board fixed in the workspace while planning and calibration.
For D405 hand-eye calibration, the camera moves with the wrist and the board
does not move.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

# Add parent directory to path for imports when this script is run directly.
WORKING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKING_DIR not in sys.path:
    sys.path.insert(0, WORKING_DIR)

import config as cfg
from robot.franka_setup import LOAD_PROFILE_D405_CALIBRATION, apply_franka_control_config
from camera_calibration.charuco_utils import (
    create_charuco_board,
    detect_charuco_board_pose,
    draw_charuco_detection,
    draw_pose_axes,
    get_aruco_dictionary,
    require_charuco_support,
)
from camera_calibration.intrinsics_math import get_opencv_matrices, load_intrinsics

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


SCRIPT_DIR = Path(__file__).resolve().parent

# The planner and run_extrinsics_d405_hand_eye.py share this path by default.
# That keeps the two-step workflow simple:
#   1. Save poses here with this script.
#   2. Run the calibration script with --pose-plan and no extra path argument.
DEFAULT_POSE_PLAN_PATH = SCRIPT_DIR / "calibration_data" / "d405_hand_eye_pose_plan.json"

# OpenCV waitKey returns integer key codes. ESC is 27.
QUIT_KEYS = {ord("q"), ord("Q"), 27}


def parse_args():
    """
    Parse operator controls for pose planning and saved-pose replay.

    The movement speed flags affect only the optional 'g' key, which moves back
    to a selected saved pose. They do not affect hand-guiding or saving poses.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Preview the D405 ChArUco target and save reusable Franka poses for "
            "D405 hand-eye calibration."
        )
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_POSE_PLAN_PATH),
        help="Pose-plan JSON path. Default: Working/camera_calibration/calibration_data/d405_hand_eye_pose_plan.json",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Start with an empty pose list even if --output already exists.",
    )
    parser.add_argument(
        "--min-corners",
        type=int,
        default=15,
        help="Minimum ChArUco corners required for green ready status. Default: 15.",
    )
    parser.add_argument(
        "--move-speed-mps",
        type=float,
        default=0.025,
        help="Cartesian translation speed for moving to saved poses. Default: 0.025.",
    )
    parser.add_argument(
        "--move-rot-speed-radps",
        type=float,
        default=0.25,
        help="Approximate rotation speed for moving to saved poses. Default: 0.25.",
    )
    parser.add_argument(
        "--settle-s",
        type=float,
        default=2.0,
        help="Seconds to wait after a commanded move. Default: 2.0.",
    )
    parser.add_argument(
        "--skip-motion-confirmation",
        action="store_true",
        help="Do not require typing 'move' before moving to a saved pose.",
    )
    return parser.parse_args()


def require_runtime_dependencies():
    """Fail early with readable errors before opening hardware resources."""
    require_charuco_support(cv2)
    if rs is None:
        raise RuntimeError("pyrealsense2 is required for RealSense preview.")
    if pylibfranka is None:
        raise RuntimeError("pylibfranka is required to read and move the Franka.")


def resolve_existing_config_path(path_value):
    """
    Resolve config paths that may be relative to the repo root or script folder.

    config.py stores calibration paths like calibration_data/d405_intrinsics.json.
    When this script runs from the repo root, that path may be interpreted
    differently than when it runs from Working/camera_calibration. This helper
    accepts either layout and chooses the existing file when possible.
    """
    path = Path(path_value)
    if path.is_absolute() or path.exists():
        return path

    script_relative = SCRIPT_DIR / path
    if script_relative.exists():
        return script_relative

    return path


def franka_pose_to_matrix(pose_values):
    """
    Convert Franka O_T_EE from a flat list into a 4x4 T_ee_to_base matrix.

    Franka exposes O_T_EE as 16 values in column-major order. The output matrix
    maps points written in EE coordinates into the robot base frame.
    """
    pose_array = np.array(pose_values, dtype=float)
    if pose_array.shape != (16,):
        raise ValueError(f"Expected Franka O_T_EE as 16 values, got shape {pose_array.shape}")
    return pose_array.reshape((4, 4), order="F")


def matrix_to_franka_pose(matrix):
    """Convert a 4x4 matrix back to Franka's flat column-major pose format."""
    return np.array(matrix, dtype=float).reshape(16, order="F").tolist()


def get_robot_ee_pose(robot):
    """Read the current Franka end-effector pose as T_ee_to_base."""
    state = robot.read_once()
    return franka_pose_to_matrix(state.O_T_EE)


def duration_to_seconds(duration):
    """Handle pylibfranka duration objects and plain numeric duration values."""
    to_sec = getattr(duration, "to_sec", None)
    if callable(to_sec):
        return float(to_sec())
    return float(duration)


def smoothstep(alpha):
    """
    Return a smooth 0-to-1 interpolation value.

    Smoothstep makes replayed saved-pose motions less abrupt than a raw linear
    interpolation while still reaching the exact target pose.
    """
    bounded = max(0.0, min(1.0, float(alpha)))
    return bounded * bounded * (3.0 - 2.0 * bounded)


def rotation_angle(rotation):
    """Return the rotation magnitude, in radians, for a 3x3 rotation matrix."""
    trace = np.trace(rotation)
    return float(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))


def interpolate_rotation(start_rotation, target_rotation, alpha):
    """
    Interpolate orientation between two saved wrist rotations.

    We use Rodrigues vectors so the commanded orientation changes gradually while
    replaying a full saved pose. This is important because saved calibration
    poses include orientation, not just XYZ.
    """
    rotation_delta = start_rotation.T @ target_rotation
    rvec_delta, _ = cv2.Rodrigues(rotation_delta)
    partial_delta, _ = cv2.Rodrigues(rvec_delta * float(alpha))
    return start_rotation @ partial_delta


def validate_motion_args(args):
    """
    Validate command-line settings before any hardware starts moving.

    These checks catch invalid numbers and prevent a typo like --distance inf.
    They are not a substitute for physical clearance, collision behavior, or the
    Franka user stop button.
    """
    if args.min_corners < 6:
        raise ValueError("--min-corners must be at least 6 for pose estimation")
    if not np.isfinite(args.move_speed_mps) or args.move_speed_mps <= 0.0:
        raise ValueError("--move-speed-mps must be finite and positive")
    if not np.isfinite(args.move_rot_speed_radps) or args.move_rot_speed_radps <= 0.0:
        raise ValueError("--move-rot-speed-radps must be finite and positive")
    if not np.isfinite(args.settle_s) or args.settle_s < 0.0:
        raise ValueError("--settle-s must be finite and non-negative")


def validate_workspace_target(target_pose):
    """
    Ensure a selected saved pose is still inside configured workspace bounds.

    This protects against replaying a stale or hand-edited JSON file after the
    workspace limits in config.py have changed.
    """
    target_xyz = np.array(target_pose[:3, 3], dtype=float)
    workspace_min = np.array(cfg.ROBOT_WORKSPACE_MIN_M, dtype=float)
    workspace_max = np.array(cfg.ROBOT_WORKSPACE_MAX_M, dtype=float)
    if not np.all((workspace_min <= target_xyz) & (target_xyz <= workspace_max)):
        raise ValueError(
            "Saved pose target is outside ROBOT_WORKSPACE bounds: "
            f"target={target_xyz.tolist()}, "
            f"min={workspace_min.tolist()}, max={workspace_max.tolist()}"
        )


def connect_robot():
    """
    Connect to the Franka and apply conservative control/collision settings.

    This mirrors the calibration scripts so pose planning and calibration use
    the same robot safety configuration. The planner needs a robot connection
    even before replaying poses because it continuously reads O_T_EE for saving.
    """
    realtime_config = getattr(getattr(pylibfranka, "RealtimeConfig", None), "kIgnore", None)
    if realtime_config is None:
        robot = pylibfranka.Robot(cfg.FRANKA_IP)
    else:
        robot = pylibfranka.Robot(cfg.FRANKA_IP, realtime_config)

    apply_franka_control_config(robot, load_profile=LOAD_PROFILE_D405_CALIBRATION)

    return robot


def start_d405_pipeline():
    """
    Start the D405 color/depth streams used by calibration.

    The planner only displays and analyzes the color image, but enabling depth
    keeps the camera setup consistent with the D405 calibration/runtime stream
    configuration.
    """
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_device(cfg.D405_SERIAL)
    rs_config.enable_stream(
        rs.stream.depth,
        cfg.D405_RESOLUTION[0],
        cfg.D405_RESOLUTION[1],
        rs.format.z16,
        cfg.CAMERA_FPS,
    )
    rs_config.enable_stream(
        rs.stream.color,
        cfg.D405_RESOLUTION[0],
        cfg.D405_RESOLUTION[1],
        rs.format.bgr8,
        cfg.CAMERA_FPS,
    )
    pipeline.start(rs_config)
    time.sleep(cfg.CAMERA_WARMUP_SECONDS)
    return pipeline


def load_pose_plan(path, replace=False):
    """
    Load an existing pose plan, unless the operator asked to replace it.

    This lets you build a plan over multiple sessions. Each record is checked
    for a valid 4x4 T_ee_to_base matrix before the script offers to replay it.
    """
    if replace or not path.exists():
        return []

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    poses = data.get("poses", [])
    if not isinstance(poses, list):
        raise ValueError(f"Pose plan {path} has invalid 'poses' field")

    for pose in poses:
        matrix = np.array(pose.get("T_ee_to_base"), dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError(f"Pose plan {path} contains a non-4x4 pose")
    return poses


def reindex_poses(poses):
    """Keep visible pose numbers sequential after undo/delete operations."""
    for index, pose in enumerate(poses, start=1):
        pose["index"] = index


def save_pose_plan(path, poses):
    """
    Write the current pose plan to JSON.

    The file stores enough metadata to understand which camera, board model, and
    robot endpoint produced the plan. It also stores optional ChArUco detection
    data from the moment each pose was saved, which is helpful when diagnosing
    whether a planned pose had a strong board view.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    reindex_poses(poses)
    payload = {
        "description": "Preset Franka EE poses for D405 hand-eye calibration",
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "frame_convention": "T_ee_to_base maps points from Franka EE frame into robot base frame",
        "robot_ip": cfg.FRANKA_IP,
        "camera": "D405",
        "camera_serial": cfg.D405_SERIAL,
        "camera_resolution": list(cfg.D405_RESOLUTION),
        "board": {
            "inner_corners": list(cfg.HAND_EYE_BOARD_CORNERS),
            "square_size_m": cfg.HAND_EYE_SQUARE_SIZE,
            "marker_size_m": cfg.HAND_EYE_MARKER_SIZE,
            "aruco_dictionary": cfg.HAND_EYE_ARUCO_DICT_NAME,
            "legacy_pattern": cfg.HAND_EYE_CHARUCO_LEGACY_PATTERN,
        },
        "pose_count": len(poses),
        "poses": poses,
    }
    path.write_text(json.dumps(payload, indent=4), encoding="utf-8")


def make_pose_record(pose, detection, save_without_detection=False):
    """
    Build one JSON record for the current robot pose and preview detection.

    T_ee_to_base is the critical value used later by the calibration script.
    T_board_to_cam is included only as diagnostic context from the planning
    moment; the actual calibration script re-detects the board after moving to
    each saved pose so it captures fresh synchronized pose pairs.
    """
    T_board_to_cam = detection.get("T_board_to_cam")
    return {
        "index": None,
        "timestamp_s": time.time(),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "T_ee_to_base": pose.tolist(),
        "translation_m": pose[:3, 3].tolist(),
        "detected": bool(detection.get("success")),
        "saved_without_detection": bool(save_without_detection),
        "marker_count": int(detection.get("marker_count", 0)),
        "charuco_count": int(detection.get("charuco_count", 0)),
        "detection_reason": detection.get("reason", ""),
        "T_board_to_cam": T_board_to_cam.tolist() if T_board_to_cam is not None else None,
    }


def confirm_motion(current_pose, target_pose, selected_index, args):
    """
    Ask for terminal confirmation before replaying a saved full pose.

    The preview window uses keyboard shortcuts, but actual robot motion is
    confirmed in the terminal so accidental keypresses in the camera window do
    not immediately move the arm.
    """
    current_xyz = current_pose[:3, 3]
    target_xyz = target_pose[:3, 3]
    translation_distance = float(np.linalg.norm(target_xyz - current_xyz))
    angle_deg = np.degrees(rotation_angle(current_pose[:3, :3].T @ target_pose[:3, :3]))

    print("\nWARNING: This will move the Franka to a saved full EE pose.")
    print("Keep the user stop button within reach.")
    print(f"Selected pose: {selected_index + 1}")
    print(f"Translation distance: {translation_distance:.4f} m")
    print(f"Rotation distance:    {angle_deg:.2f} deg")
    print(f"Target xyz [m]:       {target_xyz.tolist()}")

    if args.skip_motion_confirmation:
        return

    answer = input("Type 'move' to execute this motion: ").strip().lower()
    if answer != "move":
        raise RuntimeError("Motion cancelled by operator")


def move_to_full_pose(robot, target_pose, args):
    """
    Command the robot to a saved full T_ee_to_base pose.

    Unlike the simpler XYZ waypoint helpers elsewhere in the project, this
    interpolates both translation and orientation. Replaying the saved wrist
    orientations is the whole point of the pose plan: hand-eye calibration needs
    real rotational diversity.
    """
    validate_workspace_target(target_pose)

    start_pose = get_robot_ee_pose(robot)
    translation_distance = float(np.linalg.norm(target_pose[:3, 3] - start_pose[:3, 3]))
    angle_rad = rotation_angle(start_pose[:3, :3].T @ target_pose[:3, :3])
    duration_s = max(
        translation_distance / args.move_speed_mps,
        angle_rad / args.move_rot_speed_radps,
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
            start_pose[:3, :3],
            target_pose[:3, :3],
            eased,
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
    cv2.putText(
        image,
        text,
        (position[0] + 1, position[1] + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
    )
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
    )


def draw_preview(display, detection, robot_pose, poses, selected_index, output_path):
    """
    Draw operator status on top of the D405 camera image.

    The overlay intentionally mirrors calibration mode: green means a strong
    ChArUco pose is available, yellow means partial board, red means searching.
    It also shows the live EE position and the pose-plan controls.
    """
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
    selected_label = "none" if selected_index is None else str(selected_index + 1)

    draw_text(display, status, (20, 40), color, scale=0.7)
    draw_text(
        display,
        f"EE xyz [m]: x={xyz[0]: .3f} y={xyz[1]: .3f} z={xyz[2]: .3f}",
        (20, 75),
        (255, 255, 255),
    )
    draw_text(
        display,
        f"Saved poses: {len(poses)} | selected: {selected_label}",
        (20, 110),
        (255, 255, 255),
    )
    draw_text(
        display,
        "s save green | p save anyway | g move selected | n/b select | u undo | x delete | q quit",
        (20, display.shape[0] - 55),
        (255, 255, 255),
        scale=0.55,
    )
    draw_text(
        display,
        f"Plan: {output_path}",
        (20, display.shape[0] - 25),
        (255, 255, 255),
        scale=0.5,
        thickness=1,
    )


def main():
    """
    Run the interactive planner loop.

    The loop continuously:
      - reads a D405 color frame,
      - estimates the current ChArUco board pose,
      - reads the current Franka O_T_EE,
      - displays all of that in one preview window,
      - responds to save/select/delete/replay keys.

    The script saves after every pose-list edit so an unexpected stop usually
    leaves the latest plan on disk.
    """
    args = parse_args()
    output_path = Path(args.output).expanduser()

    try:
        validate_motion_args(args)
        require_runtime_dependencies()
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    intrinsics_path = resolve_existing_config_path(cfg.INTRINSICS_D405_PATH)
    intrinsics = load_intrinsics(str(intrinsics_path))
    if intrinsics is None:
        print(f"ERROR: D405 intrinsics not found at {intrinsics_path}")
        print("Run calibrate_intrinsics.py for the D405 before planning hand-eye poses.")
        return 1

    try:
        poses = load_pose_plan(output_path, replace=args.replace)
    except Exception as exc:
        print(f"ERROR: Could not load pose plan: {exc}")
        return 1

    selected_index = len(poses) - 1 if poses else None
    camera_matrix, dist_coeffs = get_opencv_matrices(intrinsics)
    aruco_dict = get_aruco_dictionary(cv2, cfg.HAND_EYE_ARUCO_DICT_NAME)
    charuco_board = create_charuco_board(
        cv2,
        cfg.HAND_EYE_BOARD_CORNERS,
        cfg.HAND_EYE_SQUARE_SIZE,
        cfg.HAND_EYE_MARKER_SIZE,
        aruco_dict,
        legacy_pattern=cfg.HAND_EYE_CHARUCO_LEGACY_PATTERN,
    )

    print("=" * 72)
    print("  D405 HAND-EYE POSE PLANNER")
    print("=" * 72)
    print(f"Robot IP:      {cfg.FRANKA_IP}")
    print(f"D405 serial:   {cfg.D405_SERIAL}")
    print(f"Intrinsics:    {intrinsics_path}")
    print(f"Pose plan:     {output_path}")
    print(f"Loaded poses:  {len(poses)}")
    print(f"Min corners:   {args.min_corners}")
    print("\nControls in camera window:")
    print("  s: save current full EE pose when ChArUco status is green")
    print("  p: save current full EE pose even if ChArUco is not green")
    print("  n/b: select next/previous saved pose")
    print("  g: move robot to selected saved full pose")
    print("  u: undo last saved pose")
    print("  x: delete selected saved pose")
    print("  q or ESC: save and quit")

    robot = None
    pipeline = None
    window_name = "D405 Hand-Eye Pose Planner"

    try:
        print("\nConnecting to Franka...")
        robot = connect_robot()
        print("Franka connected")

        print("\nStarting D405 camera...")
        pipeline = start_d405_pipeline()
        print(f"D405 started at {cfg.D405_RESOLUTION[0]}x{cfg.D405_RESOLUTION[1]}")

        save_pose_plan(output_path, poses)

        while True:
            frames = pipeline.wait_for_frames(timeout_ms=cfg.CALIBRATION_FRAME_TIMEOUT_MS)
            color_frame = frames.get_color_frame()
            if not color_frame:
                print("Warning: dropped color frame")
                continue

            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            robot_pose = get_robot_ee_pose(robot)

            detection = detect_charuco_board_pose(
                cv2,
                gray,
                aruco_dict,
                charuco_board,
                camera_matrix,
                dist_coeffs,
                min_corners=args.min_corners,
            )

            display = color_image.copy()
            draw_charuco_detection(cv2, display, detection)
            if detection["success"]:
                draw_pose_axes(
                    cv2,
                    display,
                    detection["T_board_to_cam"],
                    camera_matrix,
                    dist_coeffs,
                    axis_length=0.05,
                )
            draw_preview(display, detection, robot_pose, poses, selected_index, output_path)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key in QUIT_KEYS:
                save_pose_plan(output_path, poses)
                print(f"\nSaved {len(poses)} poses to {output_path}")
                break

            if key in {ord("s"), ord("S")}:
                if not detection["success"]:
                    print("ChArUco status is not green; pose not saved. Press 'p' to force-save.")
                    continue
                poses.append(make_pose_record(robot_pose, detection))
                selected_index = len(poses) - 1
                save_pose_plan(output_path, poses)
                print(
                    f"Saved pose {len(poses)} "
                    f"with {detection.get('charuco_count', 0)} ChArUco corners"
                )

            elif key in {ord("p"), ord("P")}:
                poses.append(make_pose_record(robot_pose, detection, save_without_detection=True))
                selected_index = len(poses) - 1
                save_pose_plan(output_path, poses)
                print(
                    f"Force-saved pose {len(poses)} "
                    f"with {detection.get('charuco_count', 0)} ChArUco corners"
                )

            elif key in {ord("n"), ord("N")} and poses:
                selected_index = 0 if selected_index is None else (selected_index + 1) % len(poses)
                print(f"Selected pose {selected_index + 1}/{len(poses)}")

            elif key in {ord("b"), ord("B")} and poses:
                selected_index = len(poses) - 1 if selected_index is None else (selected_index - 1) % len(poses)
                print(f"Selected pose {selected_index + 1}/{len(poses)}")

            elif key in {ord("u"), ord("U")} and poses:
                removed = poses.pop()
                save_pose_plan(output_path, poses)
                selected_index = len(poses) - 1 if poses else None
                print(f"Undid saved pose {removed.get('index')}; {len(poses)} poses remain")

            elif key in {ord("x"), ord("X")} and poses and selected_index is not None:
                removed = poses.pop(selected_index)
                save_pose_plan(output_path, poses)
                if not poses:
                    selected_index = None
                else:
                    selected_index = min(selected_index, len(poses) - 1)
                print(f"Deleted pose {removed.get('index')}; {len(poses)} poses remain")

            elif key in {ord("g"), ord("G")}:
                if not poses or selected_index is None:
                    print("No saved pose selected.")
                    continue
                target_pose = np.array(poses[selected_index]["T_ee_to_base"], dtype=float)
                try:
                    current_pose = get_robot_ee_pose(robot)
                    confirm_motion(current_pose, target_pose, selected_index, args)
                    print(f"Moving to saved pose {selected_index + 1}/{len(poses)}...")
                    move_to_full_pose(robot, target_pose, args)
                    if args.settle_s > 0.0:
                        print(f"Waiting {args.settle_s:.2f} s for settling...")
                        time.sleep(args.settle_s)
                    print("Move complete")
                except Exception as exc:
                    print(f"Move failed: {exc}")
                    stop = getattr(robot, "stop", None)
                    if callable(stop):
                        stop()

    except KeyboardInterrupt:
        print("\nInterrupted; stopping robot and saving pose plan.")
        if robot is not None:
            stop = getattr(robot, "stop", None)
            if callable(stop):
                stop()
        save_pose_plan(output_path, poses)
        return 130
    except Exception as exc:
        print(f"\nERROR: {exc}")
        if robot is not None:
            stop = getattr(robot, "stop", None)
            if callable(stop):
                stop()
        save_pose_plan(output_path, poses)
        return 1
    finally:
        if pipeline is not None:
            pipeline.stop()
        cv2.destroyAllWindows()
        close = getattr(robot, "close", None) if robot is not None else None
        if callable(close):
            close()
        print("Cleaned up camera, robot, and display resources.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
