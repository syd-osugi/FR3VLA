"""
Script: D405 Eye-in-Hand Extrinsic Calibration
----------------------------------------------
Finds the rigid transform between the wrist-mounted D405 camera and the Franka
end-effector frame using a ChArUco board.

READ THIS FIRST: PLANNED-POSE / PATH-PLANNER WORKFLOW
=====================================================
This file has two ways to collect D405 hand-eye calibration samples:

  A. Manual mode:
     - Run this file directly.
     - Hand-guide or command the robot to each calibration pose yourself.
     - Press 's' in the preview at each still, green ChArUco view.

  B. Planned-pose mode:
     - First run the separate pose/path planner:

           python Working/camera_calibration/plan_d405_hand_eye_poses.py

       The planner opens the same D405 ChArUco preview used by this calibration
       script. While looking at that preview, move the robot to useful
       calibration viewpoints and press 's' to save each full Franka EE pose.
       It saves by default to:

           Working/camera_calibration/calibration_data/d405_hand_eye_pose_plan.json

       The saved records contain full 4x4 T_ee_to_base matrices, including
       wrist orientation. That matters: D405 hand-eye calibration needs varied
       rotations, not only varied XYZ positions.

     - Then run this calibration script with the saved plan:

           python Working/camera_calibration/run_extrinsics_d405_hand_eye.py --pose-plan

       If you saved the plan somewhere else, pass the path:

           python Working/camera_calibration/run_extrinsics_d405_hand_eye.py \
               --pose-plan /path/to/d405_hand_eye_pose_plan.json

       In planned-pose mode this script will:
         1. Load the saved full EE poses.
         2. Ask you to type 'move' before each robot motion.
         3. Command the Franka to the next saved full pose.
         4. Wait for settling.
         5. Show the D405 ChArUco preview.
         6. Let you press 's' only when the board is detected well enough.
         7. Save the synchronized pair: current O_T_EE and current board pose.

       Preview keys in planned-pose mode:
         s: capture the current planned pose when the board is green
         n: skip this planned pose and move on
         q: quit early after at least 3 captured poses

  Recommended count:
     Six poses can be useful for a quick smoke test, but a real hand-eye
     calibration should use about 12-20 sharp, varied views. Include wrist
     pitch, yaw, roll, height, and XY variation. Avoid saving several poses
     that only translate with nearly identical orientation.

  Physical setup reminder:
     The D405 is mounted rigidly to the wrist/end-effector. The ChArUco board
     must stay fixed in the workspace, usually flat on the table. If the board
     moves, or the camera mount moves relative to the EE frame, no single
     T_d405_to_ee can explain the data.

WHAT THIS SCRIPT CALIBRATES:
============================
The D405 is mounted on the robot hand/wrist assembly, so the camera moves with
the end effector. To use D405 depth points for robot motion, the system needs:

    T_d405_to_ee

Matrix convention:
    T_a_to_b maps a point from frame A into frame B.

Runtime use:
    p_base = T_ee_to_base @ T_d405_to_ee @ p_d405

CALIBRATION INPUTS:
===================
At each captured pose, this script records:

  T_ee_to_base:
    - Read from Franka O_T_EE.
    - End-effector frame -> robot base frame.
    - Changes every time the robot moves.

  T_board_to_cam:
    - Estimated from the D405 image of a fixed ChArUco board.
    - Board/target frame -> D405 optical frame.
    - Changes every time the robot moves.

OpenCV hand-eye calibration combines those pose pairs and solves for:

    T_d405_to_ee

PREREQUISITES:
==============
1. D405 intrinsic calibration must be complete.
2. A ChArUco board must be fixed in the workspace, usually flat on the table.
3. The robot must be controllable and readable through pylibfranka.

HOW TO USE:
===========
Manual mode:
  1. Place the ChArUco board where the D405 can see it from many wrist poses.
  2. Run:

         python Working/camera_calibration/run_extrinsics_d405_hand_eye.py

  3. Move the robot to varied poses while keeping the board visible.
  4. Press 's' at each still, well-detected pose.
  5. The script saves calibration_data/d405_to_wrist.json.

Planned-pose mode:
  1. Run the separate planner first:

         python Working/camera_calibration/plan_d405_hand_eye_poses.py

  2. Save a list of full EE poses while watching the D405 preview.
  3. Run:

         python Working/camera_calibration/run_extrinsics_d405_hand_eye.py --pose-plan

  4. The robot will revisit those saved full poses, and you capture each green
     ChArUco view with 's'.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

# Add parent directory to path for imports when this script is run directly.
WORKING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKING_DIR not in sys.path:
    sys.path.insert(0, WORKING_DIR)

import numpy as np

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
from camera_calibration.hand_eye_math import (
    calibrate_hand_eye,
    save_hand_eye,
    validate_hand_eye_result,
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


SCRIPT_DIR = Path(__file__).resolve().parent

# The planner and the calibration script intentionally share this default path.
# That lets the operator run:
#
#     python Working/camera_calibration/plan_d405_hand_eye_poses.py
#     python Working/camera_calibration/run_extrinsics_d405_hand_eye.py --pose-plan
#
# without needing to copy/paste a JSON path between the two commands.
DEFAULT_POSE_PLAN_PATH = SCRIPT_DIR / "calibration_data" / "d405_hand_eye_pose_plan.json"


def parse_args():
    """
    Parse optional planned-pose arguments.

    Manual calibration remains the default because it is the simplest workflow:
    running the script with no flags behaves like the original hand-guided
    capture flow. Planned-pose mode only turns on when --pose-plan is supplied.

    The motion flags are deliberately conservative. They are used only when the
    script commands the robot through a saved pose plan; they do not affect
    manual mode where the user moves the robot independently.
    """
    parser = argparse.ArgumentParser(
        description="Run D405 eye-in-hand extrinsic calibration."
    )
    parser.add_argument(
        "--pose-plan",
        nargs="?",
        const=str(DEFAULT_POSE_PLAN_PATH),
        default=None,
        help=(
            "Use a saved full-pose plan. If no path is supplied, uses "
            "Working/camera_calibration/calibration_data/d405_hand_eye_pose_plan.json."
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


def require_runtime_dependencies():
    """Checks runtime-only dependencies before connecting to hardware."""
    require_charuco_support(cv2)
    if rs is None:
        raise RuntimeError("pyrealsense2 is required for RealSense calibration.")


def connect_robot():
    """Connects to the Franka and applies conservative collision thresholds."""
    try:
        from pylibfranka import Robot
    except ImportError as exc:
        raise RuntimeError("pylibfranka is required to read robot poses.") from exc

    robot = Robot(cfg.FRANKA_IP)
    apply_franka_control_config(robot, load_profile=LOAD_PROFILE_D405_CALIBRATION)
    return robot


def resolve_pose_plan_path(path_value):
    """
    Resolve a pose-plan path from either the repo root or this script directory.

    Users often run calibration scripts from the repository root, but the default
    plan file lives beside the calibration scripts. This resolver accepts:
      - absolute paths,
      - paths that already exist relative to the current working directory,
      - paths relative to Working/camera_calibration.
    """
    path = Path(path_value).expanduser()
    if path.is_absolute() or path.exists():
        return path

    script_relative = SCRIPT_DIR / path
    if script_relative.exists():
        return script_relative

    return path


def load_pose_plan(path_value):
    """
    Load the JSON produced by plan_d405_hand_eye_poses.py.

    Each saved entry is expected to contain T_ee_to_base, a full 4x4 transform
    from Franka end-effector coordinates into the robot base frame. The full
    matrix is important because the calibration needs the wrist orientation from
    every pose, not only the end-effector XYZ position.
    """
    path = resolve_pose_plan_path(path_value)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    poses = data.get("poses", [])
    if not isinstance(poses, list) or not poses:
        raise ValueError(f"Pose plan has no saved poses: {path}")

    planned_poses = []
    for index, pose_record in enumerate(poses, start=1):
        pose = np.array(pose_record.get("T_ee_to_base"), dtype=float)
        if pose.shape != (4, 4):
            raise ValueError(f"Pose {index} in {path} is not a 4x4 T_ee_to_base")
        planned_poses.append(pose)

    return path, planned_poses


def get_robot_ee_pose(robot):
    """
    Reads end-effector -> robot base from the Franka.

    Franka O_T_EE is the pose of the end-effector frame in the robot base frame.
    The API provides it as a column-major flat array, so reshape with order='F'.
    """
    state = robot.read_once()
    return np.array(state.O_T_EE).reshape((4, 4), order="F")


def matrix_to_franka_pose(matrix):
    """
    Convert a 4x4 transform back to Franka's flat column-major pose format.

    Franka's O_T_EE uses the same 4x4 homogeneous transform values, flattened in
    column-major order. Keeping this conversion explicit avoids mixing it up
    with NumPy's row-major default.
    """
    return np.array(matrix, dtype=float).reshape(16, order="F").tolist()


def duration_to_seconds(duration):
    """Convert pylibfranka duration objects, or numeric fallback values, to seconds."""
    to_sec = getattr(duration, "to_sec", None)
    if callable(to_sec):
        return float(to_sec())
    return float(duration)


def smoothstep(alpha):
    """
    Smooth interpolation curve for planned-pose robot motion.

    A raw linear alpha has abrupt velocity changes at the start and end of a
    move. Smoothstep keeps the command profile gentler while still ending exactly
    at the saved pose.
    """
    bounded = max(0.0, min(1.0, float(alpha)))
    return bounded * bounded * (3.0 - 2.0 * bounded)


def rotation_angle(rotation):
    """Return the angle, in radians, represented by a 3x3 rotation matrix."""
    trace = np.trace(rotation)
    return float(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))


def interpolate_rotation(start_rotation, target_rotation, alpha):
    """
    Interpolate between two wrist orientations using Rodrigues vectors.

    The saved pose plan contains full orientations. During replay we command a
    gradual rotation from the current orientation to the saved orientation,
    rather than snapping orientation at the final Cartesian point.
    """
    rotation_delta = start_rotation.T @ target_rotation
    rvec_delta, _ = cv2.Rodrigues(rotation_delta)
    partial_delta, _ = cv2.Rodrigues(rvec_delta * float(alpha))
    return start_rotation @ partial_delta


def validate_motion_args(args):
    """
    Reject obviously unsafe or unusable command-line motion settings.

    This does not replace Franka safety, collision thresholds, workspace bounds,
    or human supervision. It only catches invalid numeric input before the
    script connects to hardware.
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
    Check that a saved pose's target XYZ lies inside configured workspace bounds.

    The planner saves real robot poses, so this usually passes. Keeping the check
    here protects against editing the JSON by hand or replaying an old plan after
    tightening ROBOT_WORKSPACE_MIN_M / ROBOT_WORKSPACE_MAX_M in config.py.
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


def confirm_planned_motion(current_pose, target_pose, planned_index, planned_count, args):
    """
    Print a readable summary and require typed confirmation before robot motion.

    The preview window cannot safely confirm terminal actions by itself, so the
    planned-pose workflow asks for 'move' in the terminal before each saved pose.
    Use --skip-motion-confirmation only when the workspace is clear and you
    intentionally want unattended stepping through the saved plan.
    """
    if args.skip_motion_confirmation:
        return

    current_xyz = current_pose[:3, 3]
    target_xyz = target_pose[:3, 3]
    translation_distance = float(np.linalg.norm(target_xyz - current_xyz))
    angle_deg = np.degrees(rotation_angle(current_pose[:3, :3].T @ target_pose[:3, :3]))

    print("\nWARNING: This will move the Franka to a saved full EE pose.")
    print("Keep the user stop button within reach.")
    print(f"Planned pose:         {planned_index + 1}/{planned_count}")
    print(f"Translation distance: {translation_distance:.4f} m")
    print(f"Rotation distance:    {angle_deg:.2f} deg")
    print(f"Target xyz [m]:       {target_xyz.tolist()}")
    answer = input("Type 'move' to execute this motion: ").strip().lower()
    if answer != "move":
        raise RuntimeError("Motion cancelled by operator")


def move_to_full_pose(robot, target_pose, args):
    """
    Move the robot to a saved full T_ee_to_base pose.

    This function commands both translation and orientation. That is different
    from the simpler helper in robot_interface.py, which currently preserves the
    existing orientation while moving XYZ. Preserving orientation would defeat
    the purpose of a hand-eye pose plan, because the calibration needs varied
    wrist rotations.
    """
    try:
        from pylibfranka import CartesianPose, ControllerMode
    except ImportError as exc:
        raise RuntimeError("pylibfranka is required for planned-pose motion.") from exc

    validate_workspace_target(target_pose)

    start_pose = get_robot_ee_pose(robot)
    translation_distance = float(np.linalg.norm(target_pose[:3, 3] - start_pose[:3, 3]))
    angle_rad = rotation_angle(start_pose[:3, :3].T @ target_pose[:3, :3])
    duration_s = max(
        translation_distance / args.move_speed_mps,
        angle_rad / args.move_rot_speed_radps,
        0.5,
    )

    active_control = robot.start_cartesian_pose_control(ControllerMode.JointImpedance)
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

        cartesian_pose = CartesianPose(matrix_to_franka_pose(command_pose))
        if alpha >= 1.0:
            cartesian_pose.motion_finished = True
            motion_finished = True

        active_control.writeOnce(cartesian_pose)


def move_to_next_planned_pose(robot, planned_poses, planned_index, args):
    """
    Confirm, command, and settle at the next saved pose-plan entry.

    After this returns, the main preview loop resumes and waits for the user to
    press 's'. The script does not auto-capture immediately after motion because
    the operator should inspect the ChArUco axes and corner count first.
    """
    target_pose = planned_poses[planned_index]
    current_pose = get_robot_ee_pose(robot)
    confirm_planned_motion(
        current_pose,
        target_pose,
        planned_index,
        len(planned_poses),
        args,
    )
    print(f"\nMoving to planned pose {planned_index + 1}/{len(planned_poses)}...")
    move_to_full_pose(robot, target_pose, args)
    if args.settle_s > 0.0:
        print(f"Waiting {args.settle_s:.2f} s for robot settling...")
        time.sleep(args.settle_s)
    print("Move complete. Use the preview, then press 's' when green.")


def main():
    """Runs D405 eye-in-hand calibration with a ChArUco target."""
    args = parse_args()

    try:
        require_runtime_dependencies()
        validate_motion_args(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    pose_plan_path = None
    planned_poses = None
    if args.pose_plan is not None:
        try:
            pose_plan_path, planned_poses = load_pose_plan(args.pose_plan)
        except Exception as exc:
            print(f"ERROR: Could not load pose plan: {exc}")
            return 1

    print("=" * 70)
    print("  D405 EYE-IN-HAND EXTRINSIC CALIBRATION")
    print("=" * 70)
    if planned_poses is not None:
        print(f"  Planned-pose mode: {pose_plan_path}")
        print(f"  Planned poses:     {len(planned_poses)}")
    print(f"  Min ChArUco corners for capture: {args.min_corners}")

    print(f"\n[1/4] Connecting to Franka robot at {cfg.FRANKA_IP}...")
    try:
        robot = connect_robot()
        print("      Robot connected")
    except Exception as exc:
        print(f"      ERROR: Could not connect to robot: {exc}")
        return 1

    print(f"\n[2/4] Starting D405 camera...")
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

    try:
        pipeline.start(rs_config)
        print(
            f"      Camera started at {cfg.D405_RESOLUTION[0]}x{cfg.D405_RESOLUTION[1]}"
        )
    except Exception as exc:
        print(f"      ERROR: Could not start D405 camera: {exc}")
        return 1

    time.sleep(cfg.CAMERA_WARMUP_SECONDS)

    print(f"\n[3/4] Loading D405 intrinsics from {cfg.INTRINSICS_D405_PATH}...")
    intrinsics = load_intrinsics(cfg.INTRINSICS_D405_PATH)
    if intrinsics is None:
        print("      ERROR: D405 intrinsics not found.")
        print("      Run Working/camera_calibration/calibrate_intrinsics.py first.")
        pipeline.stop()
        return 1

    camera_matrix, dist_coeffs = get_opencv_matrices(intrinsics)
    print("      Intrinsics loaded")

    print(f"\n[4/4] Preparing ChArUco board detector...")
    aruco_dict = get_aruco_dictionary(cv2, cfg.HAND_EYE_ARUCO_DICT_NAME)
    charuco_board = create_charuco_board(
        cv2,
        cfg.HAND_EYE_BOARD_CORNERS,
        cfg.HAND_EYE_SQUARE_SIZE,
        cfg.HAND_EYE_MARKER_SIZE,
        aruco_dict,
        legacy_pattern=cfg.HAND_EYE_CHARUCO_LEGACY_PATTERN,
    )
    poses_required = len(planned_poses) if planned_poses is not None else cfg.HAND_EYE_POSES_REQUIRED
    print("      ChArUco detector ready")

    robot_poses = []   # T_ee_to_base for each capture.
    camera_poses = []  # T_board_to_cam for each matching D405 image.

    print("\n" + "=" * 70)
    print("  CAPTURE PHASE")
    print("=" * 70)
    print(
        f"""
  SETUP:
    - Place a ChArUco board fixed in the workspace, usually flat on the table.
    - The board should not move while this script runs.
    - Board inner corners: {cfg.HAND_EYE_BOARD_CORNERS}
    - Square size: {cfg.HAND_EYE_SQUARE_SIZE:.4f} m
    - Marker size: {cfg.HAND_EYE_MARKER_SIZE:.4f} m
    - ArUco dictionary: {cfg.HAND_EYE_ARUCO_DICT_NAME}
    - Legacy ChArUco pattern: {cfg.HAND_EYE_CHARUCO_LEGACY_PATTERN}

  INSTRUCTIONS:
    - Move the robot to varied positions and wrist orientations.
    - Keep the ChArUco board visible in the D405 image.
    - Watch the preview: green means the board is recognized.
    - Press 's' when the board is detected and the robot is still.
    - Capture {poses_required} poses.
    - Press 'q' to quit early after at least 3 good poses.

  TIPS:
    - Use strong variation in rotation and translation.
    - Avoid only sliding the wrist in a straight line.
    - Discard blurred or partially hidden detections.
"""
    )
    if planned_poses is not None:
        print(
            """
  PLANNED-POSE MODE:
    - The robot will move through your saved full EE poses.
    - Type 'move' in the terminal before each planned motion.
    - After each move, use the D405 preview and press 's' when green.
    - Press 'n' in the preview to skip a planned pose.
"""
        )
    print("=" * 70 + "\n")

    captured_count = 0
    planned_index = 0
    planned_pose_ready = planned_poses is None

    try:
        while captured_count < poses_required:
            if planned_poses is not None and not planned_pose_ready:
                if planned_index >= len(planned_poses):
                    print("\n  No more planned poses.")
                    break
                move_to_next_planned_pose(robot, planned_poses, planned_index, args)
                planned_pose_ready = True

            frames = pipeline.wait_for_frames(timeout_ms=cfg.CALIBRATION_FRAME_TIMEOUT_MS)
            color_frame = frames.get_color_frame()

            if not color_frame:
                print("  Warning: Dropped frame, retrying...")
                continue

            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            detection = detect_charuco_board_pose(
                cv2,
                gray,
                aruco_dict,
                charuco_board,
                camera_matrix,
                dist_coeffs,
                min_corners=args.min_corners,
            )
            detected = detection["success"]
            T_board_to_cam = detection["T_board_to_cam"]
            marker_count = detection.get("marker_count", 0)
            charuco_count = detection.get("charuco_count", 0)

            display = color_image.copy()
            draw_charuco_detection(cv2, display, detection)

            if detected:
                draw_pose_axes(
                    cv2,
                    display,
                    T_board_to_cam,
                    camera_matrix,
                    dist_coeffs,
                    axis_length=0.05,
                )
                cv2.putText(
                    display,
                    f"CHARUCO DETECTED: {charuco_count} corners",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    display,
                    "Press 's' to capture this pose",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )
            else:
                if marker_count:
                    status = (
                        f"PARTIAL BOARD: {marker_count} markers, "
                        f"{charuco_count} corners"
                    )
                    color = (0, 220, 255)
                else:
                    status = "Searching for ChArUco board..."
                    color = (0, 0, 255)

                cv2.putText(
                    display,
                    status,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
                cv2.putText(
                    display,
                    "Move until the board status turns green",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                )

            cv2.putText(
                display,
                f"Captured: {captured_count}/{poses_required}",
                (20, display.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            if planned_poses is not None:
                cv2.putText(
                    display,
                    f"Planned pose: {planned_index + 1}/{len(planned_poses)}  |  n: skip",
                    (20, display.shape[0] - 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("D405 ChArUco Hand-Eye Calibration", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                if not detected:
                    print("  ChArUco board is not detected; capture skipped.")
                    continue

                print(f"  [{captured_count + 1}/{poses_required}] Capturing pose...")

                # Store one synchronized pose pair:
                #   T_ee_to_base comes from robot encoders.
                #   T_board_to_cam comes from the D405 image.
                #
                # OpenCV's hand-eye solver uses these repeated pairs to compute
                # the fixed transform T_d405_to_ee.
                T_ee_to_base = get_robot_ee_pose(robot)
                robot_poses.append(T_ee_to_base)
                camera_poses.append(T_board_to_cam)

                captured_count += 1
                if planned_poses is not None:
                    planned_index += 1
                    planned_pose_ready = False
                print("      Pose saved")

            elif key == ord("n") and planned_poses is not None:
                print(f"\n  Skipping planned pose {planned_index + 1}/{len(planned_poses)}...")
                planned_index += 1
                planned_pose_ready = False
                if planned_index >= len(planned_poses):
                    print("  No more planned poses.")
                    break

            elif key == ord("q"):
                print("\n  Quitting early...")
                break

        if captured_count < 3:
            print(f"\nERROR: Need at least 3 poses, only captured {captured_count}.")
            return 1

        print("\n" + "=" * 70)
        print("  COMPUTING CALIBRATION")
        print("=" * 70)
        print(f"  Using {captured_count} captured poses...")

        T_cam_to_ee = calibrate_hand_eye(robot_poses, camera_poses)
        is_valid, rot_err, trans_err = validate_hand_eye_result(
            T_cam_to_ee,
            robot_poses,
            camera_poses,
        )

        print("\n  Results:")
        print(f"    Max rotation error:    {rot_err:.3f} deg")
        print(f"    Max translation error: {trans_err * 1000:.2f} mm")
        print(f"    Validation:            {'PASS' if is_valid else 'FAIL'}")

        print("\n  Transform matrix (D405 camera -> end-effector):")
        for row in T_cam_to_ee:
            print(f"    [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f} {row[3]:8.4f}]")

        robot_serial = f"franka@{cfg.FRANKA_IP}"
        save_hand_eye(
            cfg.HAND_EYE_D405_PATH,
            T_cam_to_ee,
            robot_serial,
            validation_results=(is_valid, rot_err, trans_err),
        )

        print("\n" + "=" * 70)
        print("  CALIBRATION COMPLETE")
        print("=" * 70)
        print(f"\n  File saved: {cfg.HAND_EYE_D405_PATH}")

        if not is_valid:
            print("\n  WARNING: Validation failed.")
            print("  Consider recapturing with more varied, sharper ChArUco views.")

    except Exception as exc:
        print(f"\nERROR: {exc}")
        import traceback

        traceback.print_exc()
        return 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\nCleaned up camera and display resources.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
