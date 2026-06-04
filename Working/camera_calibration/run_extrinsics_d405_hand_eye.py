"""
Script: D405 Eye-in-Hand Extrinsic Calibration
----------------------------------------------
Finds the rigid transform between the wrist-mounted D405 camera and the Franka
end-effector frame using a ChArUco board.

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
1. Place the ChArUco board where the D405 can see it from many wrist poses.
2. Run this script.
3. Move the robot to varied poses while keeping the board visible.
4. Press 's' at each still, well-detected pose.
5. The script saves calibration_data/d405_to_wrist.json.
"""

import os
import sys
import time

# Add parent directory to path for imports when this script is run directly.
WORKING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKING_DIR not in sys.path:
    sys.path.insert(0, WORKING_DIR)

import numpy as np

import config as cfg
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
    robot.set_collision_behavior(
        [cfg.FRANKA_COLLISION_TORQUE_NM] * 7,
        [cfg.FRANKA_COLLISION_TORQUE_NM] * 7,
        [cfg.FRANKA_COLLISION_FORCE_N] * 6,
        [cfg.FRANKA_COLLISION_FORCE_N] * 6,
    )
    return robot


def get_robot_ee_pose(robot):
    """
    Reads end-effector -> robot base from the Franka.

    Franka O_T_EE is the pose of the end-effector frame in the robot base frame.
    The API provides it as a column-major flat array, so reshape with order='F'.
    """
    state = robot.read_once()
    return np.array(state.O_T_EE).reshape((4, 4), order="F")


def main():
    """Runs D405 eye-in-hand calibration with a ChArUco target."""
    try:
        require_runtime_dependencies()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    print("=" * 70)
    print("  D405 EYE-IN-HAND EXTRINSIC CALIBRATION")
    print("=" * 70)

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
    poses_required = cfg.HAND_EYE_POSES_REQUIRED
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
    print("=" * 70 + "\n")

    captured_count = 0

    try:
        while captured_count < poses_required:
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
                print("      Pose saved")

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
