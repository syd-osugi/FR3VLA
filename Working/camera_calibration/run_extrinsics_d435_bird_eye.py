"""
Script: D435 Bird's-Eye Extrinsic Calibration
---------------------------------------------
Calibrates the fixed overhead D435 camera using a ChArUco board mounted rigidly
to the robot end-effector.

WHAT THIS SCRIPT DOES:
======================
The D435 is fixed above the workspace. The board moves with the robot wrist.
For each captured pose:
  1. OpenCV estimates the board pose in the D435 camera frame.
  2. The Franka reports the end-effector pose in the robot base frame.
  3. The known board mount transform connects board frame to wrist frame.

Those measurements are combined into one direct transform:

    D435 optical frame -> robot base frame

MATRIX DEFINITIONS:
===================
All transforms are 4x4 homogeneous matrices. A transform named T_a_to_b maps a
point expressed in frame A into frame B:

    p_b = T_a_to_b @ p_a

The D435 mounted-board calibration uses these matrices:

  T_board_to_cam:
    - Measured from the D435 image at each captured pose.
    - OpenCV estimates this from interpolated ChArUco corners.
    - This changes every time the robot moves.

  T_board_to_ee:
    - Fixed physical mount from ChArUco board frame to end-effector frame.
    - Read from config.py.
    - This should not change unless the board is remounted.

  T_ee_to_base:
    - Current Franka end-effector pose in the robot base frame.
    - Read from the robot at each captured pose.
    - This changes every time the robot moves.

  T_cam_to_base:
    - Final desired D435 extrinsic transform.
    - Converts D435 depth points into robot base coordinates.
    - Computed for each pose, then averaged and saved.

For each pose:

    T_cam_to_base = T_ee_to_base @ T_board_to_ee @ inverse(T_board_to_cam)

This file is the only D435 extrinsic calibration script the user needs to run.
bird_eye_math.py is a helper module imported by this script.

PREREQUISITES:
==============
1. D435 intrinsic calibration must be complete.
2. The ChArUco board must be rigidly mounted to the robot end-effector.
3. The board-to-end-effector transform must be known or intentionally identity.
4. The D435 must remain fixed during and after calibration.

HOW TO USE:
===========
1. Mount the ChArUco board on the robot end-effector.
2. Run this script.
3. Confirm the board-to-end-effector mount values from config.py.
4. Move the robot through varied poses while the D435 sees the board.
5. Press 's' at each good pose until the required pose count is reached.
6. The script saves the configured D435 extrinsic file.
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
from camera_calibration.bird_eye_math import (
    average_transforms,
    camera_to_robot_from_pose,
    make_board_to_ee_transform,
    save_fixed_camera_transform,
    validate_transform_set,
)
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


def require_runtime_dependencies():
    """Checks runtime-only dependencies before starting hardware."""
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

    # Franka names this O_T_EE: pose of the End Effector (EE) in the robot
    # base/world frame (O). In this code's naming convention, that is
    # T_ee_to_base: it maps a point written in EE coordinates into base
    # coordinates.
    return np.array(state.O_T_EE).reshape((4, 4), order="F")


def confirm_board_to_ee_transform():
    """
    Confirms the configured ChArUco board frame -> end-effector frame transform.

    The values live in config.py because they describe the physical board mount.
    If they are wrong, stop calibration and update config before continuing.

    Matrix meaning:
        p_ee = T_board_to_ee @ p_board

    Because the board is securely mounted to the end effector, this matrix is
    constant across every calibration pose. The script does not estimate this
    mount from images; it trusts the measured config value after confirmation.
    """
    translation_m = tuple(cfg.BIRD_EYE_BOARD_TO_EE_TRANSLATION_M)
    rpy_deg = tuple(cfg.BIRD_EYE_BOARD_TO_EE_RPY_DEG)

    print("\n" + "=" * 70)
    print("  MOUNTED BOARD TO END-EFFECTOR TRANSFORM")
    print("=" * 70)
    print(
        """
  The D435 can see board -> camera, and the robot can report wrist -> base.
  To connect those measurements, the script needs board -> wrist.

  These values are read from config.py. Use all zeros only if the board
  coordinate frame is intentionally identical to the robot end-effector frame.
"""
    )

    # Build the fixed physical mount matrix:
    #   - translation_m is the board origin expressed in EE coordinates.
    #   - rpy_deg is the board frame orientation relative to the EE frame.
    T_board_to_ee = make_board_to_ee_transform(translation_m, rpy_deg)
    identity = all(abs(value) < 1e-12 for value in translation_m + rpy_deg)

    print("  Config values:")
    print(
        "    BIRD_EYE_BOARD_TO_EE_TRANSLATION_M = "
        f"{translation_m} meters"
    )
    print(
        "    BIRD_EYE_BOARD_TO_EE_RPY_DEG       = "
        f"{rpy_deg} degrees"
    )
    print("\n  Board -> end-effector transform:")
    for row in T_board_to_ee:
        print(f"    [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f} {row[3]:8.4f}]")

    while True:
        choice = input("  Are these config values correct? [y/n]: ")
        choice = choice.strip().lower()
        if choice in ("y", "yes"):
            break
        if choice in ("n", "no"):
            raise RuntimeError(
                "Update BIRD_EYE_BOARD_TO_EE_TRANSLATION_M and "
                "BIRD_EYE_BOARD_TO_EE_RPY_DEG in config.py before calibrating."
            )
        print("  Please answer y or n.")

    mount_metadata = {
        "source": "config.py",
        "translation_m": list(translation_m),
        "rpy_deg": list(rpy_deg),
        "identity_asserted": identity,
    }
    return T_board_to_ee, mount_metadata


def main():
    """Runs fixed-camera D435 extrinsic calibration."""
    try:
        require_runtime_dependencies()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    print("=" * 70)
    print("  D435 BIRD'S-EYE EXTRINSIC CALIBRATION")
    print("=" * 70)

    print(f"\n[1/5] Connecting to Franka robot at {cfg.FRANKA_IP}...")
    try:
        robot = connect_robot()
        print("      Robot connected")
    except Exception as exc:
        print(f"      ERROR: Could not connect to robot: {exc}")
        return 1

    print(f"\n[2/5] Loading D435 intrinsics from {cfg.INTRINSICS_D435_PATH}...")
    intrinsics = load_intrinsics(cfg.INTRINSICS_D435_PATH)
    if intrinsics is None:
        print("      ERROR: D435 intrinsics not found.")
        print("      Run Working/camera_calibration/calibrate_intrinsics.py first.")
        return 1

    camera_matrix, dist_coeffs = get_opencv_matrices(intrinsics)
    print("      Intrinsics loaded")

    print(f"\n[3/5] Confirming mounted-board setup...")
    try:
        T_board_to_ee, mount_metadata = confirm_board_to_ee_transform()
    except RuntimeError as exc:
        print(f"      ERROR: {exc}")
        return 1

    print(f"\n[4/5] Starting D435 camera...")
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_device(cfg.D435_SERIAL)
    rs_config.enable_stream(
        rs.stream.depth,
        cfg.D435_RESOLUTION[0],
        cfg.D435_RESOLUTION[1],
        rs.format.z16,
        cfg.CAMERA_FPS,
    )
    rs_config.enable_stream(
        rs.stream.color,
        cfg.D435_RESOLUTION[0],
        cfg.D435_RESOLUTION[1],
        rs.format.bgr8,
        cfg.CAMERA_FPS,
    )

    try:
        pipeline.start(rs_config)
        print(
            f"      Camera started at {cfg.D435_RESOLUTION[0]}x{cfg.D435_RESOLUTION[1]}"
        )
    except Exception as exc:
        print(f"      ERROR: Could not start D435 camera: {exc}")
        return 1

    time.sleep(cfg.CAMERA_WARMUP_SECONDS)

    # Reuse the shared robot-pose capture count instead of adding another knob.
    poses_required = cfg.HAND_EYE_POSES_REQUIRED

    aruco_dict = get_aruco_dictionary(cv2, cfg.BIRD_EYE_ARUCO_DICT_NAME)
    charuco_board = create_charuco_board(
        cv2,
        cfg.BIRD_EYE_BOARD_CORNERS,
        cfg.BIRD_EYE_SQUARE_SIZE,
        cfg.BIRD_EYE_MARKER_SIZE,
        aruco_dict,
    )

    camera_to_base_estimates = []
    captured_count = 0

    print("\n" + "=" * 70)
    print("  CAPTURE PHASE")
    print("=" * 70)
    print(
        f"""
  SETUP:
    - Keep the D435 fixed above the workspace.
    - Keep the ChArUco board rigidly mounted to the end-effector.
    - Move the robot so the D435 can see the board clearly.

  INSTRUCTIONS:
    - Move the robot to varied positions and wrist orientations.
    - Press 's' when the board is detected and the robot is still.
    - Capture {poses_required} poses.
    - Press 'q' to quit early after at least 3 good poses.

  TIPS:
    - Use poses across the workspace the D435 will actually observe.
    - Avoid only translating in a straight line; vary orientation too.
    - Discard any capture where the board is blurred or partly hidden.
"""
    )
    print("=" * 70 + "\n")

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

            display = color_image.copy()
            if detected:
                draw_charuco_detection(cv2, display, detection)
                draw_pose_axes(
                    cv2,
                    display,
                    T_board_to_cam,
                    camera_matrix,
                    dist_coeffs,
                    axis_length=0.08,
                )
                cv2.putText(
                    display,
                    "CHARUCO DETECTED - Press 's' to capture",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    display,
                    "Searching for mounted ChArUco board...",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
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

            cv2.imshow("D435 Bird's-Eye Calibration", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                if not detected:
                    print("  ChArUco board is not detected; capture skipped.")
                    continue

                print(
                    f"  [{captured_count + 1}/{poses_required}] "
                    "Capturing pose..."
                )

                # Capture the robot pose at the same moment as the image-based
                # board pose. For each saved sample:
                #   T_board_to_cam comes from the D435 image.
                #   T_board_to_ee is fixed from config.py.
                #   T_ee_to_base comes from the Franka encoders.
                #
                # Multiplying them gives one estimate of the fixed D435 camera
                # transform into robot base:
                #
                #   T_cam_to_base =
                #       T_ee_to_base @ T_board_to_ee @ inverse(T_board_to_cam)
                T_ee_to_base = get_robot_ee_pose(robot)
                T_cam_to_base = camera_to_robot_from_pose(
                    T_board_to_cam,
                    T_board_to_ee,
                    T_ee_to_base,
                )
                camera_to_base_estimates.append(T_cam_to_base)
                captured_count += 1
                print("      Pose saved")

            elif key == ord("q"):
                print("\n  Quitting early...")
                break

        if captured_count < 3:
            print(f"\nERROR: Need at least 3 poses, only captured {captured_count}.")
            return 1

        print(f"\n[5/5] Computing fixed-camera transform from {captured_count} poses...")
        T_cam_to_base = average_transforms(camera_to_base_estimates)
        validation = validate_transform_set(camera_to_base_estimates, T_cam_to_base)

        print("\n  Transform matrix (D435 camera -> robot base):")
        for row in T_cam_to_base:
            print(f"    [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f} {row[3]:8.4f}]")

        print("\n  Per-pose agreement:")
        print(
            "    Max translation spread: "
            f"{validation['max_translation_error_m'] * 1000:.2f} mm"
        )
        print(
            "    Mean translation spread: "
            f"{validation['mean_translation_error_m'] * 1000:.2f} mm"
        )
        print(
            "    Max rotation spread: "
            f"{validation['max_rotation_error_deg']:.3f} deg"
        )
        print(
            "    Mean rotation spread: "
            f"{validation['mean_rotation_error_deg']:.3f} deg"
        )

        robot_serial = f"franka@{cfg.FRANKA_IP}"
        save_fixed_camera_transform(
            cfg.BIRD_EYE_D435_PATH,
            T_cam_to_base,
            robot_serial,
            board_to_ee=T_board_to_ee,
            mount_metadata=mount_metadata,
            validation_results=validation,
        )

        print("\n" + "=" * 70)
        print("  CALIBRATION COMPLETE")
        print("=" * 70)
        print(f"\n  File saved: {cfg.BIRD_EYE_D435_PATH}")
        print("  Runtime D435 depth points can now transform directly to robot base.")

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
