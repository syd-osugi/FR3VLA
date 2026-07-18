"""
Camera Calibration Math Module
===============================
Shared mathematical helpers for camera extrinsic calibration.

This module provides:
  - ChArUco board creation and corner detection
  - Board pose estimation via solvePnP
  - Hand-eye (AX=XB) calibration for eye-in-hand cameras
  - Fixed-camera (bird's-eye) transform computation
  - Transform averaging and validation
  - JSON file I/O for calibration results

Frame convention:
  T_a_to_b maps a point from frame A into frame B.
  p_b = T_a_to_b @ p_a

Coordinate frames used in calibration:
  - charuco_board: origin at a corner of the ChArUco board, Z perpendicular to board
  - camera (optical): origin at pinhole, Z along optical axis, X right, Y down
  - ee (end-effector): Franka wrist/flange frame
  - base: robot base world frame
"""

import json
import math
import os
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# ChArUco helpers
# ---------------------------------------------------------------------------

def require_charuco_support(cv2):
    """Raise a clear error if the installed OpenCV lacks ChArUco support."""
    if cv2 is None:
        raise RuntimeError("opencv-contrib-python is required for ChArUco calibration.")
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "Your OpenCV build does not include cv2.aruco. "
            "Install opencv-contrib-python."
        )
    if not hasattr(cv2.aruco, "interpolateCornersCharuco"):
        raise RuntimeError(
            "Your OpenCV aruco module does not include ChArUco helpers. "
            "Install a recent opencv-contrib-python build."
        )


def get_aruco_dictionary(cv2, dict_name):
    """Convert a config dictionary name string into an OpenCV ArUco dictionary object."""
    dict_map = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    }
    dict_id = dict_map.get(dict_name, cv2.aruco.DICT_4X4_50)
    return cv2.aruco.getPredefinedDictionary(dict_id)


def create_charuco_board(cv2, board_corners, square_size, marker_size, aruco_dict, legacy_pattern=False):
    """
    Create a ChArUco board object across OpenCV API versions.

    Args:
        cv2: OpenCV module.
        board_corners: Tuple (inner_corners_x, inner_corners_y).
        square_size: Physical size of one chessboard square in meters.
        marker_size: Physical size of one ArUco marker in meters.
        aruco_dict: OpenCV ArUco dictionary object.
        legacy_pattern: Use pre-4.6 legacy marker layout.

    Returns:
        cv2.aruco.CharucoBoard object.
    """
    squares_x = board_corners[0] + 1
    squares_y = board_corners[1] + 1

    if hasattr(cv2.aruco, "CharucoBoard_create"):
        board = cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_size, marker_size, aruco_dict)
    else:
        board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_size, marker_size, aruco_dict)

    if legacy_pattern and hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(True)

    return board


def get_charuco_object_points(charuco_board, charuco_ids):
    """
    Return 3D board-frame coordinates for detected ChArUco corner IDs.

    The board lies in Z=0. The returned object points are paired with the 2D
    detected ChArUco corners and used by solvePnP to compute board -> camera.

    Args:
        charuco_board: ChArUco board object.
        charuco_ids: 1D array of detected corner IDs.

    Returns:
        np.ndarray: 3D object points of shape (N, 1, 3).
    """
    if hasattr(charuco_board, "getChessboardCorners"):
        all_corners = charuco_board.getChessboardCorners()
    else:
        all_corners = charuco_board.chessboardCorners
    return all_corners[charuco_ids.flatten()].astype(np.float32)


def detect_charuco_corners(cv2, gray_image, aruco_dict, charuco_board,
                           camera_matrix=None, dist_coeffs=None,
                           min_markers=4, min_corners=4):
    """
    Detect ArUco markers and interpolate ChArUco corners.

    Args:
        cv2: OpenCV module.
        gray_image: Grayscale camera frame.
        aruco_dict: ArUco dictionary.
        charuco_board: ChArUco board object.
        camera_matrix: 3x3 camera intrinsic matrix (optional, helps interpolation).
        dist_coeffs: Distortion coefficients (optional).
        min_markers: Minimum ArUco markers required.
        min_corners: Minimum ChArUco corners required.

    Returns:
        dict with keys: success, marker_corners, marker_ids, rejected_markers,
                        charuco_corners, charuco_ids, marker_count, charuco_count, reason.
    """
    if hasattr(cv2.aruco, "DetectorParameters"):
        detector_params = cv2.aruco.DetectorParameters()
    else:
        detector_params = cv2.aruco.DetectorParameters_create()

    corners, ids, rejected = cv2.aruco.detectMarkers(gray_image, aruco_dict, parameters=detector_params)
    marker_count = 0 if ids is None else len(ids)

    result = {
        "success": False,
        "marker_corners": corners,
        "marker_ids": ids,
        "rejected_markers": rejected,
        "charuco_corners": None,
        "charuco_ids": None,
        "marker_count": marker_count,
        "charuco_count": 0,
        "reason": f"Need at least {min_markers} markers",
    }

    if marker_count < min_markers:
        return result

    # Refine detected markers if supported
    try:
        corners, ids, rejected = cv2.aruco.refineDetectedMarkers(
            gray_image, charuco_board, corners, ids, rejected
        )
        marker_count = 0 if ids is None else len(ids)
        result.update({
            "marker_corners": corners, "marker_ids": ids,
            "rejected_markers": rejected, "marker_count": marker_count,
        })
    except Exception:
        pass

    if marker_count < min_markers:
        return result

    # Interpolate ChArUco corners
    if camera_matrix is not None and dist_coeffs is not None:
        try:
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray_image, charuco_board, camera_matrix, dist_coeffs
            )
        except Exception:
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray_image, charuco_board
            )
    else:
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray_image, charuco_board
        )

    charuco_count = 0 if charuco_ids is None else len(charuco_ids)
    result.update({
        "marker_corners": corners, "marker_ids": ids,
        "rejected_markers": rejected,
        "charuco_corners": charuco_corners, "charuco_ids": charuco_ids,
        "marker_count": marker_count, "charuco_count": charuco_count,
        "reason": f"Need at least {min_corners} ChArUco corners",
    })

    if not retval or charuco_corners is None or charuco_ids is None:
        return result
    if charuco_count < min_corners:
        return result

    result["success"] = True
    result["reason"] = f"{marker_count} markers, {charuco_count} ChArUco corners"
    return result


def detect_charuco_board_pose(cv2, gray_image, aruco_dict, charuco_board,
                              camera_matrix, dist_coeffs,
                              min_markers=4, min_corners=6):
    """
    Detect a ChArUco board and estimate T_board_to_camera.

    Args:
        cv2: OpenCV module.
        gray_image: Grayscale camera frame.
        aruco_dict: ArUco dictionary.
        charuco_board: ChArUco board object.
        camera_matrix: 3x3 camera intrinsic matrix.
        dist_coeffs: Distortion coefficients.
        min_markers: Minimum ArUco markers required.
        min_corners: Minimum ChArUco corners required for solvePnP.

    Returns:
        dict with keys: success, T_board_to_cam (4x4 np.ndarray or None),
                        marker_corners, marker_ids, charuco_corners, charuco_ids,
                        marker_count, charuco_count, reason.
    """
    corner_detection = detect_charuco_corners(
        cv2, gray_image, aruco_dict, charuco_board,
        camera_matrix=camera_matrix, dist_coeffs=dist_coeffs,
        min_markers=min_markers, min_corners=min_corners,
    )
    empty_result = dict(corner_detection)
    empty_result["T_board_to_cam"] = None

    if not corner_detection["success"]:
        return empty_result

    charuco_corners = corner_detection["charuco_corners"]
    charuco_ids = corner_detection["charuco_ids"]
    object_points = get_charuco_object_points(charuco_board, charuco_ids)

    if len(object_points) < 6:
        empty_result["success"] = False
        empty_result["reason"] = f"Need at least 6 ChArUco corners for pose, got {len(object_points)}"
        return empty_result

    retval, rvec, tvec = cv2.solvePnP(object_points, charuco_corners, camera_matrix, dist_coeffs)
    if not retval or rvec is None or tvec is None:
        empty_result["reason"] = "solvePnP failed"
        return empty_result

    rotation, _ = cv2.Rodrigues(rvec)
    T_board_to_cam = np.eye(4, dtype=float)
    T_board_to_cam[:3, :3] = rotation
    T_board_to_cam[:3, 3] = tvec.flatten()

    result = dict(corner_detection)
    result["T_board_to_cam"] = T_board_to_cam
    return result


def draw_charuco_detection(cv2, image, detection):
    """Draw detected ArUco markers and interpolated ChArUco corners on an image."""
    marker_corners = detection.get("marker_corners")
    marker_ids = detection.get("marker_ids")
    charuco_corners = detection.get("charuco_corners")
    charuco_ids = detection.get("charuco_ids")

    if marker_corners is not None and marker_ids is not None:
        cv2.aruco.drawDetectedMarkers(image, marker_corners, marker_ids)
    if charuco_corners is not None and charuco_ids is not None:
        cv2.aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids, (255, 0, 255))


def draw_pose_axes(cv2, image, T_board_to_cam, camera_matrix, dist_coeffs, axis_length=0.05):
    """Draw board-frame axes (X=red, Y=green, Z=blue) on an image using a board->camera pose."""
    if not hasattr(cv2, "drawFrameAxes"):
        return
    rvec, _ = cv2.Rodrigues(T_board_to_cam[:3, :3])
    tvec = T_board_to_cam[:3, 3]
    cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, axis_length)


# ---------------------------------------------------------------------------
# Hand-eye calibration (AX=XB)
# ---------------------------------------------------------------------------

def calibrate_hand_eye(robot_poses, camera_poses, method=None):
    """
    Solve the AX=XB hand-eye calibration problem.

    Given N pose pairs:
      robot_poses[i] = T_gripper_to_base[i]  (end-effector -> base)
      camera_poses[i] = T_target_to_cam[i]   (board -> camera)

    Compute X = T_cam_to_gripper such that:
      T_gripper_to_base[i] @ X @ T_target_to_cam[i] ≈ T_target_to_base (constant)

    Args:
        robot_poses: List of 4x4 np.ndarray (T_ee_to_base).
        camera_poses: List of 4x4 np.ndarray (T_board_to_cam).
        method: OpenCV hand-eye method constant (default: TSAI).

    Returns:
        np.ndarray: 4x4 T_cam_to_ee (camera frame -> end-effector frame).

    Raises:
        ValueError: If inputs are invalid or mismatched.
    """
    if len(robot_poses) != len(camera_poses):
        raise ValueError(f"Pose count mismatch: {len(robot_poses)} robot vs {len(camera_poses)} camera")
    if len(robot_poses) < 3:
        raise ValueError(f"Need at least 3 poses for hand-eye calibration, got {len(robot_poses)}")

    R_g2b = [p[:3, :3] for p in robot_poses]
    t_g2b = [p[:3, 3].reshape(3, 1) for p in robot_poses]
    R_t2c = [p[:3, :3] for p in camera_poses]
    t_t2c = [p[:3, 3].reshape(3, 1) for p in camera_poses]

    if method is None:
        method = cv2.CALIB_HAND_EYE_TSAI

    R_cam2ee, t_cam2ee = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=method)

    T_cam_to_ee = np.eye(4, dtype=float)
    T_cam_to_ee[:3, :3] = R_cam2ee
    T_cam_to_ee[:3, 3] = t_cam2ee.flatten()
    return T_cam_to_ee


def validate_hand_eye_result(T_cam_to_ee, robot_poses, camera_poses,
                             tolerance_deg=5.0, tolerance_m=0.02):
    """
    Validate hand-eye calibration by checking consistency across all poses.

    For each pose, compute:
      T_target_to_base = T_ee_to_base @ T_cam_to_ee @ T_board_to_cam

    If calibration is correct, every pose should predict the same fixed target pose.

    Args:
        T_cam_to_ee: Computed 4x4 transform.
        robot_poses: List of T_ee_to_base matrices.
        camera_poses: List of T_board_to_cam matrices.
        tolerance_deg: Max allowed rotation error in degrees.
        tolerance_m: Max allowed translation error in meters.

    Returns:
        tuple: (is_valid, max_rotation_error_deg, max_translation_error_m)
    """
    target_to_base_poses = [g @ T_cam_to_ee @ c for g, c in zip(robot_poses, camera_poses)]

    rotations = np.array([p[:3, :3] for p in target_to_base_poses], dtype=float)
    translations = np.array([p[:3, 3] for p in target_to_base_poses], dtype=float)

    raw_rotation = np.mean(rotations, axis=0)
    u, _, vt = np.linalg.svd(raw_rotation)
    reference_rotation = u @ vt
    if np.linalg.det(reference_rotation) < 0:
        u[:, -1] *= -1
        reference_rotation = u @ vt
    reference_translation = np.mean(translations, axis=0)

    rot_errors = []
    trans_errors = []
    for target_to_base in target_to_base_poses:
        rotation_delta = reference_rotation.T @ target_to_base[:3, :3]
        trace = np.trace(rotation_delta)
        angle_rad = float(np.arccos(np.clip((trace - 1) / 2, -1, 1)))
        rot_errors.append(np.degrees(angle_rad))
        trans_errors.append(float(np.linalg.norm(target_to_base[:3, 3] - reference_translation)))

    max_rot = max(rot_errors)
    max_trans = max(trans_errors)
    return (max_rot <= tolerance_deg) and (max_trans <= tolerance_m), max_rot, max_trans


def save_hand_eye(filepath, matrix_4x4, camera_serial, validation_results=None):
    """
    Save a 4x4 hand-eye calibration matrix to JSON.

    Args:
        filepath: Output file path.
        matrix_4x4: 4x4 np.ndarray transform.
        camera_serial: Serial number of the calibrated camera.
        validation_results: Optional (is_valid, rot_err, trans_err) tuple.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data = {
        "description": "Hand-eye calibration: transform from camera frame to robot end-effector frame",
        "from_frame": "camera_optical",
        "to_frame": "ee",
        "camera_serial": camera_serial,
        "matrix": matrix_4x4.tolist(),
    }
    if validation_results is not None:
        is_valid, rot_err, trans_err = validation_results
        data["validation"] = {
            "is_valid": is_valid,
            "max_rotation_error_deg": float(rot_err),
            "max_translation_error_m": float(trans_err),
        }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    print(f"  -> Saved hand-eye calibration to: {filepath}")


# ---------------------------------------------------------------------------
# Fixed-camera (bird's-eye) calibration helpers
# ---------------------------------------------------------------------------

def rotation_matrix_from_rpy_deg(roll_deg, pitch_deg, yaw_deg):
    """Build R = Rz(yaw) @ Ry(pitch) @ Rx(roll) from degree inputs."""
    roll, pitch, yaw = [math.radians(v) for v in (roll_deg, pitch_deg, yaw_deg)]
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def make_board_to_ee_transform(translation_m, rpy_deg):
    """
    Create 4x4 T_board_to_ee from translation (meters) and RPY (degrees).

    Args:
        translation_m: Board origin in EE coordinates, (x, y, z) in meters.
        rpy_deg: Board orientation relative to EE frame as roll, pitch, yaw in degrees.

    Returns:
        np.ndarray: 4x4 homogeneous transform.
    """
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rotation_matrix_from_rpy_deg(*rpy_deg)
    matrix[:3, 3] = np.array(translation_m, dtype=float)
    return matrix


def camera_to_robot_from_pose(T_board_to_cam, T_board_to_ee, T_ee_to_base):
    """
    Compute T_cam_to_base for one captured pose.

    T_cam_to_base = T_ee_to_base @ T_board_to_ee @ inv(T_board_to_cam)

    Args:
        T_board_to_cam: Board -> camera from solvePnP.
        T_board_to_ee: Fixed board -> EE mount transform from config.
        T_ee_to_base: End-effector -> base from robot encoders.

    Returns:
        np.ndarray: 4x4 T_cam_to_base.
    """
    T_cam_to_board = np.linalg.inv(T_board_to_cam)
    return T_ee_to_base @ T_board_to_ee @ T_cam_to_board


def average_transforms(transforms):
    """
    Average a list of 4x4 transforms. Translation is mean-averaged;
    rotation is averaged then projected onto SO(3) via SVD.

    Args:
        transforms: List of 4x4 np.ndarray.

    Returns:
        np.ndarray: 4x4 averaged transform.
    """
    if not transforms:
        raise ValueError("Need at least one transform to average")

    rotations = np.array([t[:3, :3] for t in transforms], dtype=float)
    translations = np.array([t[:3, 3] for t in transforms], dtype=float)

    raw_rotation = np.mean(rotations, axis=0)
    u, _, vt = np.linalg.svd(raw_rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt

    averaged = np.eye(4, dtype=float)
    averaged[:3, :3] = rotation
    averaged[:3, 3] = np.mean(translations, axis=0)
    return averaged


def validate_transform_set(transforms, reference_transform=None):
    """
    Report how tightly per-pose camera->robot estimates agree.

    Args:
        transforms: List of 4x4 np.ndarray estimates.
        reference_transform: Reference transform (computed from mean if None).

    Returns:
        dict with mean/max translation and rotation errors.
    """
    if not transforms:
        raise ValueError("Need at least one transform to validate")

    reference = reference_transform if reference_transform is not None else average_transforms(transforms)

    translation_errors = []
    rotation_errors = []
    for t in transforms:
        translation_errors.append(float(np.linalg.norm(t[:3, 3] - reference[:3, 3])))
        delta = reference[:3, :3].T @ t[:3, :3]
        cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
        rotation_errors.append(math.degrees(math.acos(cosine)))

    return {
        "pose_count": len(transforms),
        "mean_translation_error_m": float(np.mean(translation_errors)),
        "max_translation_error_m": float(np.max(translation_errors)),
        "mean_rotation_error_deg": float(np.mean(rotation_errors)),
        "max_rotation_error_deg": float(np.max(rotation_errors)),
    }


def save_fixed_camera_transform(filepath, matrix, camera_serial,
                                board_to_ee=None, mount_metadata=None,
                                validation_results=None):
    """
    Save a fixed-camera extrinsic calibration (camera -> base) to JSON.

    Args:
        filepath: Output file path.
        matrix: 4x4 T_cam_to_base transform.
        camera_serial: Serial number of the calibrated camera.
        board_to_ee: Board-to-EE mount transform used during calibration.
        mount_metadata: Dict with mount source, translation, RPY.
        validation_results: Dict from validate_transform_set().
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    payload = {
        "description": (
            "Fixed-camera extrinsic calibration: transform from camera optical frame "
            "to robot base frame"
        ),
        "from_frame": "camera_optical",
        "to_frame": "base",
        "camera_serial": camera_serial,
        "matrix": matrix.tolist(),
        "board_to_ee_matrix_used_for_calibration": board_to_ee.tolist() if board_to_ee is not None else None,
        "board_to_ee_mount": mount_metadata,
        "validation": validation_results,
    }
    with open(filepath, "w") as f:
        json.dump(payload, f, indent=4)
    print(f"  -> Saved fixed-camera calibration to: {filepath}")


# ---------------------------------------------------------------------------
# Intrinsic calibration helpers
# ---------------------------------------------------------------------------

def extract_factory_intrinsics(camera_pipeline):
    """
    Read factory calibration from RealSense camera EEPROM.

    Args:
        camera_pipeline: Active pyrealsense2 pipeline.

    Returns:
        dict with keys: width, height, fx, fy, ppx, ppy, distortion.
    """
    import pyrealsense2 as rs
    profile = camera_pipeline.get_active_profile()
    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().intrinsics
    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "ppx": intrinsics.ppx,
        "ppy": intrinsics.ppy,
        "distortion": list(intrinsics.coeffs),
    }


def calibrate_with_images(image_points_list, object_points_list, image_size):
    """
    Run OpenCV camera calibration on collected 2D/3D point correspondences.

    Args:
        image_points_list: List of 2D pixel arrays, one per image.
        object_points_list: List of 3D world arrays, one per image.
        image_size: (width, height) tuple.

    Returns:
        dict with camera_matrix, distortion, width, height, reprojection_error.
    """
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points_list, image_points_list, image_size, None, None
    )
    if not ret or camera_matrix is None:
        raise RuntimeError("OpenCV calibration failed. Check input data quality.")

    total_error = 0.0
    total_points = 0
    for obj_pts, img_pts, rvec, tvec in zip(object_points_list, image_points_list, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
        total_error += cv2.norm(img_pts, projected, cv2.NORM_L2) ** 2
        total_points += len(obj_pts)

    error_px = (total_error / total_points) ** 0.5 if total_points else float("inf")
    return {
        "width": image_size[0],
        "height": image_size[1],
        "camera_matrix": camera_matrix.tolist(),
        "distortion": dist_coeffs.tolist(),
        "reprojection_error": float(error_px),
    }


def load_intrinsics(filepath):
    """Load intrinsic calibration dictionary from JSON. Returns None if file missing."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def save_intrinsics(filepath, intrinsics_dict, camera_serial):
    """Save intrinsic calibration to JSON, injecting serial number."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data = dict(intrinsics_dict)
    data["serial"] = camera_serial
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    print(f"  -> Saved intrinsics to: {filepath}")


def get_opencv_matrices(intrinsics_dict):
    """
    Convert a calibration dict to numpy arrays for OpenCV.

    Handles both 20-image calibration format (camera_matrix key) and
    factory calibration format (fx, fy, ppx, ppy keys).

    Returns:
        tuple: (camera_matrix 3x3 np.ndarray, dist_coeffs 1D np.ndarray)
    """
    if "camera_matrix" in intrinsics_dict:
        cam_matrix = np.array(intrinsics_dict["camera_matrix"], dtype=np.float64)
    else:
        cam_matrix = np.array([
            [intrinsics_dict["fx"], 0, intrinsics_dict["ppx"]],
            [0, intrinsics_dict["fy"], intrinsics_dict["ppy"]],
            [0, 0, 1],
        ], dtype=np.float64)
    dist_coeffs = np.array(intrinsics_dict["distortion"], dtype=np.float64)
    return cam_matrix, dist_coeffs
