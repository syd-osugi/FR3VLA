"""
Shared ChArUco Calibration Helpers
----------------------------------
Utilities used by camera calibration scripts that estimate a board pose.

This file is not a user-run script. It exists so intrinsic calibration, D405
hand-eye calibration, and D435 bird's-eye calibration can use the same ChArUco
conventions.

Frame convention:
    T_a_to_b maps a point from frame A into frame B.

For a detected ChArUco board:
    p_camera = T_board_to_camera @ p_board

Why ChArUco:
    A ChArUco board combines ArUco marker IDs with chessboard-like corners.
    Marker IDs make detection robust, while interpolated ChArUco corners give
    accurate sub-pixel geometry for pose estimation.
"""

import numpy as np


def require_charuco_support(cv2):
    """Raises a clear error if the installed OpenCV lacks ChArUco support."""
    if cv2 is None:
        raise RuntimeError(
            "opencv-contrib-python is required for ChArUco calibration."
        )
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
    """Converts a config dictionary name into an OpenCV ArUco dictionary."""
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


def create_charuco_board(cv2, board_corners, square_size, marker_size, aruco_dict):
    """
    Creates a ChArUco board object across OpenCV API versions.

    config.py stores the number of inner chessboard corners because that is how
    calibration boards are usually described. OpenCV expects square counts, so
    each dimension is one larger than the inner-corner count.
    """
    squares_x = board_corners[0] + 1
    squares_y = board_corners[1] + 1

    if hasattr(cv2.aruco, "CharucoBoard_create"):
        return cv2.aruco.CharucoBoard_create(
            squares_x,
            squares_y,
            square_size,
            marker_size,
            aruco_dict,
        )

    return cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_size,
        marker_size,
        aruco_dict,
    )


def get_charuco_object_points(charuco_board, charuco_ids):
    """
    Returns 3D board-frame coordinates for detected ChArUco corner IDs.

    The board lies in Z=0. The returned object points are paired with the 2D
    detected ChArUco corners and used by solvePnP to compute board -> camera.
    """
    if hasattr(charuco_board, "getChessboardCorners"):
        all_corners = charuco_board.getChessboardCorners()
    else:
        all_corners = charuco_board.chessboardCorners

    return all_corners[charuco_ids.flatten()].astype(np.float32)


def _refine_detected_markers(cv2, gray_image, charuco_board, corners, ids, rejected):
    """
    Refines marker detection when the OpenCV build supports it.

    Different OpenCV versions return slightly different tuple lengths. This
    helper keeps the calibration scripts independent of that API wrinkle.
    """
    try:
        refined = cv2.aruco.refineDetectedMarkers(
            gray_image,
            charuco_board,
            corners,
            ids,
            rejected,
        )
        return refined[:3]
    except Exception:
        return corners, ids, rejected


def _interpolate_charuco_corners(
    cv2,
    corners,
    ids,
    gray_image,
    charuco_board,
    camera_matrix,
    dist_coeffs,
):
    """
    Interpolates ChArUco chessboard corners from detected ArUco markers.

    Some OpenCV versions accept camera intrinsics here and some do not. Both
    paths produce ChArUco corner IDs that are later solved with solvePnP.
    """
    try:
        return cv2.aruco.interpolateCornersCharuco(
            corners,
            ids,
            gray_image,
            charuco_board,
            camera_matrix,
            dist_coeffs,
        )
    except Exception:
        return cv2.aruco.interpolateCornersCharuco(
            corners,
            ids,
            gray_image,
            charuco_board,
        )


def detect_charuco_board_pose(
    cv2,
    gray_image,
    aruco_dict,
    charuco_board,
    camera_matrix,
    dist_coeffs,
    min_markers=4,
    min_corners=4,
):
    """
    Detects a ChArUco board and estimates T_board_to_camera.

    Returns a dictionary with:
        success: bool
        T_board_to_cam: 4x4 matrix if success else None
        marker_corners / marker_ids: raw ArUco marker detections
        charuco_corners / charuco_ids: interpolated ChArUco corners

    The board can be flat on the table for D405 hand-eye calibration or mounted
    to the wrist for D435 bird's-eye calibration. The pose math is identical.
    """
    corners, ids, rejected = cv2.aruco.detectMarkers(gray_image, aruco_dict)

    empty_result = {
        "success": False,
        "T_board_to_cam": None,
        "marker_corners": corners,
        "marker_ids": ids,
        "charuco_corners": None,
        "charuco_ids": None,
    }

    if ids is None or len(ids) < min_markers:
        return empty_result

    corners, ids, rejected = _refine_detected_markers(
        cv2,
        gray_image,
        charuco_board,
        corners,
        ids,
        rejected,
    )

    retval, charuco_corners, charuco_ids = _interpolate_charuco_corners(
        cv2,
        corners,
        ids,
        gray_image,
        charuco_board,
        camera_matrix,
        dist_coeffs,
    )

    if (
        not retval
        or charuco_corners is None
        or charuco_ids is None
        or len(charuco_ids) < min_corners
    ):
        empty_result["marker_corners"] = corners
        empty_result["marker_ids"] = ids
        return empty_result

    object_points = get_charuco_object_points(charuco_board, charuco_ids)

    # solvePnP gives the transform from board coordinates to camera coordinates:
    #     p_camera = T_board_to_cam @ p_board
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        charuco_corners,
        camera_matrix,
        dist_coeffs,
    )

    if not success:
        empty_result["marker_corners"] = corners
        empty_result["marker_ids"] = ids
        empty_result["charuco_corners"] = charuco_corners
        empty_result["charuco_ids"] = charuco_ids
        return empty_result

    rotation, _ = cv2.Rodrigues(rvec)
    T_board_to_cam = np.eye(4, dtype=float)
    T_board_to_cam[:3, :3] = rotation
    T_board_to_cam[:3, 3] = tvec.flatten()

    return {
        "success": True,
        "T_board_to_cam": T_board_to_cam,
        "marker_corners": corners,
        "marker_ids": ids,
        "charuco_corners": charuco_corners,
        "charuco_ids": charuco_ids,
    }


def draw_charuco_detection(cv2, image, detection):
    """Draws detected ArUco markers and interpolated ChArUco corners."""
    marker_corners = detection.get("marker_corners")
    marker_ids = detection.get("marker_ids")
    charuco_corners = detection.get("charuco_corners")
    charuco_ids = detection.get("charuco_ids")

    if marker_corners is not None and marker_ids is not None:
        cv2.aruco.drawDetectedMarkers(image, marker_corners, marker_ids)

    if charuco_corners is not None and charuco_ids is not None:
        cv2.aruco.drawDetectedCornersCharuco(
            image,
            charuco_corners,
            charuco_ids,
            (255, 0, 255),
        )


def draw_pose_axes(cv2, image, T_board_to_cam, camera_matrix, dist_coeffs, axis_length):
    """Draws board-frame axes using a board -> camera pose."""
    if not hasattr(cv2, "drawFrameAxes"):
        return

    rvec, _ = cv2.Rodrigues(T_board_to_cam[:3, :3])
    tvec = T_board_to_cam[:3, 3]
    cv2.drawFrameAxes(
        image,
        camera_matrix,
        dist_coeffs,
        rvec,
        tvec,
        axis_length,
    )
