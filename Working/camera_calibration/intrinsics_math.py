"""
Intrinsic Calibration Math Module
----------------------------------
Handles all mathematical operations for camera lens calibration.

WHAT THIS FILE DOES:
====================
1. extract_factory_intrinsics() - Reads calibration stored in camera hardware EEPROM
2. calibrate_with_images() - Solves the math for 20-image calibration
3. validate_saved_intrinsics() - Checks if a saved JSON matches the hardware
4. load_intrinsics() / save_intrinsics() - JSON file I/O
5. get_opencv_matrices() - Converts dict to numpy arrays for OpenCV functions

WHY SEPARATE MATH FROM SCRIPTS?
================================
The calibration scripts (calibrate_intrinsics.py) handle user interaction,
camera control, and visualization. This file is PURE MATH - no user input,
no camera control, just calculations. This makes it testable and reusable.
"""

import json
import os
import numpy as np


MAX_IMAGE_CALIBRATION_REPROJECTION_ERROR_PX = 1.0


def _load_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "opencv-python is required for camera calibration. "
            "Install with: pip install opencv-contrib-python"
        ) from exc
    return cv2


def _load_rs():
    try:
        import pyrealsense2 as rs
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyrealsense2 is required to read RealSense factory intrinsics."
        ) from exc
    return rs


def extract_factory_intrinsics(camera_pipeline):
    """
    Reads the factory calibration stored in the RealSense camera's EEPROM.
    
    INTEL'S FACTORY CALIBRATION:
    =============================
    Every RealSense camera is calibrated at the factory before shipping.
    This calibration is stored in non-volatile memory on the camera itself.
    
    PROS:
    - Always available, no setup required
    - Decent accuracy for general use
    
    CONS:
    - May drift over time as the lens ages
    - Doesn't account for your specific mounting/usage conditions
    - Cannot be improved without re-running 20-image calibration
    
    Args:
        camera_pipeline: Active pyrealsense2 pipeline object
        
    Returns:
        dict: Dictionary containing fx, fy, ppx, ppy, distortion, width, height
    """
    rs = _load_rs()

    # Get the active stream profile to extract intrinsics
    profile = camera_pipeline.get_active_profile()
    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().intrinsics
    
    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,           # Focal length in x (pixels)
        "fy": intrinsics.fy,           # Focal length in y (pixels)
        "ppx": intrinsics.ppx,         # Principal point x (pixels)
        "ppy": intrinsics.ppy,         # Principal point y (pixels)
        "distortion": list(intrinsics.coeffs)  # k1, k2, p1, p2, k3 (radial + tangential)
    }


def calibrate_with_images(image_points_list, object_points_list, image_size):
    """
    Runs OpenCV's camera calibration solver on collected images.
    
    THE MATH BEHIND CALIBRATION:
    =============================
    Each image provides a set of 2D pixel coordinates (what we see) and
    their corresponding 3D world coordinates (what we know the board looks like).
    
    The solver finds the camera matrix K and distortion coefficients D that
    minimize the reprojection error:
    
        For each corner i: minimize || projected_3D_point - observed_2D_point ||
    
    The camera matrix K has the form:
        [fx  0   ppx]
        [0   fy  ppy]
        [0   0   1 ]
    
    where fx, fy are focal lengths and (ppx, ppy) is the principal point.
    
    Args:
        image_points_list: List of 2D pixel coordinate arrays, one per image
                          Each array has shape (N, 1, 2) where N is corners found
        object_points_list: List of 3D world coordinate arrays, one per image
                           Each array has shape (N, 3) - the known board geometry
        image_size: Tuple (width, height) of the captured images
        
    Returns:
        dict: Contains camera_matrix, distortion, width, height, reprojection_error
        
    Raises:
        RuntimeError: If OpenCV calibration fails (usually means bad input data)
    """
    cv2 = _load_cv2()

    # Run the OpenCV calibrator
    # This is an iterative optimization that typically converges in <1 second
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points_list,    # 3D points (known board geometry)
        image_points_list,     # 2D points (detected in images)
        image_size,            # Image dimensions
        None,                  # No initial guess for camera matrix
        None                   # No initial guess for distortion
    )
    
    # Validate the result
    if not ret or camera_matrix is None:
        raise RuntimeError(
            "OpenCV calibration failed. This usually means:\n"
            "  - Not enough images with sufficient corner coverage\n"
            "  - Board dimensions in config.py don't match your physical board\n"
            "  - Images are too blurry"
        )
    
    # Calculate reprojection error
    # This tells us how accurate the calibration is in pixels
    # < 0.5 pixels = excellent
    # 0.5-1.0 pixels = good
    # > 1.0 pixels = may need more/better images
    total_error = 0.0
    total_points = 0

    for obj_points, img_points, rvec, tvec in zip(
        object_points_list,
        image_points_list,
        rvecs,
        tvecs,
    ):
        projected_points, _ = cv2.projectPoints(
            obj_points,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        error = cv2.norm(img_points, projected_points, cv2.NORM_L2)
        total_error += error * error
        total_points += len(obj_points)

    error_px = (total_error / total_points) ** 0.5 if total_points else float("inf")
    print(f"  -> Calibration successful!")
    print(f"  -> Reprojection error: {error_px:.4f} pixels")
    
    if error_px > 1.0:
        print(f"  -> WARNING: Error > 1.0px. Consider capturing more images.")
    elif error_px < 0.3:
        print(f"  -> EXCELLENT: Error < 0.3px. Very high quality calibration.")
    
    return {
        "width": image_size[0],
        "height": image_size[1],
        "camera_matrix": camera_matrix.tolist(),
        "distortion": dist_coeffs.tolist(),
        "reprojection_error": float(error_px)
    }


def validate_saved_intrinsics(camera_pipeline, json_filepath):
    """
    Validates that a saved calibration file matches the current hardware.
    
    VALIDATION CHECKS:
    ===================
    1. File exists?
    2. Serial number matches? (Prevents using D435 calibration on D405)
    3. Resolution matches the active stream?
    4. For factory calibrations: do the parameters match the EEPROM?
    5. For image calibrations: was the saved reprojection error acceptable?
    
    Args:
        camera_pipeline: Active pyrealsense2 pipeline
        json_filepath: Path to the saved JSON file
        
    Returns:
        tuple: (is_valid: bool, reason: str)
    """
    # Check 1: File exists
    if not os.path.exists(json_filepath):
        return False, f"Calibration file not found: {json_filepath}"
    
    # Load the saved data
    saved_data = load_intrinsics(json_filepath)
    if saved_data is None:
        return False, f"Could not parse calibration file: {json_filepath}"
    
    # Determine calibration type
    is_image_calib = "camera_matrix" in saved_data
    
    # Check 2: Serial number present and matches
    rs = _load_rs()
    profile = camera_pipeline.get_active_profile()
    current_serial = profile.get_device().get_info(rs.camera_info.serial_number)
    live_intrinsics = extract_factory_intrinsics(camera_pipeline)
    
    if "serial" not in saved_data:
        return False, (
            "Old JSON format detected (missing serial number field). "
            "Delete the file and re-run calibration."
        )
        
    if saved_data["serial"] != current_serial:
        return False, (
            f"Serial number mismatch!\n"
            f"  File is for: {saved_data['serial']}\n"
            f"  Camera is:   {current_serial}\n"
            f"Delete the file and re-run calibration for this camera."
        )

    # Check 3: Resolution must match the stream being used now.
    for key in ("width", "height"):
        if key not in saved_data:
            return False, f"Calibration file is missing '{key}'. Re-run intrinsic calibration."
        if int(saved_data[key]) != int(live_intrinsics[key]):
            return False, (
                f"Calibration resolution mismatch for {key}:\n"
                f"  File:   {saved_data[key]}\n"
                f"  Active: {live_intrinsics[key]}\n"
                f"Use the same resolution or re-run intrinsic calibration."
            )
    
    # Check 3/4: Type-specific validation
    if is_image_calib:
        return _validate_image_calibration(saved_data)
    else:
        # Factory calibration: Verify parameters match EEPROM
        return _validate_factory_params(saved_data, camera_pipeline)


def _validate_image_calibration(saved_data):
    """
    Validates saved OpenCV image calibration data.

    A custom image calibration cannot be compared against the RealSense EEPROM;
    instead we check that the matrix fields are present and that the reprojection
    error saved at calibration time was low enough to trust.
    """
    try:
        camera_matrix, dist_coeffs = get_opencv_matrices(saved_data)
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"Invalid image calibration matrix fields: {exc}"

    if camera_matrix.shape != (3, 3):
        return False, f"Camera matrix must be 3x3, got {camera_matrix.shape}"
    if dist_coeffs.size == 0:
        return False, "Distortion coefficients are missing or empty."

    error = saved_data.get("reprojection_error")
    if error is None:
        return False, (
            "Image calibration is missing reprojection_error, so quality cannot be verified. "
            "Re-run intrinsic calibration."
        )

    try:
        error = float(error)
    except (TypeError, ValueError):
        return False, f"Invalid reprojection_error value: {error}"

    if not np.isfinite(error):
        return False, f"Invalid reprojection_error value: {error}"

    if error > MAX_IMAGE_CALIBRATION_REPROJECTION_ERROR_PX:
        return False, (
            f"Reprojection error is too high ({error:.4f}px). "
            f"Target is <= {MAX_IMAGE_CALIBRATION_REPROJECTION_ERROR_PX:.2f}px. "
            "Re-run intrinsic calibration with sharper, more varied images."
        )

    return True, f"Valid 20-image calibration (reprojection error: {error:.4f} px)"


def _validate_factory_params(saved_data, camera_pipeline):
    """
    Validates factory calibration parameters against hardware EEPROM.
    
    Args:
        saved_data: Dictionary loaded from JSON
        camera_pipeline: Active pyrealsense2 pipeline
        
    Returns:
        tuple: (is_valid: bool, reason: str)
    """
    live_intrinsics = extract_factory_intrinsics(camera_pipeline)
    
    # Check focal lengths and principal point
    params_to_check = ["fx", "fy", "ppx", "ppy"]
    for key in params_to_check:
        saved_val = saved_data.get(key)
        live_val = live_intrinsics.get(key)
        
        if saved_val is None or live_val is None:
            return False, f"Missing parameter '{key}' in saved file"
        
        # Use relative tolerance of 0.1% (1e-3)
        if not np.isclose(saved_val, live_val, rtol=1e-3):
            return False, (
                f"Parameter '{key}' has drifted:\n"
                f"  Saved: {saved_val}\n"
                f"  Live:  {live_val}\n"
                f"Re-run calibration."
            )
    
    # Check distortion coefficients
    saved_dist = saved_data.get("distortion", [])
    live_dist = live_intrinsics.get("distortion", [])
    
    if not np.allclose(saved_dist, live_dist, rtol=1e-3):
        return False, (
            "Distortion coefficients have changed.\n"
            "This can happen if the lens was physically bumped or damaged.\n"
            "Re-run calibration."
        )
        
    return True, "Valid: Factory calibration matches hardware EEPROM"


def load_intrinsics(filepath):
    """
    Loads intrinsic calibration dictionary from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        dict: Calibration data, or None if file doesn't exist/invalid
    """
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to load intrinsics from {filepath}: {e}")
        return None


def save_intrinsics(filepath, intrinsics_dict, camera_serial):
    """
    Saves intrinsic calibration dictionary to a JSON file.
    
    Injects the camera serial number for validation purposes.
    Creates the directory structure if it doesn't exist.
    
    Args:
        filepath: Where to save the JSON file
        intrinsics_dict: The calibration data dictionary
        camera_serial: Serial number of the camera (for validation)
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Add serial number to the data
    intrinsics_dict["serial"] = camera_serial
    
    # Write with pretty formatting for human readability
    with open(filepath, 'w') as f:
        json.dump(intrinsics_dict, f, indent=4)
    
    print(f"  -> Saved to: {filepath}")


def get_opencv_matrices(intrinsics_dict):
    """
    Converts a calibration dictionary into numpy arrays for OpenCV functions.
    
    OpenCV functions like solvePnP, undistort, etc. expect:
    - camera_matrix: 3x3 numpy array
    - dist_coeffs: 1D numpy array (typically 5 elements: k1, k2, p1, p2, k3)
    
    This function handles both formats:
    1. 20-image calibration: Has "camera_matrix" key with full 3x3 matrix
    2. Factory calibration: Has separate "fx", "fy", "ppx", "ppy" keys
    
    Args:
        intrinsics_dict: Dictionary from load_intrinsics() or extract_factory_intrinsics()
        
    Returns:
        tuple: (camera_matrix: np.ndarray, dist_coeffs: np.ndarray)
    """
    if "camera_matrix" in intrinsics_dict:
        # 20-image calibration format: full 3x3 matrix
        cam_matrix = np.array(intrinsics_dict["camera_matrix"], dtype=np.float64)
    else:
        # Factory calibration format: separate parameters
        cam_matrix = np.array([
            [intrinsics_dict["fx"], 0, intrinsics_dict["ppx"]],
            [0, intrinsics_dict["fy"], intrinsics_dict["ppy"]],
            [0, 0, 1]
        ], dtype=np.float64)
    
    # Distortion is always a 1D array
    dist_coeffs = np.array(intrinsics_dict["distortion"], dtype=np.float64)
    
    return cam_matrix, dist_coeffs
