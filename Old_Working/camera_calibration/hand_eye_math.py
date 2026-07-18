"""
Hand-Eye Calibration Math Module (AX=XB Problem)
--------------------------------------------------
Computes the rigid transform between a camera mounted on a robot wrist
and the robot's end-effector frame.

THE AX=XB PROBLEM:
===================
If we denote:
  - T_gripper_to_base = End-effector pose from robot encoders
  - T_target_to_cam = Calibration target pose from the camera image
  - T_cam_to_gripper = Camera-to-wrist transform we are solving for

Then each captured pose should predict the same fixed target location:

    T_target_to_base = T_gripper_to_base * T_cam_to_gripper * T_target_to_cam

OpenCV solves the equivalent AX=XB motion problem internally. The result tells
us where the camera is relative to the robot wrist.
This is essential because:
  - The robot knows where its wrist is
  - The camera sees where objects are
  - We need to connect these two coordinate systems

WHY MULTIPLE POSES?
===================
A single pose gives us 6 equations but X has 6 unknowns (3 rotation, 3 translation).
However, the equations are nonlinear and have local minima.
Multiple poses (typically 15-20) provide redundant constraints that make the
solution unique and robust to noise.

METHODS AVAILABLE:
==================
OpenCV provides several algorithms:
  - TSAI: Classic method, good general performance [DEFAULT]
  - PARK: Alternative formulation
  - HORAUD: Uses quaternions
  - ANDREFF: Analytical solution
  - DANIELIDIS: Uses dual quaternions
"""

import json
import os
import numpy as np


def _load_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "opencv-python is required for hand-eye calibration. "
            "Install with: pip install opencv-contrib-python"
        ) from exc
    return cv2


def calibrate_hand_eye(robot_poses, camera_poses, method=None):
    """
    Solves the AX=XB hand-eye calibration problem.
    
    Args:
        robot_poses: List of 4x4 transformation matrices (end-effector to base)
                    Each matrix is a numpy array with shape (4, 4)
        camera_poses: List of 4x4 transformation matrices (target to camera)
                     Each matrix is a numpy array with shape (4, 4)
        method: OpenCV hand-eye method constant, or None for default (TSAI)
        
    Returns:
        numpy.ndarray: 4x4 transformation matrix from camera to end-effector
        
    Raises:
        ValueError: If inputs are invalid or have mismatched lengths
    """
    cv2 = _load_cv2()

    # Validate inputs
    if len(robot_poses) != len(camera_poses):
        raise ValueError(
            f"Pose count mismatch: {len(robot_poses)} robot poses vs "
            f"{len(camera_poses)} camera poses"
        )
    
    if len(robot_poses) < 3:
        raise ValueError(
            f"Need at least 3 poses for hand-eye calibration, got {len(robot_poses)}"
        )
    
    # Extract rotation matrices and translation vectors
    # OpenCV's calibrateHandEye expects separate R and t lists
    R_gripper2base = []  # Rotation: gripper to base
    t_gripper2base = []  # Translation: gripper origin in base frame
    R_target2cam = []    # Rotation: target to camera
    t_target2cam = []    # Translation: target to camera
    
    for pose in robot_poses:
        R_gripper2base.append(pose[:3, :3])
        t_gripper2base.append(pose[:3, 3].reshape(3, 1))
        
    for pose in camera_poses:
        R_target2cam.append(pose[:3, :3])
        t_target2cam.append(pose[:3, 3].reshape(3, 1))
    
    # Select calibration method
    if method is None:
        method = cv2.CALIB_HAND_EYE_TSAI
    
    # Run the calibration solver
    # This returns R and t separately
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=method
    )
    
    # Combine into a 4x4 homogeneous transformation matrix
    T_cam2gripper = np.eye(4)
    T_cam2gripper[:3, :3] = R_cam2gripper
    T_cam2gripper[:3, 3] = t_cam2gripper.flatten()
    
    return T_cam2gripper


def validate_hand_eye_result(T_cam2gripper, robot_poses, camera_poses, tolerance_deg=5.0, tolerance_m=0.02):
    """
    Validates a hand-eye calibration result by checking consistency across all poses.
    
    For each pose, we compute:

        target -> base = gripper -> base @ camera -> gripper @ target -> camera

    If the calibration is correct, every pose should predict the same fixed
    target pose in the robot base frame.
    
    Args:
        T_cam2gripper: The computed 4x4 transform
        robot_poses: List of robot poses used for calibration
        camera_poses: List of camera poses used for calibration
        tolerance_deg: Maximum allowed rotation error in degrees
        tolerance_m: Maximum allowed translation error in meters
        
    Returns:
        tuple: (is_valid: bool, max_rotation_error_deg: float, max_translation_error_m: float)
    """
    target_to_base_poses = []
    for gripper_to_base, target_to_camera in zip(robot_poses, camera_poses):
        target_to_base_poses.append(
            gripper_to_base @ T_cam2gripper @ target_to_camera
        )

    rotations = np.array([pose[:3, :3] for pose in target_to_base_poses], dtype=float)
    translations = np.array([pose[:3, 3] for pose in target_to_base_poses], dtype=float)

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
        angle_rad = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        rot_errors.append(float(np.degrees(angle_rad)))
        trans_errors.append(
            float(np.linalg.norm(target_to_base[:3, 3] - reference_translation))
        )

    max_rot_error = max(rot_errors)
    max_trans_error = max(trans_errors)
    
    is_valid = (max_rot_error <= tolerance_deg) and (max_trans_error <= tolerance_m)
    return is_valid, max_rot_error, max_trans_error


def load_hand_eye(filepath):
    """
    Loads a 4x4 hand-eye transformation matrix from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        numpy.ndarray: 4x4 transformation matrix, or None if file doesn't exist
    """
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return np.array(data["matrix"])


def save_hand_eye(filepath, matrix_4x4, robot_serial, validation_results=None):
    """
    Saves a 4x4 hand-eye transformation matrix to a JSON file.
    
    Args:
        filepath: Where to save the file
        matrix_4x4: 4x4 numpy transformation matrix
        robot_serial: Serial number of the robot (for validation)
        validation_results: Optional tuple from validate_hand_eye_result()
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Build the data dictionary
    data = {
        "description": "Hand-eye calibration: transform from camera frame to robot end-effector frame",
        "robot_serial": robot_serial,
        "matrix": matrix_4x4.tolist(),
    }
    
    # Add validation metrics if provided
    if validation_results is not None:
        is_valid, rot_err, trans_err = validation_results
        data["validation"] = {
            "is_valid": is_valid,
            "max_rotation_error_deg": float(rot_err),
            "max_translation_error_m": float(trans_err),
        }
    
    # Write with pretty formatting
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"  -> Saved hand-eye calibration to: {filepath}")
