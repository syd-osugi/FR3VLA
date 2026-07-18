"""
Trajectory Translation Module
------------------------------
Transforms 3D points from camera frames to the robot's base frame.

This file also accounts for the configured gripper/tool-center-point (TCP)
offset when generating robot waypoints. The camera localization target is an
object point. The motion target should usually be the gripper/TCP point, not the
Franka EE origin.

COORDINATE FRAME HIERARCHY:
===========================
This system uses the following coordinate frames:

  1. ROBOT_BASE: The robot's fixed base frame (where the robot is bolted down)
  2. ROBOT_EE: The robot's end-effector (wrist) frame (moves with the robot)
  3. D405_CAM: The D405 camera's optical frame (moves with the wrist)
  4. D435_CAM: The D435 camera's optical frame (fixed, looking down)
  5. GRIPPER_TCP: Configured gripper/contact point attached to the EE

TRANSFORM CHAINS:
=================
For D435 (bird's eye, fixed camera):
  Point_RobotBase = T_cam_to_robot_base x Point_D435

  Where:
  - T_cam_to_robot_base comes from run_extrinsics_d435_bird_eye.py
  - Calibration uses a ChArUco board mounted to the robot end-effector
  - Runtime does not need the board mounted; the saved matrix already includes
    the board-to-end-effector mount and the robot pose samples used during
    calibration

For D405 (eye-in-hand, moving camera):
  Point_RobotBase = T_ee_to_base x T_cam_to_ee x Point_D405
  
  Where:
  - T_cam_to_ee comes from run_extrinsics_d405_hand_eye.py
  - T_ee_to_base comes from the robot's current joint state
  - Because D405 moves with the wrist, every D405 localization after robot
    motion must use a fresh robot pose

For gripper/TCP waypoints:
  EE_Target_RobotBase = Gripper_Target_RobotBase - R_ee_to_base x Offset_TCP_in_EE

  Where:
  - Gripper_Target_RobotBase is the object/goal point from camera localization
  - Offset_TCP_in_EE comes from config.py
  - R_ee_to_base comes from the current robot pose
  - This module currently returns XYZ waypoints only; orientation control belongs
    in a future motion executor

This module provides:
  - translate_point_to_robot_frame(): Single camera point transform
  - translate_points_fused(): Multi-camera point fusion
  - get_robot_trajectory_to_point(): EE waypoints that place the configured
    gripper/TCP point at the target
"""

import json
import math
import os

import numpy as np
import config as cfg


FUSION_WARNING_DISAGREEMENT_M = 0.05
DEFAULT_APPROACH_HEIGHT_M = 0.10


def rotation_matrix_from_rpy_deg(roll_deg, pitch_deg, yaw_deg):
    """Builds R = Rz(yaw) * Ry(pitch) * Rx(roll) from degree inputs."""
    roll, pitch, yaw = [math.radians(value) for value in (roll_deg, pitch_deg, yaw_deg)]

    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def make_transform_from_translation_rpy(translation_m, rpy_deg):
    """Creates a 4x4 transform from translation and roll/pitch/yaw values."""
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rotation_matrix_from_rpy_deg(*rpy_deg)
    matrix[:3, 3] = np.array(translation_m, dtype=float)
    return matrix


def get_gripper_tcp_to_ee_transform():
    """
    Builds the configured gripper/TCP -> EE transform.

    Matrix meaning:
        p_ee = T_gripper_tcp_to_ee @ p_gripper_tcp

    The translation column is the gripper/TCP origin expressed in EE
    coordinates. It is the physical offset from the Franka EE origin to the
    point on the gripper you want to place at an object.
    """
    return make_transform_from_translation_rpy(
        cfg.GRIPPER_TCP_IN_EE_TRANSLATION_M,
        cfg.GRIPPER_TCP_IN_EE_RPY_DEG,
    )


def _coerce_ee_to_base_pose(robot_ee_pose):
    """Validates a Franka EE -> robot base pose and returns it as an array."""
    try:
        T_ee_to_base = np.array(robot_ee_pose, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("robot_ee_pose must be a numeric 4x4 transform")

    if T_ee_to_base.shape != (4, 4):
        raise ValueError(f"robot_ee_pose must be 4x4, got {T_ee_to_base.shape}")

    return T_ee_to_base


def get_gripper_tcp_offset_in_base(robot_ee_pose=None):
    """
    Returns the configured gripper/TCP offset expressed in robot base axes.

    The config offset is expressed in EE coordinates. To subtract it from a
    robot-base object target, we rotate the offset by the current EE orientation:

        offset_base = R_ee_to_base @ offset_ee

    If the configured offset is zero, no robot pose is required. If it is
    nonzero, robot_ee_pose is required because the offset direction depends on
    the current EE orientation.
    """
    T_gripper_to_ee = get_gripper_tcp_to_ee_transform()
    offset_ee = T_gripper_to_ee[:3, 3]

    if np.linalg.norm(offset_ee) < 1e-12:
        return np.zeros(3, dtype=float)

    if robot_ee_pose is None:
        raise ValueError(
            "robot_ee_pose is required when GRIPPER_TCP_IN_EE_TRANSLATION_M "
            "is nonzero, because the EE orientation is needed to rotate the "
            "gripper offset into robot base coordinates."
        )

    T_ee_to_base = _coerce_ee_to_base_pose(robot_ee_pose)
    return T_ee_to_base[:3, :3] @ offset_ee


def load_transform_matrix(filepath):
    """
    Loads a 4x4 transformation matrix from a JSON file.
    
    Args:
        filepath: Path to the JSON file containing {"matrix": [[...], ...]}
        
    Returns:
        numpy.ndarray: 4x4 transformation matrix, or None if file not found
    """
    if not os.path.exists(filepath):
        print(f"Warning: Transform file not found: {filepath}")
        return None
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        matrix = np.array(data["matrix"], dtype=float)
        if matrix.shape != (4, 4):
            print(f"Warning: transform in {filepath} must be 4x4, got {matrix.shape}")
            return None
        return matrix
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        print(f"Warning: Failed to load transform from {filepath}: {e}")
        return None


def get_d435_transform():
    """
    Gets the D435 camera-to-robot-base transform.
    
    Returns:
        numpy.ndarray: 4x4 matrix transforming D435 points to robot base frame
    """
    return load_transform_matrix(cfg.BIRD_EYE_D435_PATH)


def get_d405_hand_eye_transform():
    """
    Gets the D405 camera-to-end-effector (hand-eye) transform.
    
    Returns:
        numpy.ndarray: 4x4 matrix transforming D405 points to EE frame
    """
    return load_transform_matrix(cfg.HAND_EYE_D405_PATH)


def translate_point_to_robot_frame(point_xyz, source_camera, robot_ee_pose=None):
    """
    Transforms a 3D point from camera frame to robot base frame.
    
    Args:
        point_xyz: [x, y, z] coordinates in the source camera's frame (meters)
        source_camera: "d435" or "d405" - which camera the point came from
        robot_ee_pose: 4x4 numpy array of current robot EE pose (REQUIRED for d405)
                      This is the transform from end-effector to robot base.
        
    Returns:
        dict: {
            "xyz": [x, y, z] in robot base frame,
            "valid": bool,
            "source": str,
            "reason": str (if invalid)
        }
    """
    # Convert to homogeneous coordinates [x, y, z, 1]
    try:
        point_homogeneous = np.array([point_xyz[0], point_xyz[1], point_xyz[2], 1.0], dtype=float)
    except (TypeError, IndexError, ValueError):
        return {
            "xyz": None,
            "valid": False,
            "source": source_camera,
            "reason": f"Expected a 3D point, got: {point_xyz}"
        }
    
    if source_camera == "d435":
        # Fixed-camera chain: D435 optical frame -> robot base.
        T_cam_to_robot = get_d435_transform()
        if T_cam_to_robot is None:
            return {
                "xyz": None,
                "valid": False,
                "source": "d435",
                "reason": f"Missing or invalid D435 calibration: {cfg.BIRD_EYE_D435_PATH}"
            }

        result_homogeneous = T_cam_to_robot @ point_homogeneous
        result_xyz = result_homogeneous[:3].tolist()
        
        return {
            "xyz": result_xyz,
            "valid": True,
            "source": "d435",
            "reason": None
        }
        
    elif source_camera == "d405":
        if robot_ee_pose is None:
            return {
                "xyz": None,
                "valid": False,
                "source": "d405",
                "reason": "robot_ee_pose required for d405 transforms"
            }
        
        # Get camera-to-EE transform from hand-eye calibration.
        T_cam_to_ee = get_d405_hand_eye_transform()
        if T_cam_to_ee is None:
            return {
                "xyz": None,
                "valid": False,
                "source": "d405",
                "reason": f"Missing or invalid D405 hand-eye calibration: {cfg.HAND_EYE_D405_PATH}"
            }
        
        # Transform: camera -> EE -> base
        try:
            T_ee_to_base = np.array(robot_ee_pose, dtype=float)
        except (TypeError, ValueError):
            return {
                "xyz": None,
                "valid": False,
                "source": "d405",
                "reason": "robot_ee_pose must be a numeric 4x4 transform"
            }

        if T_ee_to_base.shape != (4, 4):
            return {
                "xyz": None,
                "valid": False,
                "source": "d405",
                "reason": f"robot_ee_pose must be 4x4, got {T_ee_to_base.shape}"
            }

        point_in_ee = T_cam_to_ee @ point_homogeneous
        point_in_base = T_ee_to_base @ point_in_ee
        
        result_xyz = point_in_base[:3].tolist()
        
        return {
            "xyz": result_xyz,
            "valid": True,
            "source": "d405",
            "reason": None
        }
        
    else:
        return {
            "xyz": None,
            "valid": False,
            "source": source_camera,
            "reason": f"Unknown camera: {source_camera}"
        }


def translate_points_fused(d435_points, d405_points, robot_ee_pose=None):
    """
    Fuses 3D points from both cameras into robot base frame coordinates.
    
    When an object is seen by both cameras, we average their robot-frame
    estimates for improved accuracy. When seen by only one, we use that estimate.
    This matches the project requirement that objects are not guaranteed to be
    visible in both cameras, but both should be used when available.
    
    FUSION STRATEGY:
    ================
    1. If object seen in D435 only: Use D435 estimate
    2. If object seen in D405 only: Use D405 estimate (requires robot_ee_pose)
    3. If object seen in both: Weighted average
       - Weight by estimated uncertainty (or equal weights if unknown)
       - This reduces noise from either camera
    
    Args:
        d435_points: List of dicts from translate_point_to_robot_frame() for d435,
                     or empty list if d435 didn't see the object
        d405_points: List of dicts from translate_point_to_robot_frame() for d405,
                     or empty list if d405 didn't see the object
        robot_ee_pose: Current robot EE pose (required if d405_points used)
        
    Returns:
        dict: {
            "xyz": [x, y, z] fused position in robot base frame,
            "valid": bool,
            "sources_used": list of cameras that contributed,
            "d435_estimate": [x, y, z] or None,
            "d405_estimate": [x, y, z] or None
        }
    """
    result = {
        "xyz": None,
        "valid": False,
        "sources_used": [],
        "d435_estimate": None,
        "d405_estimate": None
    }
    
    # Check D435 estimates
    d435_valid = [p for p in d435_points if p.get("valid", False)]
    if d435_valid:
        # Average all valid D435 estimates
        d435_xyz = np.mean([p["xyz"] for p in d435_valid], axis=0).tolist()
        result["d435_estimate"] = d435_xyz
        result["sources_used"].append("d435")
    
    # Check D405 estimates
    d405_valid = [p for p in d405_points if p.get("valid", False)]
    if d405_valid:
        # Average all valid D405 estimates
        d405_xyz = np.mean([p["xyz"] for p in d405_valid], axis=0).tolist()
        result["d405_estimate"] = d405_xyz
        result["sources_used"].append("d405")
    
    # Fuse based on what's available
    if len(result["sources_used"]) == 0:
        result["valid"] = False
        result["reason"] = "No valid estimates from either camera"
        
    elif len(result["sources_used"]) == 1:
        # Only one camera saw the object
        if "d435" in result["sources_used"]:
            result["xyz"] = result["d435_estimate"]
        else:
            result["xyz"] = result["d405_estimate"]
        result["valid"] = True
        result["reason"] = f"Single camera estimate ({result['sources_used'][0]})"
        
    else:
        # Both cameras saw the object - fuse with equal weights
        # TODO: Could weight by depth uncertainty, detection confidence, etc.
        d435_arr = np.array(result["d435_estimate"])
        d405_arr = np.array(result["d405_estimate"])
        
        # Simple average (equal weights)
        fused_xyz = (d435_arr + d405_arr) / 2.0
        result["xyz"] = fused_xyz.tolist()
        result["valid"] = True
        result["reason"] = "Fused estimate from both cameras"
        
        # Calculate disagreement between cameras (for debugging/quality check)
        disagreement = np.linalg.norm(d435_arr - d405_arr) * 1000  # mm
        result["camera_disagreement_mm"] = float(disagreement)
        
        if disagreement > FUSION_WARNING_DISAGREEMENT_M * 1000:
            result["quality_warning"] = f"Cameras disagree by {disagreement:.1f}mm"
    
    return result


def get_robot_trajectory_to_point(
    target_xyz,
    approach_height=DEFAULT_APPROACH_HEIGHT_M,
    approach_direction="z",
    robot_ee_pose=None,
    return_metadata=False,
):
    """
    Generates EE-origin waypoints that place the gripper/TCP at a target point.

    target_xyz is the desired gripper/TCP position in robot base coordinates,
    usually the object point returned by camera localization. This function
    subtracts the configured gripper offset so the returned waypoints are EE
    origin positions suitable for a robot motion layer.

    With a configured gripper offset:

        p_ee_base = p_gripper_target_base - offset_gripper_in_base

    This creates a 2-point trajectory:
    1. Approach EE point: positions the gripper/TCP above/beside the target
    2. Target EE point: positions the gripper/TCP at the target
    
    Args:
        target_xyz: desired gripper/TCP [x, y, z] in robot base frame
        approach_height: Height above target for approach (meters)
        approach_direction: "z" for vertical approach, "x"/"y" for horizontal
        robot_ee_pose: Current 4x4 EE -> base pose. Required when the configured
                       gripper/TCP offset is nonzero, because the offset is
                       stored in EE coordinates and must be rotated into base.
        return_metadata: If True, return (waypoints, metadata).
        
    Returns:
        list: EE-origin [x, y, z] waypoints in robot base frame, or
              (waypoints, metadata) when return_metadata=True.
    """
    target = np.array(target_xyz, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError(f"target_xyz must be a finite [x, y, z] point, got: {target_xyz}")

    try:
        approach_height = float(approach_height)
    except (TypeError, ValueError):
        raise ValueError(f"approach_height must be numeric, got: {approach_height}")

    if approach_height <= 0:
        raise ValueError(f"approach_height must be positive, got: {approach_height}")
    
    if approach_direction == "z":
        tcp_approach = target.copy()
        tcp_approach[2] += approach_height
    elif approach_direction == "x":
        tcp_approach = target.copy()
        tcp_approach[0] -= approach_height
    elif approach_direction == "y":
        tcp_approach = target.copy()
        tcp_approach[1] -= approach_height
    else:
        raise ValueError("approach_direction must be one of: 'x', 'y', 'z'")

    gripper_offset_base = get_gripper_tcp_offset_in_base(robot_ee_pose)
    ee_approach = tcp_approach - gripper_offset_base
    ee_target = target - gripper_offset_base
    waypoints = [ee_approach.tolist(), ee_target.tolist()]

    if not return_metadata:
        return waypoints

    T_gripper_to_ee = get_gripper_tcp_to_ee_transform()
    metadata = {
        "target_xyz_is": "desired_gripper_tcp_position_in_robot_base",
        "waypoints_are": "end_effector_origin_positions_in_robot_base",
        "gripper_tcp_waypoints_robot_base": [
            tcp_approach.tolist(),
            target.tolist(),
        ],
        "gripper_tcp_offset_in_ee_m": T_gripper_to_ee[:3, 3].tolist(),
        "gripper_tcp_rpy_in_ee_deg": list(cfg.GRIPPER_TCP_IN_EE_RPY_DEG),
        "gripper_tcp_offset_in_base_m": gripper_offset_base.tolist(),
        "assumption": (
            "The EE orientation used to rotate the configured gripper offset "
            "stays fixed for these XYZ-only waypoints. Re-plan if the robot "
            "changes wrist orientation."
        ),
    }
    return waypoints, metadata
