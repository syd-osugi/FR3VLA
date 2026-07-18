"""
Bird-Eye Calibration Math
-------------------------
Helper functions for the fixed overhead D435 calibration.

The user should not run this file directly. The executable script is
run_extrinsics_d435_bird_eye.py.

The D435 calibration uses a ChArUco board rigidly mounted to the robot
end-effector. At each captured robot pose, the camera sees board -> camera and
the robot reports end-effector -> robot base. This helper module combines those
measurements into one direct D435 camera -> robot base transform and saves it.

Matrix naming convention:
    T_a_to_b maps a point from frame A into frame B.

Example:
    p_base = T_cam_to_base @ p_cam

D435 calibration chain:
    T_cam_to_base = T_ee_to_base @ T_board_to_ee @ inverse(T_board_to_cam)

Only T_board_to_ee is fixed by the secure physical board mount. The other two
matrices are measured at each captured robot pose.
"""

import json
import math
import os

import numpy as np


def rotation_matrix_from_rpy_deg(roll_deg, pitch_deg, yaw_deg):
    """
    Builds R = Rz(yaw) * Ry(pitch) * Rx(roll) from degree inputs.

    The config file stores the mounted board orientation as roll/pitch/yaw in
    degrees because that is easier to measure and edit than a 3x3 matrix.
    This helper converts those human-facing values into the rotation block of
    the 4x4 T_board_to_ee matrix.
    """
    roll, pitch, yaw = [math.radians(value) for value in (roll_deg, pitch_deg, yaw_deg)]

    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def make_board_to_ee_transform(translation_m, rpy_deg):
    """
    Creates a 4x4 transform from the mounted board frame to end-effector frame.

    translation_m is the board origin expressed in end-effector coordinates.
    rpy_deg is the board frame orientation relative to the end-effector frame.

    Matrix meaning:
        p_ee = T_board_to_ee @ p_board

    This is the only matrix in the D435 calibration chain that comes from the
    physical mount geometry instead of from a camera or robot measurement.
    """
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rotation_matrix_from_rpy_deg(*rpy_deg)
    matrix[:3, 3] = np.array(translation_m, dtype=float)
    return matrix


def camera_to_robot_from_pose(T_board_to_cam, T_board_to_ee, T_ee_to_base):
    """
    Computes D435 camera -> robot base for one captured robot pose.

    OpenCV estimates board -> camera. The mounted-board measurement supplies
    board -> end-effector. Franka O_T_EE supplies end-effector -> robot base.
    Chaining those transforms gives:

        T_cam_to_base = T_ee_to_base @ T_board_to_ee @ inverse(T_board_to_cam)

    Why the inverse is needed:
        OpenCV gives board -> camera, but the final saved calibration needs
        camera -> robot base. The inverse turns board -> camera into
        camera -> board before continuing through board -> EE -> base.
    """
    T_cam_to_board = np.linalg.inv(T_board_to_cam)
    return T_ee_to_base @ T_board_to_ee @ T_cam_to_board


def average_transforms(transforms):
    """
    Averages a set of 4x4 transforms into one representative transform.

    Translation is averaged directly. Rotation is averaged then projected back
    onto the closest valid rotation matrix with SVD.

    Each captured pose produces a slightly different T_cam_to_base because of
    camera noise, sub-pixel corner noise, robot encoder precision, and tiny
    operator/capture differences. The final saved calibration is the average of
    those per-pose estimates.
    """
    if not transforms:
        raise ValueError("Need at least one transform to average")

    rotations = np.array([transform[:3, :3] for transform in transforms], dtype=float)
    translations = np.array([transform[:3, 3] for transform in transforms], dtype=float)

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


def _rotation_difference_deg(reference_rotation, measured_rotation):
    """Returns the angle between two rotation matrices in degrees."""
    delta = reference_rotation.T @ measured_rotation
    cosine = (np.trace(delta) - 1.0) / 2.0
    cosine = np.clip(cosine, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def validate_transform_set(transforms, reference_transform=None):
    """
    Reports how tightly per-pose camera -> robot estimates agree.

    A low spread means the fixed camera, mounted board transform, and robot pose
    readings are mutually consistent. A high spread usually means the board
    moved on the wrist, poses were not varied enough, or a capture was noisy.

    This is not a proof that the absolute calibration is perfect. It tells us
    whether the repeated measurements agree with each other.
    """
    if not transforms:
        raise ValueError("Need at least one transform to validate")

    reference = reference_transform
    if reference is None:
        reference = average_transforms(transforms)

    translation_errors = []
    rotation_errors = []
    for transform in transforms:
        translation_errors.append(
            float(np.linalg.norm(transform[:3, 3] - reference[:3, 3]))
        )
        rotation_errors.append(
            float(_rotation_difference_deg(reference[:3, :3], transform[:3, :3]))
        )

    return {
        "pose_count": len(transforms),
        "mean_translation_error_m": float(np.mean(translation_errors)),
        "max_translation_error_m": float(np.max(translation_errors)),
        "mean_rotation_error_deg": float(np.mean(rotation_errors)),
        "max_rotation_error_deg": float(np.max(rotation_errors)),
    }


def save_fixed_camera_transform(
    filepath,
    matrix,
    robot_serial,
    board_to_ee=None,
    mount_metadata=None,
    validation_results=None,
):
    """
    Writes the direct D435 camera -> robot base transform to JSON.

    The saved "matrix" is the runtime calibration:
        p_base = matrix @ p_d435

    The board-to-EE matrix is saved as metadata so the calibration can be
    audited later, but runtime object localization does not need the board
    mounted anymore. Runtime only needs the final D435 camera -> robot base
    matrix.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    payload = {
        "description": (
            "D435 fixed-camera calibration: transform from D435 optical frame "
            "to robot base frame"
        ),
        "from_frame": "d435_optical",
        "to_frame": "robot_base",
        "robot_serial": robot_serial,
        "matrix": matrix.tolist(),
        "board_to_ee_matrix_used_for_calibration": (
            board_to_ee.tolist() if board_to_ee is not None else None
        ),
        "board_to_ee_mount": mount_metadata,
        "validation": validation_results,
    }

    with open(filepath, "w") as file:
        json.dump(payload, file, indent=4)
