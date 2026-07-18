"""
Shared helpers for hardware-facing extrinsic calibration check scripts.
"""

from __future__ import annotations

import json
import os

import numpy as np


def load_matrix_from_json(filepath):
    """Load a saved {"matrix": ...} calibration JSON file."""
    if not os.path.exists(filepath):
        return None, f"File not found at {filepath}"

    try:
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
        matrix = np.array(data["matrix"], dtype=float)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return None, f"Could not load transform matrix from {filepath}: {exc}"

    return matrix, None


def validate_rigid_transform(matrix):
    """Return (ok, reason) for a 4x4 rigid-body transform matrix."""
    if matrix is None:
        return False, "Matrix is missing."

    if matrix.shape != (4, 4):
        return False, f"Matrix is wrong shape. Expected (4, 4), got {matrix.shape}."

    if not np.all(np.isfinite(matrix)):
        return False, "Matrix contains non-finite values."

    bottom_row = matrix[3, :]
    if not np.allclose(bottom_row, [0, 0, 0, 1]):
        return False, f"Bottom row is not [0, 0, 0, 1]. Got {bottom_row}."

    rotation = matrix[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5):
        return False, "Rotation matrix is not orthogonal."

    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=1e-5):
        return False, f"Rotation determinant should be 1.0, got {determinant:.6f}."

    return True, "Valid 4x4 rigid transform."
