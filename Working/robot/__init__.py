"""
Robot Package
-------------
Legacy robot-frame transform helpers.

The active runtime uses robot/trajectory.py for D435/D405 change-of-basis,
camera fusion, and waypoint planning. This package initializer is kept for
compatibility with older imports.

A placeholder feedback controller module is also available at
robot.feedback_controller.py. It defines a pluggable interface that can be
swapped in later when closed-loop motion control is implemented.

The future learned-policy path is scaffolded in:
  - robot.robot_interface.py: hardware boundary for Franka/no-op drivers
  - robot.safety.py: action checks before hardware execution
  - robot.learning_controller.py: observation -> policy -> safety -> robot loop
"""
import numpy as np
import json
import os
import config as cfg

def load_transform_matrix(filepath):
    """Loads a 4x4 transformation matrix from a JSON file."""
    if not os.path.exists(filepath):
        return np.eye(4) # Return Identity matrix if no file exists (fail-safe)
    with open(filepath, 'r') as f:
        data = json.load(f)
    return np.array(data["matrix"])

def translate_point_to_robot_frame(point_xyz, source_camera):
    """
    Translates a 3D point into the robot's base frame based on which camera saw it.
    
    Args:
        point_xyz: [x, y, z] raw coordinate from the depth camera.
        source_camera: "d435" or "d405".
        
    Returns:
        [x, y, z] coordinate relative to the robot's base frame.
    """
    point_arr = np.array([point_xyz[0], point_xyz[1], point_xyz[2], 1.0])
    
    if source_camera == "d435":
        # D435 is bolted to the table. We consider it the origin (0,0,0) of the world.
        # No transformation is needed.
        result = point_arr
    elif source_camera == "d405":
        # D405 is moving. We must chain the transforms:
        # Base -> Wrist -> Camera -> Point.
        # To reverse this: Point_World = (Wrist_to_Camera)^-1 * Point_D405
        T_wrist_cam = load_transform_matrix(cfg.HAND_EYE_D405_PATH)
        result = np.linalg.inv(T_wrist_cam) @ point_arr
    else:
        return point_arr
        
    return [result[0], result[1], result[2]]
