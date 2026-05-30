"""
Test 11: Hand-Eye Check
-------------------------
Verifies the D405 extrinsic calibration file is valid and mathematically sound.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'working'))

import numpy as np
import config as cfg
from calibration.hand_eye import load_hand_eye

def main():
    print("--- Testing D405 Hand-Eye Extrinsics ---")
    
    matrix = load_hand_eye(cfg.HAND_EYE_D405_PATH)
    
    if matrix is None:
        print(f"FAIL: File not found at {cfg.HAND_EYE_D405_PATH}")
        print("ACTION: Run 'scripts/run_hand_eye.py'.")
        return
        
    # Check shape
    if matrix.shape != (4, 4):
        print(f"FAIL: Matrix is wrong shape. Expected (4,4), got {matrix.shape}")
        return
        
    # Check if it's a valid rigid body transform
    # The bottom row of a 4x4 transform matrix MUST be [0, 0, 0, 1]
    bottom_row = matrix[3, :]
    if not np.allclose(bottom_row, [0, 0, 0, 1]):
        print(f"FAIL: Bottom row is not [0,0,0,1]. Got {bottom_row}")
        return
        
    # Check if rotation matrix is orthogonal (R * R_transpose = Identity)
    R = matrix[:3, :3]
    identity_check = R @ R.T
    if not np.allclose(identity_check, np.eye(3), atol=1e-5):
        print("FAIL: Rotation matrix is not orthogonal.")
        return
        
    print(f"PASS: Valid 4x4 transform matrix loaded.")
    print(f"Wrist to Camera Translation (X,Y,Z in meters): {matrix[:3, 3]}")

if __name__ == "__main__":
    main()