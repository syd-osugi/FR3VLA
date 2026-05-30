"""
Script: 20-Image Intrinsic Calibration
---------------------------------------
Run this manually to generate highly accurate intrinsic calibration files.
Uses the large ChArUco board.

WHAT IS INTRINSIC CALIBRATION?
==============================
Every camera lens has imperfections that cause distortion (barrel/pincushion).
Intrinsic calibration finds:
1. The focal length (how much the lens zooms)
2. The principal point (where the optical center is)
3. The distortion coefficients (how much the lens bends the image)

WHY 20 IMAGES?
==============
A single image can't separate focal length from distortion. By capturing the
board from many different angles and distances, the math solver can uniquely
determine all parameters.

HOW TO USE:
===========
1. Print a large ChArUco board with the dimensions specified in config.py
2. Run this script
3. Wave the board in front of each camera, covering edges/corners
4. Press ENTER 20 times for each camera
5. JSON files are saved to the calibration_data/ folder
"""

import sys
import os

# Add parent directory to path so we can import from the working folder
WORKING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKING_DIR not in sys.path:
    sys.path.insert(0, WORKING_DIR)

import numpy as np
import config as cfg
from hardware.camera import RealSense
from camera_calibration.charuco_utils import (
    create_charuco_board,
    get_aruco_dictionary,
    get_charuco_object_points,
    require_charuco_support,
)
# Import the math functions from intrinsics_math.py
from camera_calibration.intrinsics_math import calibrate_with_images, save_intrinsics

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

try:
    import pyrealsense2 as rs
except ModuleNotFoundError:
    rs = None


def require_runtime_dependencies():
    require_charuco_support(cv2)
    if rs is None:
        raise RuntimeError("pyrealsense2 is required for RealSense calibration.")


def get_camera_serial(camera):
    """
    Safely extracts the serial number from a RealSense camera.
    
    Args:
        camera: RealSense object with an active pipeline
        
    Returns:
        str: The camera's serial number
    """
    return camera.profile.get_device().get_info(rs.camera_info.serial_number)


def get_camera_resolution(camera_name):
    """
    Returns the correct resolution for a camera based on its name.
    
    This ensures D435 and D405 use their configured resolutions, not a hardcoded value.
    
    Args:
        camera_name: "D435" or "D405"
        
    Returns:
        tuple: (width, height) from config
    """
    if "D435" in camera_name:
        return cfg.D435_RESOLUTION
    else:
        return cfg.D405_RESOLUTION


def get_intrinsics_save_path(camera_name):
    """
    Returns the correct save path for a camera's intrinsics file.
    
    Args:
        camera_name: "D435" or "D405"
        
    Returns:
        str: File path from config
    """
    if "D435" in camera_name:
        return cfg.INTRINSICS_D435_PATH
    else:
        return cfg.INTRINSICS_D405_PATH


def detect_charuco_corners(gray_image, aruco_dict, charuco_board, detector_params):
    """
    Detects ChArUco corners in a grayscale image.
    
    The detection process:
    1. Find ArUco markers in the image
    2. Refine marker corners using the board geometry
    3. Interpolate chessboard corners between the markers
    
    Args:
        gray_image: Grayscale numpy array
        aruco_dict: OpenCV ArUco dictionary
        charuco_board: CharucoBoard object
        detector_params: DetectorParameters object
        
    Returns:
        tuple: (charuco_corners, charuco_ids) or (None, None) if failed
    """
    # Step 1: Find ArUco markers
    corners, ids, rejected = cv2.aruco.detectMarkers(
        gray_image, aruco_dict, parameters=detector_params
    )
    
    # Need at least 4 markers to reliably interpolate corners
    if ids is None or len(ids) < 4:
        return None, None
    
    # Step 2: Refine marker corners using board constraints.
    # OpenCV versions differ in how many values refineDetectedMarkers returns,
    # so keep only the first three values when refinement is available.
    try:
        refined = cv2.aruco.refineDetectedMarkers(
            gray_image, charuco_board, corners, ids, rejected
        )
        corners, ids, rejected = refined[:3]
    except Exception:
        pass
    
    # Step 3: Interpolate chessboard corners from marker positions
    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray_image, charuco_board
    )
    
    if retval:
        return charuco_corners, charuco_ids
    else:
        return None, None


def capture_images(camera, camera_name, serial_num, required_images):
    """
    Captures calibration images from a single camera.
    
    This function:
    1. Shows a live preview
    2. Waits for ENTER key presses
    3. Validates each captured image has enough visible corners
    4. Stores the 2D/3D point correspondences
    5. Runs the calibration math
    6. Saves the results to JSON
    
    Args:
        camera: RealSense camera object
        camera_name: "D435" or "D405" (for display and config lookup)
        serial_num: Camera serial number (for JSON validation)
        required_images: Number of images to capture (from config)
    """
    print(f"\n{'='*60}")
    print(f"  Calibrating {camera_name} (Serial: {serial_num})")
    print(f"{'='*60}")
    print(f"Board size: {cfg.INTRINSIC_BOARD_CORNERS[0]}x{cfg.INTRINSIC_BOARD_CORNERS[1]} corners")
    print(f"Square size: {cfg.INTRINSIC_SQUARE_SIZE*100:.1f}cm, Marker size: {cfg.INTRINSIC_MARKER_SIZE*100:.1f}cm")
    print(f"\nINSTRUCTIONS:")
    print(f"  - Hold the board at various angles and distances")
    print(f"  - Cover all corners and edges of the image")
    print(f"  - Avoid blurry images (hold still)")
    print(f"  - Press ENTER to capture each image")
    print(f"{'='*60}\n")
    
    # Storage for point correspondences
    obj_points = []  # 3D world coordinates of corners
    img_points = []  # 2D pixel coordinates of corners
    
    # Get resolution from config (not hardcoded)
    resolution = get_camera_resolution(camera_name)
    
    # Create ChArUco board and detector from config values
    aruco_dict = get_aruco_dictionary(cv2, cfg.INTRINSIC_ARUCO_DICT_NAME)
    charuco_board = create_charuco_board(
        cv2,
        cfg.INTRINSIC_BOARD_CORNERS,
        cfg.INTRINSIC_SQUARE_SIZE,
        cfg.INTRINSIC_MARKER_SIZE,
        aruco_dict,
    )
    detector_params = cv2.aruco.DetectorParameters()
    
    captured = 0
    
    while captured < required_images:
        input(f"  [{captured+1}/{required_images}] Press ENTER to capture {camera_name} image...")
        
        # Get frame from camera
        rgb, _, _ = camera.get_frames()
        if rgb is None:
            print("    -> ERROR: Could not get frame from camera. Retrying...")
            continue
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        
        # Detect ChArUco corners
        charuco_corners, charuco_ids = detect_charuco_corners(
            gray, aruco_dict, charuco_board, detector_params
        )
        
        if charuco_corners is not None:
            # Success: Store the 3D board corners and their 2D pixel positions
            obj_points.append(get_charuco_object_points(charuco_board, charuco_ids))
            img_points.append(charuco_corners)
            
            # Visual feedback: draw detected corners on image
            display = rgb.copy()
            cv2.aruco.drawDetectedMarkers(display, 
                cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)[0],
                cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)[1])
            cv2.aruco.drawDetectedCornersCharuco(
                display,
                charuco_corners,
                charuco_ids,
                (0, 255, 0),
            )
            
            # Show the successful capture briefly
            cv2.imshow(f"{camera_name} - Image {captured+1} captured!", display)
            cv2.waitKey(500)
            cv2.destroyAllWindows()
            
            captured += 1
            print(f"    -> SUCCESS: {len(charuco_ids)} corners detected")
        else:
            print("    -> FAILED: Could not detect enough corners. Try a different angle/distance.")
    
    # All images captured - now run the calibration math
    print(f"\nCalculating intrinsics for {camera_name} using {captured} images...")
    
    # Run OpenCV's calibration solver
    intrinsics_dict = calibrate_with_images(img_points, obj_points, resolution)
    
    # Save to JSON file
    save_path = get_intrinsics_save_path(camera_name)
    save_intrinsics(save_path, intrinsics_dict, serial_num)
    print(f"\n✓ SAVED: {camera_name} intrinsics to {save_path}")


def main():
    """
    Main entry point for intrinsic calibration.
    
    Calibrates both D435 and D405 cameras sequentially.
    """
    try:
        require_runtime_dependencies()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    print("=" * 60)
    print("  20-IMAGE INTRINSIC CALIBRATION TOOL")
    print("=" * 60)
    print("\nThis will calibrate BOTH cameras (D435 then D405).")
    print("Make sure both cameras are plugged in and the ChArUco board is ready.\n")
    
    # Initialize cameras with their configured serial numbers and resolutions
    print("Initializing cameras...")
    try:
        d435 = RealSense(
            serial_number=cfg.D435_SERIAL, 
            resolution=cfg.D435_RESOLUTION, 
            fps=cfg.CAMERA_FPS
        )
        d405 = RealSense(
            serial_number=cfg.D405_SERIAL, 
            resolution=cfg.D405_RESOLUTION, 
            fps=cfg.CAMERA_FPS
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize cameras: {e}")
        return 1
    
    # Allow cameras to warm up (auto-exposure needs a few frames to stabilize)
    import time
    time.sleep(cfg.CAMERA_WARMUP_SECONDS)
    
    # Get actual serial numbers from hardware
    d435_serial = get_camera_serial(d435)
    d405_serial = get_camera_serial(d405)
    
    try:
        # Calibrate D435 first
        capture_images(d435, "D435", d435_serial, cfg.INTRINSIC_IMAGES_REQUIRED)
        
        # Pause between cameras so user can reposition
        input("\n" + "="*60)
        print("D435 calibration COMPLETE!")
        print("Now prepare to calibrate the D405 (wrist camera).")
        print("="*60)
        input("Press ENTER when ready to start D405 calibration...")
        
        # Calibrate D405
        capture_images(d405, "D405", d405_serial, cfg.INTRINSIC_IMAGES_REQUIRED)
        
    finally:
        # Always clean up camera resources, even if calibration fails
        print("\nShutting down cameras...")
        d435.stop()
        d405.stop()
    
    print("\n" + "=" * 60)
    print("  CALIBRATION COMPLETE!")
    print("=" * 60)
    print(f"\nIntrinsic files saved to: {cfg.CALIBRATION_DIR}/")
    print(f"  - {cfg.INTRINSICS_D435_PATH}")
    print(f"  - {cfg.INTRINSICS_D405_PATH}")
    print("\nYou can now run intrinsics_check.py to verify results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
