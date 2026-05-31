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
4. Use the live preview to confirm the board is detected
5. Press s, ENTER, or SPACE in the preview window to capture each image
6. JSON files are saved to the calibration_data/ folder
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
    detect_charuco_corners,
    draw_charuco_detection,
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


CAPTURE_KEYS = {ord("s"), ord("S"), ord(" "), 10, 13}
QUIT_KEYS = {ord("q"), ord("Q"), 27}


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


def create_detector_params():
    """Creates OpenCV ArUco detector parameters across OpenCV versions."""
    if hasattr(cv2.aruco, "DetectorParameters"):
        return cv2.aruco.DetectorParameters()
    return cv2.aruco.DetectorParameters_create()


def draw_preview_text(image, text, position, color, scale=0.7, thickness=2):
    """Draws high-contrast overlay text on a calibration preview image."""
    x, y = position
    cv2.putText(
        image,
        text,
        (x + 1, y + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
    )
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
    )


def render_intrinsic_preview(display, camera_name, detection, captured, required_images):
    """Adds ChArUco detection drawings and operator status text to a frame."""
    draw_charuco_detection(cv2, display, detection)

    marker_count = detection.get("marker_count", 0)
    charuco_count = detection.get("charuco_count", 0)

    if detection["success"]:
        status = f"CHARUCO DETECTED: {charuco_count} corners"
        color = (0, 255, 0)
        action = "Press s / ENTER / SPACE to capture"
    elif marker_count:
        status = f"PARTIAL BOARD: {marker_count} markers, {charuco_count} corners"
        color = (0, 220, 255)
        action = "Move board until the status turns green"
    else:
        status = "Searching for ChArUco board..."
        color = (0, 0, 255)
        action = "Show the full board to the camera"

    draw_preview_text(display, f"{camera_name} intrinsic calibration", (20, 35), (255, 255, 255), 0.7)
    draw_preview_text(display, status, (20, 70), color, 0.7)
    draw_preview_text(display, action, (20, 105), (255, 255, 255), 0.6)
    draw_preview_text(
        display,
        f"Captured: {captured}/{required_images}    q/ESC: quit",
        (20, display.shape[0] - 20),
        (255, 255, 255),
        0.65,
    )
    return display


def capture_images(camera, camera_name, serial_num, required_images):
    """
    Captures calibration images from a single camera.
    
    This function:
    1. Shows a live preview
    2. Waits for preview-window capture keys
    3. Validates each captured image has enough visible corners
    4. Stores the 2D/3D point correspondences
    5. Runs the calibration math
    6. Saves the results to JSON
    
    Args:
        camera: RealSense camera object
        camera_name: "D435" or "D405" (for display and config lookup)
        serial_num: Camera serial number (for JSON validation)
        required_images: Number of images to capture (from config)

    Returns:
        bool: True when the camera was calibrated and saved, False if aborted.
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
    print(f"  - Watch the live preview for green ChArUco detection")
    print(f"  - Press s, ENTER, or SPACE in the preview window to capture")
    print(f"  - Press q or ESC in the preview window to quit")
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
    detector_params = create_detector_params()
    
    captured = 0

    window_name = f"{camera_name} Intrinsic Calibration"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    except cv2.error:
        pass

    try:
        while captured < required_images:
            rgb, _, _ = camera.get_frames()
            if rgb is None:
                display = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
                draw_preview_text(
                    display,
                    f"Waiting for {camera_name} camera frame...",
                    (20, 40),
                    (0, 0, 255),
                )
                cv2.imshow(window_name, display)
                key = cv2.waitKey(30) & 0xFF
                if key in QUIT_KEYS:
                    print(f"  {camera_name} calibration aborted by operator.")
                    return False
                continue

            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            detection = detect_charuco_corners(
                cv2,
                gray,
                aruco_dict,
                charuco_board,
                detector_params=detector_params,
            )

            display = render_intrinsic_preview(
                rgb.copy(),
                camera_name,
                detection,
                captured,
                required_images,
            )
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key in QUIT_KEYS:
                print(f"  {camera_name} calibration aborted by operator.")
                return False

            if key not in CAPTURE_KEYS:
                continue

            if not detection["success"]:
                print(
                    "    -> WAITING: "
                    f"{detection['reason']} "
                    f"({detection.get('marker_count', 0)} markers, "
                    f"{detection.get('charuco_count', 0)} corners)"
                )
                continue

            charuco_corners = detection["charuco_corners"]
            charuco_ids = detection["charuco_ids"]
            obj_points.append(get_charuco_object_points(charuco_board, charuco_ids))
            img_points.append(charuco_corners)

            captured += 1
            print(f"    -> SUCCESS: {len(charuco_ids)} corners detected")

            draw_preview_text(
                display,
                f"CAPTURE SAVED ({captured}/{required_images})",
                (20, 140),
                (0, 255, 0),
                0.75,
            )
            cv2.imshow(window_name, display)
            cv2.waitKey(250)

    finally:
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass
    
    # All images captured - now run the calibration math
    print(f"\nCalculating intrinsics for {camera_name} using {captured} images...")
    
    # Run OpenCV's calibration solver
    intrinsics_dict = calibrate_with_images(img_points, obj_points, resolution)
    
    # Save to JSON file
    save_path = get_intrinsics_save_path(camera_name)
    save_intrinsics(save_path, intrinsics_dict, serial_num)
    print(f"\n✓ SAVED: {camera_name} intrinsics to {save_path}")
    return True


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
        if not capture_images(d435, "D435", d435_serial, cfg.INTRINSIC_IMAGES_REQUIRED):
            return 1
        
        # Pause between cameras so user can reposition
        print("\n" + "="*60)
        print("D435 calibration COMPLETE!")
        print("Now prepare to calibrate the D405 (wrist camera).")
        print("="*60)
        input("Press ENTER when ready to start D405 calibration...")
        
        # Calibrate D405
        if not capture_images(d405, "D405", d405_serial, cfg.INTRINSIC_IMAGES_REQUIRED):
            return 1
        
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
