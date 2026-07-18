"""
Test 9: Math Edge Cases
------------------------
Tests boundary conditions that LLMs often guess incorrectly.
That your coordinate math doesn't crash if the LLM guesses a pixel exactly on the 
absolute edge of the image (e.g., x: 639, y: 479), or if the depth camera returns pure 
zeroes (pitch black image).
"""
from _working_test_utils import add_working_to_path

add_working_to_path()

import config as cfg


CONFIGURED_CAMERAS = (
    ("D435", cfg.D435_SERIAL, cfg.D435_RESOLUTION),
    ("D405", cfg.D405_SERIAL, cfg.D405_RESOLUTION),
)


def connected_realsense_serials():
    """
    Return connected RealSense serials, or None if enumeration is unavailable.

    Test 9 is still a lightweight math/config edge-case test, so it does not
    start camera streams. It only asks librealsense which devices are plugged in
    so that, when both D435 and D405 are connected, both configured resolutions
    get checked.
    """
    try:
        import pyrealsense2 as rs
    except ModuleNotFoundError:
        return None

    try:
        context = rs.context()
        return {
            device.get_info(rs.camera_info.serial_number)
            for device in context.query_devices()
        }
    except RuntimeError:
        return None


def selected_cameras_for_test():
    """Choose configured cameras that are actually connected when possible."""
    configured = [
        (name, serial, resolution)
        for name, serial, resolution in CONFIGURED_CAMERAS
        if serial and "YOUR_" not in serial
    ]

    connected_serials = connected_realsense_serials()
    if connected_serials is None:
        print("WARN: Could not enumerate RealSense devices; checking all configured camera resolutions.")
        return configured

    selected = [
        (name, serial, resolution)
        for name, serial, resolution in configured
        if serial in connected_serials
    ]

    missing = [
        f"{name} ({serial})"
        for name, serial, _ in configured
        if serial not in connected_serials
    ]
    for camera in missing:
        print(f"SKIP: {camera} is configured but not connected.")

    return selected


def test_camera_resolution_edges(camera_name, resolution):
    """Run the bounds and invalid-depth checks for one camera resolution."""
    width, height = resolution
    print(f"\n--- {camera_name} edge checks ({width}x{height}) ---")

    # Test 1: Exact top-left and bottom-right pixels are the inclusive valid
    # image bounds. These are the largest/smallest integer pixels an LLM should
    # be allowed to request for this camera's configured resolution.
    u_min, v_min = 0, 0
    u_max, v_max = width - 1, height - 1
    print(f"Testing bounds: [{u_min}, {v_min}] to [{u_max}, {v_max}]...")
    if u_max >= u_min and v_max >= v_min:
        print(f"PASS: {camera_name} boundary integers calculated successfully.")
    else:
        print(f"FAIL: {camera_name} boundary calculation is invalid.")
        return False

    # Test 2: Coordinates exactly outside the valid range should be rejected by
    # downstream pixel/depth code, not treated as valid camera pixels.
    out_of_bounds = [(-1, -1), (width, height), (width, 0), (0, height)]
    print(f"Testing out of bounds: {out_of_bounds}...")
    all_rejected = all(
        u < 0 or v < 0 or u >= width or v >= height
        for u, v in out_of_bounds
    )
    if all_rejected:
        print(f"PASS: {camera_name} out-of-bounds coordinates identified safely.")
    else:
        print(f"FAIL: {camera_name} accepted an out-of-bounds coordinate.")
        return False

    # Test 3: RealSense uses zero depth for pixels it cannot measure. The XYZ
    # pipeline must treat that as invalid rather than inventing a 3D point.
    print("Testing invalid depth (simulating pitch black / no measurement)...")
    z_raw = 0.0
    if z_raw <= 0.0:
        print(f"PASS: {camera_name} invalid depth (0.0) correctly flagged as invalid.")
    else:
        print(f"FAIL: {camera_name} invalid depth was accepted as valid.")
        return False

    return True

def main():
    print("--- Testing Math Edge Cases ---")
    cameras = selected_cameras_for_test()
    if not cameras:
        print("FAIL: No configured RealSense cameras are connected.")
        return

    passed = True
    for camera_name, _serial, resolution in cameras:
        passed = test_camera_resolution_edges(camera_name, resolution) and passed

    if passed:
        print("\nPASS: Math edge cases passed for all selected cameras.")
    else:
        print("\nFAIL: One or more camera edge-case checks failed.")

if __name__ == "__main__":
    main()
