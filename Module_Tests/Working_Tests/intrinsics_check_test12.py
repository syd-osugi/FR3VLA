"""
Test 12: Intrinsic Validation Check (D435 & D405)
--------------------------------------------------
Ensures that saved calibration files exactly match the plugged-in hardware.

 A test script. It reads the JSON files generated above, checks if the serial 
 numbers match the plugged-in cameras, and verifies the math hasn't drifted.

 loops through both the D435 and D405 serial numbers, checks their specific 
 JSON files, and validates both independently.
"""
from _working_test_utils import add_working_to_path

add_working_to_path()

import config as cfg
from hardware.camera import RealSense
from calibration.intrinsics import validate_saved_intrinsics

def check_camera(camera_name, serial, json_path):
    print(f"\n--- Checking {camera_name} Intrinsics ---")
    cam = RealSense(serial_number=serial)
    import time; time.sleep(2)
    
    is_valid, reason = validate_saved_intrinsics(cam.pipeline, json_path)
    
    if is_valid:
        print(f"PASS: {reason}")
    else:
        print(f"FAIL: {reason}")
        print("ACTION: Delete the JSON file and run 'scripts/run_intrinsics.py'.")
        
    cam.stop()

def main():
    print("=== Testing Intrinsics for Both Cameras ===")
    
    # Check D435
    d435 = RealSense(serial_number=cfg.D435_SERIAL)
    d435_serial = d435.profile.get_device().get_info(rs.camera_info.serial_number)
    d435.stop()
    check_camera("D435", d435_serial, cfg.INTRINSICS_D435_PATH)
    
    # Check D405
    d405 = RealSense(serial_number=cfg.D405_SERIAL)
    d405_serial = d405.profile.get_device().get_info(rs.camera_info.serial_number)
    d405.stop()
    check_camera("D405", d405_serial, cfg.INTRINSICS_D405_PATH)

if __name__ == "__main__":
    main()
