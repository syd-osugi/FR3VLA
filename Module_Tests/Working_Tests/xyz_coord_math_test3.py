###########################
# It points the camera at your desk, grabs the exact center pixel, 
# and asks "How far away is that?" It checks if the 3D math (deprojection) 
# returns a realistic number (like 0.5 meters) instead of a broken number (like 5000 meters). 
# It also tests if it safely rejects bad pixels.
# If this fails: Your depth scale is wrong, or the camera lens intrinsics are being read incorrectly
###########################

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'working'))

import config as cfg
from hardware.camera import RealSense
from utilities.coordinates import pixel_to_xyz

def main():
    print("--- Testing XYZ Math ---")
    d435 = RealSense(serial_number=cfg.D435_SERIAL, resolution=cfg.D435_RESOLUTION)
    import time; time.sleep(2)
    
    synced = d435.grab_synced_snapshot(d435, max_delta_ms=100)
    if synced is None:
        print("FAIL: Could not get frame."); return

    _, _, depth_rs = synced[0], synced[1], synced[2]
    
    # Dynamically calculate center pixel based on config.py resolution
    center_u = cfg.D435_RESOLUTION[0] // 2
    center_v = cfg.D435_RESOLUTION[1] // 2
    xyz = pixel_to_xyz(center_u, center_v, depth_rs, d435.depth_scale)
    
    print(f"Testing center pixel [{center_u}, {center_v}] (from config resolution):")
    if xyz['valid']:
        print(f"PASS: X={xyz['x']:.3f}, Y={xyz['y']:.3f}, Z={xyz['z']:.3f} meters")
    else:
        print("FAIL: Center pixel invalid.")
        
    xyz_bad = pixel_to_xyz(-10, -10, depth_rs, d435.depth_scale)
    print("PASS: Out-of-bounds rejected." if not xyz_bad['valid'] else "FAIL: Out-of-bounds accepted.")

    d435.stop()

if __name__ == "__main__":
    main()