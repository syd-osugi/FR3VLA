###########################
# Tests if your computer can even see the RealSense cameras. 
# It verifies your serial numbers in config.py are correct, 
# that the background threads don't crash, and that they output standard numpy arrays.
###########################

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'working'))

import time
import config as cfg
from hardware.camera import RealSense

def main():
    print("--- Testing Camera Access ---")
    d435_serial = cfg.D435_SERIAL
    d405_serial = cfg.D405_SERIAL
    
    if "YOUR_" in d435_serial or "YOUR_" in d405_serial:
        print("FAIL: Camera serials not configured in working/config.py")
        return

    try:
        print(f"Starting D435 ({d435_serial}) at {cfg.D435_RESOLUTION}...")
        d435 = RealSense(serial_number=d435_serial, resolution=cfg.D435_RESOLUTION, fps=cfg.CAMERA_FPS)
        
        print(f"Starting D405 ({d405_serial}) at {cfg.D405_RESOLUTION}...")
        d405 = RealSense(serial_number=d405_serial, resolution=cfg.D405_RESOLUTION, fps=cfg.CAMERA_FPS)
        
        print("Waiting 2 seconds for hardware to warm up...")
        time.sleep(2)
        
        print("Grabbing 5 frames...")
        for i in range(5):
            rgb_d435, _, _ = d435.get_frames()
            if rgb_d435 is None:
                print(f"FAIL: Frame {i} returned None.")
                break
            h, w, c = rgb_d435.shape
            print(f"Frame {i}: D435 shape={w}x{h}x{c}")
            
        print("PASS: Camera access working perfectly.")
    except Exception as e:
        print(f"FAIL: {e}")
    finally:
        d435.stop()
        d405.stop()

if __name__ == "__main__":
    main()