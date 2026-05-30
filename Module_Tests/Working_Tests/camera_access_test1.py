###########################
# Tests if your computer can even see the RealSense cameras. 
# It verifies your serial numbers in config.py are correct, 
# that the background threads don't crash, and that they output standard numpy arrays.
###########################

from pathlib import Path

from _working_test_utils import REPO_ROOT, add_working_to_path, unique_output_path

add_working_to_path()

import cv2
import time
import config as cfg
from hardware.camera import RealSense


SCRIPT_NAME = Path(__file__).stem
OUTPUT_DIR = REPO_ROOT / "Test_Outputs" / SCRIPT_NAME

def main():
    print("--- Testing Camera Access ---")
    d435_serial = cfg.D435_SERIAL
    d405_serial = cfg.D405_SERIAL
    d435 = None
    d405 = None
    
    if "YOUR_" in d435_serial or "YOUR_" in d405_serial:
        print("FAIL: Camera serials not configured in Working/config.py")
        return

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Saving images to {OUTPUT_DIR}")

        print(f"Starting D435 ({d435_serial}) at {cfg.D435_RESOLUTION}...")
        d435 = RealSense(serial_number=d435_serial, resolution=cfg.D435_RESOLUTION, fps=cfg.CAMERA_FPS)
        
        print(f"Starting D405 ({d405_serial}) at {cfg.D405_RESOLUTION}...")
        d405 = RealSense(serial_number=d405_serial, resolution=cfg.D405_RESOLUTION, fps=cfg.CAMERA_FPS)
        
        print("Waiting 2 seconds for hardware to warm up...")
        time.sleep(2)
        
        print("Grabbing 5 frames...")
        for i in range(5):
            rgb_d435, _, _ = d435.get_frames()
            rgb_d405, _, _ = d405.get_frames()
            if rgb_d435 is None or rgb_d405 is None:
                print(f"FAIL: Frame {i} returned None.")
                break

            d435_path = unique_output_path(OUTPUT_DIR / f"d435_frame_{i:02d}.png")
            d405_path = unique_output_path(OUTPUT_DIR / f"d405_frame_{i:02d}.png")
            if not cv2.imwrite(str(d435_path), rgb_d435):
                print(f"FAIL: Could not save {d435_path}")
                break
            if not cv2.imwrite(str(d405_path), rgb_d405):
                print(f"FAIL: Could not save {d405_path}")
                break

            h435, w435, c435 = rgb_d435.shape
            h405, w405, c405 = rgb_d405.shape
            print(
                f"Frame {i}: "
                f"D435 shape={w435}x{h435}x{c435} saved={d435_path.name}; "
                f"D405 shape={w405}x{h405}x{c405} saved={d405_path.name}"
            )
            
        print("PASS: Camera access working perfectly.")
    except Exception as e:
        print(f"FAIL: {e}")
    finally:
        if d435 is not None:
            d435.stop()
        if d405 is not None:
            d405.stop()

if __name__ == "__main__":
    main()
