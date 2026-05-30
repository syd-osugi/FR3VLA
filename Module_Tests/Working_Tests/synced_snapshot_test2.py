###########################
# Tests the software trigger we just wrote. 
# It pops up a window showing the D435 and D405 side-by-side. You wave your hand. 
# If your hand is in the exact same spot in both images, the timing math works.
# If this fails: The cameras are dropping frames, or the internal clock reading code is broken.
###########################

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'working'))

import cv2
import config as cfg
from hardware.camera import RealSense

def main():
    print("--- Testing Synced Snapshot ---")
    d435 = RealSense(serial_number=cfg.D435_SERIAL, resolution=cfg.D435_RESOLUTION)
    d405 = RealSense(serial_number=cfg.D405_SERIAL, resolution=cfg.D405_RESOLUTION)
    
    import time; time.sleep(2)
    print("Opening window. Press 'q' to quit. Wave your hand quickly to test sync!")
    
    while True:
        synced = d435.grab_synced_snapshot(d405, max_delta_ms=20)
        if synced is None:
            continue
            
        rgb_d435, _, _, rgb_d405, _, _ = synced
        combined_image = cv2.hconcat([rgb_d435, rgb_d405])
        
        cv2.putText(combined_image, "D435", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined_image, "D405", (cfg.D435_RESOLUTION[0] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Synced Check", combined_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    d435.stop()
    d405.stop()
    print("PASS: Sync test finished.")

if __name__ == "__main__":
    main()