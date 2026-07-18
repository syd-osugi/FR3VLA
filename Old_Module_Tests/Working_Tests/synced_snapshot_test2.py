###########################
# Tests the software trigger we just wrote. 
# It pops up a window showing the D435 and D405 side-by-side. You wave your hand. 
# If your hand is in the exact same spot in both images, the timing math works.
# If this fails: The cameras are dropping frames, or the internal clock reading code is broken.
###########################

from _working_test_utils import add_working_to_path

# These older integration tests live outside the Working folder, but they need
# to import the real Working/config.py and Working modules. The helper adds the
# correctly-cased repo path, avoiding the old ../working path bug on Linux.
add_working_to_path()

import cv2
import time
import config as cfg
from hardware.camera import RealSense


def resize_to_height(image, target_height):
    """
    Resize an image for display while preserving its aspect ratio.

    This is display-only. The camera streams still run at the resolutions from
    config.py. We resize only the preview copies because cv2.hconcat requires
    every image to have the same height and type. The D435 is usually 640x480
    while the D405 is 1280x720, so concatenating raw frames directly crashes.
    """
    height, width = image.shape[:2]
    if height == target_height:
        return image

    scale = target_height / height
    target_width = int(round(width * scale))

    # INTER_AREA gives cleaner results when shrinking the larger D405 preview.
    # INTER_LINEAR is better if a smaller image ever needs to be enlarged.
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def main():
    print("--- Testing Synced Snapshot ---")

    # Start as None so the finally block can safely clean up even if one camera
    # fails before the other one is constructed.
    d435 = None
    d405 = None

    try:
        d435 = RealSense(serial_number=cfg.D435_SERIAL, resolution=cfg.D435_RESOLUTION)
        d405 = RealSense(serial_number=cfg.D405_SERIAL, resolution=cfg.D405_RESOLUTION)

        time.sleep(2)
        print("Opening window. Press 'q' to quit. Wave your hand quickly to test sync!")

        while True:
            synced = d435.grab_synced_snapshot(d405, max_delta_ms=20)
            if synced is None:
                continue

            rgb_d435, _, _, rgb_d405, _, _ = synced

            # The two cameras intentionally use different configured
            # resolutions. Normalize only the preview height so OpenCV can draw
            # them side-by-side without changing the captured data itself.
            display_height = min(rgb_d435.shape[0], rgb_d405.shape[0])
            display_d435 = resize_to_height(rgb_d435, display_height)
            display_d405 = resize_to_height(rgb_d405, display_height)
            combined_image = cv2.hconcat([display_d435, display_d405])

            cv2.putText(combined_image, "D435", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # The D405 label starts immediately after the displayed D435 image.
            # Use the resized preview width, not cfg.D435_RESOLUTION, because
            # this window may scale images for display.
            cv2.putText(combined_image, "D405", (display_d435.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Synced Check", combined_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # RealSense pipelines keep USB resources open in background threads.
        # Always stop them, including after OpenCV errors, so the next test run
        # does not inherit half-open camera state.
        cv2.destroyAllWindows()
        if d435 is not None:
            d435.stop()
        if d405 is not None:
            d405.stop()

    print("PASS: Sync test finished.")

if __name__ == "__main__":
    main()
