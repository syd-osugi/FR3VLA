###########################
# Test 5: Synced depth image capture.
#
# This captures a synchronized D435/D405 snapshot, colorizes the depth images,
# shows them side-by-side, and saves a non-overwriting PNG in Test_Outputs.
# It complements tests 3 and 4: those mark one XYZ pixel per camera, while this
# lets you visually compare the full synchronized depth frames.
###########################

from pathlib import Path
import time

from _working_test_utils import REPO_ROOT, add_working_to_path, unique_output_path

add_working_to_path()

import cv2
import numpy as np

import config as cfg
from hardware.camera import RealSense


SCRIPT_NAME = Path(__file__).stem
OUTPUT_DIR = REPO_ROOT / "Test_Outputs" / SCRIPT_NAME
OUTPUT_IMAGE_PATH = OUTPUT_DIR / "synced_depth_side_by_side.png"


def resize_to_height(image, target_height):
    """
    Resize an image for side-by-side display while preserving aspect ratio.

    The D435 and D405 run at different configured resolutions. OpenCV hconcat
    requires equal image heights, so we resize only the display/debug copies.
    """
    height, width = image.shape[:2]
    if height == target_height:
        return image

    scale = target_height / height
    target_width = int(round(width * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def colorize_depth(depth_image):
    """
    Convert raw uint16 depth into a visible color image.

    Raw depth images are mostly dark if saved directly because their values are
    millimeters-ish integers, not normal 8-bit color pixels. We normalize only
    the valid nonzero depth range for this frame and apply a colormap so near
    and far structure is easy to inspect.
    """
    valid_depth = depth_image[depth_image > 0]
    if valid_depth.size == 0:
        normalized = np.zeros(depth_image.shape, dtype=np.uint8)
    else:
        min_depth = int(valid_depth.min())
        max_depth = int(valid_depth.max())
        if min_depth == max_depth:
            normalized = np.zeros(depth_image.shape, dtype=np.uint8)
        else:
            clipped = np.clip(depth_image, min_depth, max_depth)
            normalized = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX)
            normalized = normalized.astype(np.uint8)
            normalized[depth_image == 0] = 0

    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def add_label(image, camera_name, raw_depth_shape):
    """Draw camera name and raw depth dimensions on the display image."""
    labeled = image.copy()
    text_lines = [
        camera_name,
        f"depth {raw_depth_shape[1]}x{raw_depth_shape[0]}",
    ]

    cv2.rectangle(labeled, (0, 0), (230, 66), (0, 0, 0), -1)
    for index, text in enumerate(text_lines):
        cv2.putText(
            labeled,
            text,
            (12, 26 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
    return labeled


def save_side_by_side_depth_image(depth_d435, depth_d405):
    """Colorize, label, concatenate, and save the synced depth debug image."""
    colorized_d435 = colorize_depth(depth_d435)
    colorized_d405 = colorize_depth(depth_d405)

    display_height = min(colorized_d435.shape[0], colorized_d405.shape[0])
    display_d435 = resize_to_height(colorized_d435, display_height)
    display_d405 = resize_to_height(colorized_d405, display_height)

    display_d435 = add_label(display_d435, "D435 synced depth", depth_d435.shape)
    display_d405 = add_label(display_d405, "D405 synced depth", depth_d405.shape)
    combined_image = cv2.hconcat([display_d435, display_d405])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved_path = unique_output_path(OUTPUT_IMAGE_PATH)
    if not cv2.imwrite(str(saved_path), combined_image):
        return None, combined_image
    return saved_path, combined_image


def main():
    print("--- Testing Synced Depth Images ---")
    d435 = None
    d405 = None

    try:
        d435 = RealSense(serial_number=cfg.D435_SERIAL, resolution=cfg.D435_RESOLUTION, fps=cfg.CAMERA_FPS)
        d405 = RealSense(serial_number=cfg.D405_SERIAL, resolution=cfg.D405_RESOLUTION, fps=cfg.CAMERA_FPS)

        time.sleep(2)
        synced = d435.grab_synced_snapshot(d405, max_delta_ms=20)
        if synced is None:
            print("FAIL: Could not capture synchronized D435/D405 depth frames.")
            return

        _, depth_d435, _, _, depth_d405, _ = synced
        if depth_d435 is None or depth_d405 is None:
            print("FAIL: Synced snapshot did not include both depth images.")
            return

        saved_path, combined_image = save_side_by_side_depth_image(depth_d435, depth_d405)
        if saved_path is None:
            print(f"FAIL: Could not save synced depth image to {OUTPUT_IMAGE_PATH}")
            return

        print(f"D435 depth shape: {depth_d435.shape[1]}x{depth_d435.shape[0]}")
        print(f"D405 depth shape: {depth_d405.shape[1]}x{depth_d405.shape[0]}")
        print(f"Saved synced depth image: {saved_path}")

        cv2.imshow("Synced Depth Images", combined_image)
        print("Showing synced depth image. Press any key in the image window to close.")
        cv2.waitKey(0)
        print("PASS: Synced depth image capture finished.")
    finally:
        cv2.destroyAllWindows()
        if d435 is not None:
            d435.stop()
        if d405 is not None:
            d405.stop()


if __name__ == "__main__":
    main()
