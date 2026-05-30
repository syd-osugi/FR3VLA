"""
Shared XYZ/depth validation logic for the D435 and D405 test scripts.

The D435 and D405 should be tested the same way: grab a color/depth frame,
choose the center pixel, convert that pixel to XYZ meters, and save a marked
debug image. Keeping the shared logic here prevents the two camera-specific
scripts from drifting apart.
"""

from __future__ import annotations

from pathlib import Path
import time

from _working_test_utils import REPO_ROOT, add_working_to_path, unique_output_path

# These test scripts run from Module_Tests/Working_Tests, so add the real
# Working folder before importing hardware/camera and utilities modules.
add_working_to_path()

import cv2
import numpy as np

from hardware.camera import RealSense
from utilities.coordinates import pixel_to_xyz


def nearest_valid_depth_pixel(depth_image, center_u, center_v, search_radius=80):
    """
    Find the nearest nonzero depth pixel around the requested center pixel.

    RealSense depth frames often contain isolated zero values. A zero means the
    camera could not measure that exact pixel, not that the coordinate math is
    broken. This helper lets the test verify the deprojection math using the
    closest measurable pixel near the center instead of failing on one unlucky
    zero-depth sample.
    """
    height, width = depth_image.shape[:2]
    u_min = max(0, center_u - search_radius)
    u_max = min(width - 1, center_u + search_radius)
    v_min = max(0, center_v - search_radius)
    v_max = min(height - 1, center_v + search_radius)

    patch = depth_image[v_min : v_max + 1, u_min : u_max + 1]
    valid_v, valid_u = np.nonzero(patch)
    if valid_u.size == 0:
        return None

    absolute_u = valid_u + u_min
    absolute_v = valid_v + v_min
    distances = (absolute_u - center_u) ** 2 + (absolute_v - center_v) ** 2
    nearest_index = int(np.argmin(distances))

    u = int(absolute_u[nearest_index])
    v = int(absolute_v[nearest_index])
    return u, v, int(depth_image[v, u])


def save_marked_debug_image(
    camera_name,
    color_image,
    center_pixel,
    calculated_pixel,
    raw_depth,
    xyz,
    output_path,
):
    """
    Save the RGB frame with an obvious marker on the pixel used for XYZ math.

    The red X marks the exact pixel passed to pixel_to_xyz(). If the exact
    center pixel had invalid depth and the test had to use the nearest valid
    nearby pixel, a small yellow circle marks the original center. That makes
    it clear whether the calculation happened exactly at center or at a safe
    nearby fallback.
    """
    debug_image = color_image.copy()
    center_u, center_v = center_pixel
    calc_u, calc_v = calculated_pixel

    if calculated_pixel != center_pixel:
        cv2.circle(debug_image, (center_u, center_v), 8, (0, 255, 255), 2)

    cv2.drawMarker(
        debug_image,
        (calc_u, calc_v),
        (0, 0, 255),
        markerType=cv2.MARKER_TILTED_CROSS,
        markerSize=36,
        thickness=3,
    )
    cv2.putText(
        debug_image,
        f"{camera_name} XYZ pixel [{calc_u}, {calc_v}]",
        (max(0, calc_u - 120), max(25, calc_v - 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )

    # Put the pass/fail-relevant numeric result directly on the image. This
    # makes the saved PNG useful on its own, without needing to cross-reference
    # the terminal output from the test run.
    result_lines = [
        f"{camera_name} XYZ PASS",
        f"pixel: [{calc_u}, {calc_v}]",
        f"depth raw: {raw_depth}",
        f"X: {xyz['x']:.3f} m",
        f"Y: {xyz['y']:.3f} m",
        f"Z: {xyz['z']:.3f} m",
    ]
    x0 = 12
    y0 = 28
    line_height = 24
    box_width = 250
    box_height = line_height * len(result_lines) + 16
    overlay = debug_image.copy()
    cv2.rectangle(overlay, (0, 0), (box_width, box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, debug_image, 0.45, 0, debug_image)

    for index, line in enumerate(result_lines):
        cv2.putText(
            debug_image,
            line,
            (x0, y0 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path = unique_output_path(output_path)
    if not cv2.imwrite(str(saved_path), debug_image):
        return None
    return saved_path


def run_xyz_coord_math_test(camera_name, serial_number, resolution, fps, script_name):
    """Run the XYZ/depth math test for one configured RealSense camera."""
    print(f"--- Testing {camera_name} XYZ Math ---")
    camera = None

    try:
        camera = RealSense(serial_number=serial_number, resolution=resolution, fps=fps)

        # Give auto-exposure and the background capture thread a short warmup.
        # Without this, the first depth frame can be partially empty.
        time.sleep(2)

        # Dynamically calculate center pixel based on config.py resolution.
        center_u = resolution[0] // 2
        center_v = resolution[1] // 2
        print(f"Testing {camera_name} center pixel [{center_u}, {center_v}] from config resolution {resolution}:")

        xyz = None
        test_u = center_u
        test_v = center_v
        raw_depth = None
        center_was_invalid = False
        color_image = None
        depth_rs = None

        # Try a handful of fresh frames. Depth at a single pixel can flicker
        # between valid and zero, especially on low-texture, shiny, transparent,
        # too-close, or too-far surfaces.
        for _attempt in range(1, 16):
            color_image, depth_image, depth_rs = camera.get_frames()
            if color_image is None or depth_image is None or depth_rs is None:
                time.sleep(0.1)
                continue

            raw_depth = int(depth_image[center_v, center_u])
            xyz = pixel_to_xyz(center_u, center_v, depth_rs, camera.depth_scale)
            if xyz["valid"]:
                break

            center_was_invalid = True
            nearby = nearest_valid_depth_pixel(depth_image, center_u, center_v)
            if nearby is not None:
                test_u, test_v, raw_depth = nearby
                xyz = pixel_to_xyz(test_u, test_v, depth_rs, camera.depth_scale)
                if xyz["valid"]:
                    break

            time.sleep(0.1)

        if xyz is None or not xyz["valid"]:
            print(f"FAIL: No valid depth found near the center of the {camera_name} frame.")
            print(
                f"ACTION: Aim the {camera_name} at a matte, non-transparent surface "
                "inside that camera's usable depth range."
            )
            return

        if center_was_invalid and (test_u, test_v) != (center_u, center_v):
            print(
                "NOTE: Exact center depth was invalid, so the test used the "
                f"nearest valid nearby {camera_name} pixel [{test_u}, {test_v}] instead."
            )

        print(f"Raw depth at tested {camera_name} pixel: {raw_depth}")
        print(f"PASS: {camera_name} X={xyz['x']:.3f}, Y={xyz['y']:.3f}, Z={xyz['z']:.3f} meters")

        output_dir = REPO_ROOT / "Test_Outputs" / script_name
        output_path = output_dir / f"{camera_name.lower()}_xyz_center_pixel.png"
        saved_image_path = save_marked_debug_image(
            camera_name,
            color_image,
            (center_u, center_v),
            (test_u, test_v),
            raw_depth,
            xyz,
            output_path,
        )
        if saved_image_path is not None:
            print(f"Saved marked {camera_name} image: {saved_image_path}")
        else:
            print(f"FAIL: Could not save marked {camera_name} image: {output_path}")

        xyz_bad = pixel_to_xyz(-10, -10, depth_rs, camera.depth_scale)
        print("PASS: Out-of-bounds rejected." if not xyz_bad["valid"] else "FAIL: Out-of-bounds accepted.")
    finally:
        if camera is not None:
            camera.stop()
