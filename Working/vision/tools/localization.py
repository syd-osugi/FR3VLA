"""
Pixel Localization Helpers
--------------------------
Shared code for converting LLM-selected pixels into robot-frame 3D points and
saving debug images. Tool routing stays in dispatcher.py; the math and checks
live here.
"""

import os

import config as cfg
from robot.trajectory import translate_point_to_robot_frame
from utilities.coordinates import pixel_to_xyz


def _load_cv2():
    """OpenCV is optional at runtime; without it we simply skip debug images."""
    try:
        import cv2
        return cv2
    except ModuleNotFoundError:
        return None


def process_pixel(point, resolution, depth_rs, depth_scale, source_camera, robot_ee_pose=None):
    """
    Converts one [u, v] pixel into a robot-frame 3D point.

    The conversion has two stages:
    1. pixel_to_xyz() uses the RealSense depth frame to convert image pixel
       coordinates into a 3D point in that camera's optical frame.
    2. translate_point_to_robot_frame() uses the saved extrinsic calibration to
       convert the camera-frame point into the robot base frame.

    D435 uses a fixed camera -> robot base transform.
    D405 uses camera -> EE from hand-eye calibration plus the current EE -> base
    pose, so robot_ee_pose must be fresh when the wrist has moved.

    Returns a structured dict with either status="ok" and xyz fields, or
    status="invalid" and a reason the LLM can act on.
    """
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return {
            "pixel": list(point) if point else None,
            "status": "invalid",
            "reason": f"Expected [u, v] format, got: {point}",
        }

    try:
        u, v = int(point[0]), int(point[1])
    except (TypeError, ValueError):
        return {
            "pixel": list(point),
            "status": "invalid",
            "reason": f"Pixel values must be integers, got: {point}",
        }

    width, height = resolution
    if not (0 <= u < width and 0 <= v < height):
        return {
            "pixel": [u, v],
            "status": "invalid",
            "reason": f"Pixel [{u}, {v}] out of bounds (image is {width}x{height})",
        }

    xyz_data = pixel_to_xyz(u, v, depth_rs, depth_scale)
    if not xyz_data["valid"]:
        return {
            "pixel": [u, v],
            "status": "invalid",
            "reason": (
                "No valid depth at this pixel. Common causes: transparent, shiny, "
                "too-close, too-far, or depth-blind surfaces."
            ),
        }

    transform_result = translate_point_to_robot_frame(
        [xyz_data["x"], xyz_data["y"], xyz_data["z"]],
        source_camera=source_camera,
        robot_ee_pose=robot_ee_pose,
    )
    if not transform_result["valid"]:
        return {
            "pixel": [u, v],
            "status": "invalid",
            "reason": f"Transform failed: {transform_result.get('reason', 'Unknown')}",
        }

    return {
        "pixel": [u, v],
        "status": "ok",
        "xyz_camera": [xyz_data["x"], xyz_data["y"], xyz_data["z"]],
        "xyz_robot": transform_result["xyz"],
        "depth_m": xyz_data["z"],
    }


def valid_robot_points(results, source_camera):
    """Extracts successful robot-frame points in the format used by fusion."""
    return [
        {"xyz": result["xyz_robot"], "valid": True, "source": source_camera}
        for result in results
        if result.get("status") == "ok" and result.get("xyz_robot")
    ]


def save_debug_image(image, results, camera_name, suffix):
    """
    Saves a marked-up image showing which pixels succeeded or failed.

    Debug output is best-effort: failures here should never stop localization.
    """
    if image is None:
        return False

    cv2 = _load_cv2()
    if cv2 is None:
        return False

    try:
        debug = image.copy()
        for result in results:
            pixel = result.get("pixel")
            if pixel is None:
                continue

            u, v = int(pixel[0]), int(pixel[1])
            if result.get("status") == "ok":
                color = (0, 255, 0)
                label = f"Z={result.get('depth_m', 0):.2f}m"
            else:
                color = (0, 0, 255)
                label = "INVALID"

            cv2.drawMarker(debug, (u, v), color, cv2.MARKER_CROSS, 20, 2)
            cv2.circle(debug, (u, v), 5, color, -1)
            cv2.putText(
                debug,
                label,
                (u + 10, v - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        cv2.putText(
            debug,
            camera_name.upper(),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )

        os.makedirs(cfg.DEBUG_IMAGE_DIR, exist_ok=True)
        filepath = os.path.join(cfg.DEBUG_IMAGE_DIR, f"{camera_name}_{suffix}_debug.jpg")
        return bool(cv2.imwrite(filepath, debug))
    except Exception:
        return False
