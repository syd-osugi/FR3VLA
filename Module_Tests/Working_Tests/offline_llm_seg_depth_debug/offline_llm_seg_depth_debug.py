"""
Offline LLM / Segmentation / Depth Debug Harness
------------------------------------------------
Runs the localization path without robot hardware, RealSense cameras, or an LLM
server. By default it creates a synthetic scene, a mock LLM tool call, segmented
target pixels, fake depth frames, annotated debug images, and a JSON report.

Example:
    python3 Module_Tests/Working_Tests/offline_llm_seg_depth_debug/offline_llm_seg_depth_debug.py

Replay a saved LLM response:
    python3 Module_Tests/Working_Tests/offline_llm_seg_depth_debug/offline_llm_seg_depth_debug.py \
        --llm-response-file /path/to/response.json
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import re
import struct
import sys
import zlib

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
WORKING_TESTS_DIR = SCRIPT_DIR.parent
if str(WORKING_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(WORKING_TESTS_DIR))

from _working_test_utils import TEST_OUTPUTS_DIR, patched_attr  # noqa: E402

import config as cfg  # noqa: E402
from robot.trajectory import translate_points_fused  # noqa: E402
from vision.tools import dispatcher, localization  # noqa: E402


LOCALIZATION_TOOLS = {"get_xyz_d435", "get_xyz_d405", "get_xyz_fused"}
DEFAULT_OUTPUT_DIR = TEST_OUTPUTS_DIR / "offline_llm_seg_depth_debug"
CAMERA_OFFSETS_M = {
    "d435": np.array([0.35, -0.20, 0.0], dtype=float),
    "d405": np.array([0.35, -0.20, 0.0], dtype=float),
}


def load_cv2():
    try:
        import cv2

        return cv2
    except ModuleNotFoundError:
        return None


@dataclass
class FakeIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    coeffs: tuple = (0.0, 0.0, 0.0, 0.0, 0.0)


class FakeProfile:
    def __init__(self, intrinsics):
        self.intrinsics = intrinsics

    def as_video_stream_profile(self):
        return self


class FakeDepthFrame:
    """Small RealSense-like depth frame backed by a numpy depth map in meters."""

    def __init__(self, depth_m, fx=600.0, fy=600.0):
        self.depth_m = np.asarray(depth_m, dtype=float)
        height, width = self.depth_m.shape
        self.profile = FakeProfile(
            FakeIntrinsics(
                width=width,
                height=height,
                fx=float(fx),
                fy=float(fy),
                ppx=float(width) / 2.0,
                ppy=float(height) / 2.0,
            )
        )

    def get_distance(self, u, v):
        height, width = self.depth_m.shape
        if not (0 <= u < width and 0 <= v < height):
            raise RuntimeError("pixel outside fake frame")
        return float(self.depth_m[int(v), int(u)])


class FakeRealSenseModule:
    @staticmethod
    def rs2_deproject_pixel_to_point(intrinsics, pixel, depth_m):
        u, v = pixel
        return [
            (float(u) - intrinsics.ppx) / intrinsics.fx * depth_m,
            (float(v) - intrinsics.ppy) / intrinsics.fy * depth_m,
            depth_m,
        ]


@contextmanager
def fake_realsense_module():
    old_module = sys.modules.get("pyrealsense2")
    sys.modules["pyrealsense2"] = FakeRealSenseModule
    try:
        yield
    finally:
        if old_module is None:
            sys.modules.pop("pyrealsense2", None)
        else:
            sys.modules["pyrealsense2"] = old_module


def create_run_dir(base_dir):
    base_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / stamp
    if not run_dir.exists():
        run_dir.mkdir()
        return run_dir

    for index in range(1, 1000):
        candidate = base_dir / f"{stamp}_{index:03d}"
        if not candidate.exists():
            candidate.mkdir()
            return candidate
    raise RuntimeError(f"Could not create unique run directory under {base_dir}")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_png(path, image):
    """Write uint8 grayscale/BGR/RGB-ish arrays without Pillow or OpenCV."""
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        height, width = image.shape
        color_type = 0
        rows = image
    elif image.ndim == 3 and image.shape[2] == 3:
        height, width = image.shape[:2]
        color_type = 2
        rows = image[:, :, ::-1]
    else:
        raise ValueError(f"Unsupported image shape for PNG: {image.shape}")

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    raw_rows = b"".join(b"\x00" + np.ascontiguousarray(row).tobytes() for row in rows)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw_rows, 6))
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return True


def write_image(path, image):
    cv2 = load_cv2()
    if cv2 is None:
        return write_png(path, image)
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), image))


def load_image(path):
    cv2 = load_cv2()
    if cv2 is None:
        raise RuntimeError("OpenCV is required to load image files.")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def load_depth(path, depth_scale):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=float) * depth_scale
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        key = "depth" if "depth" in data.files else data.files[0]
        return np.asarray(data[key], dtype=float) * depth_scale

    cv2 = load_cv2()
    if cv2 is None:
        raise RuntimeError("OpenCV is required to load image depth files.")
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Could not read depth image: {path}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return np.asarray(depth, dtype=float) * depth_scale


def target_mask_for_color(image, target_color):
    b = image[:, :, 0].astype(float)
    g = image[:, :, 1].astype(float)
    r = image[:, :, 2].astype(float)

    if target_color == "red":
        return (r > 120.0) & (r > g * 1.35) & (r > b * 1.35)
    if target_color == "green":
        return (g > 120.0) & (g > r * 1.25) & (g > b * 1.25)
    if target_color == "blue":
        return (b > 120.0) & (b > r * 1.25) & (b > g * 1.25)
    if target_color == "yellow":
        return (r > 130.0) & (g > 120.0) & (b < 120.0)
    raise ValueError(f"Unsupported target color: {target_color}")


def synthetic_scene(camera_name, resolution, target_color):
    width, height = resolution
    image = np.full((height, width, 3), (46, 48, 52), dtype=np.uint8)
    depth_m = np.full((height, width), 1.15, dtype=float)

    if camera_name == "d435":
        center = (int(width * 0.54), int(height * 0.44))
        radius = (max(18, width // 18), max(16, height // 16))
    else:
        center = (int(width * 0.49), int(height * 0.52))
        radius = (max(22, width // 16), max(18, height // 14))

    yy, xx = np.ogrid[:height, :width]
    target_mask = ((xx - center[0]) / radius[0]) ** 2 + ((yy - center[1]) / radius[1]) ** 2 <= 1.0
    distractor_mask = ((xx - int(width * 0.25)) / max(14, width // 24)) ** 2 + (
        (yy - int(height * 0.65)) / max(14, height // 24)
    ) ** 2 <= 1.0

    color_bgr = {
        "red": (30, 45, 220),
        "green": (45, 190, 50),
        "blue": (210, 80, 45),
        "yellow": (35, 210, 220),
    }[target_color]
    image[target_mask] = color_bgr
    image[distractor_mask] = (210, 85, 45)
    depth_m[target_mask] = 0.72
    depth_m[distractor_mask] = 0.90

    cv2 = load_cv2()
    if cv2 is not None:
        cv2.rectangle(image, (20, 20), (width - 21, height - 21), (80, 84, 90), 1)
        cv2.line(image, (width // 2, 25), (width // 2, height - 26), (68, 70, 74), 1)
        cv2.line(image, (25, height // 2), (width - 26, height // 2), (68, 70, 74), 1)
        cv2.putText(
            image,
            f"SYNTHETIC {camera_name.upper()}",
            (24, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (210, 210, 210),
            1,
        )

    return image, depth_m, {"target_center": [int(center[0]), int(center[1])]}


def segment_image(image, target_color):
    mask = target_mask_for_color(image, target_color).astype(np.uint8)
    cv2 = load_cv2()
    if cv2 is not None:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        if count <= 1:
            return {
                "valid": False,
                "reason": f"No {target_color} segment found.",
                "mask": mask,
            }
        areas = stats[1:, cv2.CC_STAT_AREA]
        label = int(np.argmax(areas)) + 1
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label]
        component_mask = (labels == label).astype(np.uint8)
        return {
            "valid": True,
            "pixel": [int(round(cx)), int(round(cy))],
            "area_px": int(stats[label, cv2.CC_STAT_AREA]),
            "bbox": [x, y, w, h],
            "mask": component_mask,
        }

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {"valid": False, "reason": f"No {target_color} segment found.", "mask": mask}
    return {
        "valid": True,
        "pixel": [int(round(float(xs.mean()))), int(round(float(ys.mean())))],
        "area_px": int(len(xs)),
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
        "mask": mask,
    }


def depth_from_segmentation(image, target_color, target_depth_m, background_depth_m):
    depth_m = np.full(image.shape[:2], float(background_depth_m), dtype=float)
    depth_m[target_mask_for_color(image, target_color)] = float(target_depth_m)
    return depth_m


def save_depth_colormap(path, depth_m):
    cv2 = load_cv2()

    valid = depth_m > 0.0
    normalized = np.zeros(depth_m.shape, dtype=np.uint8)
    if np.any(valid):
        min_depth = float(np.min(depth_m[valid]))
        max_depth = float(np.max(depth_m[valid]))
        if max_depth > min_depth:
            clipped = np.clip(depth_m, min_depth, max_depth)
            normalized = ((clipped - min_depth) / (max_depth - min_depth) * 255.0).astype(np.uint8)
            normalized[~valid] = 0
    if cv2 is None:
        ramp = normalized.astype(float) / 255.0
        colorized = np.zeros((*normalized.shape, 3), dtype=np.uint8)
        colorized[:, :, 0] = (255.0 * (1.0 - ramp)).astype(np.uint8)
        colorized[:, :, 1] = (180.0 * (1.0 - np.abs(ramp - 0.5) * 2.0)).astype(np.uint8)
        colorized[:, :, 2] = (255.0 * ramp).astype(np.uint8)
        colorized[~valid] = (0, 0, 0)
        return write_image(path, colorized)

    colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colorized[~valid] = (0, 0, 0)
    return write_image(path, colorized)


def coerce_pixel(point):
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return None
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in point):
        return None
    return [int(point[0]), int(point[1])]


def clamp_pixel(pixel, resolution):
    width, height = resolution
    return [
        max(0, min(width - 1, int(pixel[0]))),
        max(0, min(height - 1, int(pixel[1]))),
    ]


def offset_pixel(pixel, dx, dy, resolution):
    return clamp_pixel([pixel[0] + dx, pixel[1] + dy], resolution)


def parse_coord_list(text):
    coords = []
    for item in str(text).split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Expected 'u,v' coordinate, got: {item}")
        coords.append([int(parts[0]), int(parts[1])])
    return coords


def json_or_python_loads(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def load_jsonish_text(text):
    stripped = text.strip()
    try:
        return json_or_python_loads(stripped)
    except Exception:
        pass

    block = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if block:
        try:
            return json_or_python_loads(block.group(1).strip())
        except Exception:
            pass

    start = stripped.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM response text.")

    for end in range(len(stripped), start, -1):
        candidate = stripped[start:end].strip()
        if not candidate.endswith("}"):
            continue
        try:
            return json_or_python_loads(candidate)
        except Exception:
            continue

    raise ValueError("Could not parse a JSON object from LLM response text.")


def decode_tool_arguments(arguments):
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        parsed = load_jsonish_text(arguments)
        if not isinstance(parsed, dict):
            raise ValueError("Tool arguments parsed successfully but were not a JSON object.")
        return parsed
    raise ValueError(f"Unsupported tool arguments type: {type(arguments).__name__}")


def normalize_tool_call(obj, default_camera):
    if not isinstance(obj, dict):
        raise ValueError(f"Expected LLM response JSON object, got {type(obj).__name__}")

    if "choices" in obj and obj["choices"]:
        return normalize_tool_call(obj["choices"][0].get("message", {}), default_camera)

    if "message" in obj and isinstance(obj["message"], dict):
        return normalize_tool_call(obj["message"], default_camera)

    if "tool_calls" in obj and obj["tool_calls"]:
        for index, tool_call in enumerate(obj["tool_calls"]):
            try:
                tool_name, tool_args, selected_index = normalize_tool_call(tool_call, default_camera)
            except ValueError:
                continue
            if tool_name in LOCALIZATION_TOOLS:
                return tool_name, tool_args, selected_index if selected_index is not None else index
        return normalize_tool_call(obj["tool_calls"][0], default_camera)

    if "function" in obj and isinstance(obj["function"], dict):
        function = obj["function"]
        return (
            function.get("name"),
            decode_tool_arguments(function.get("arguments", {})),
            None,
        )

    if "name" in obj and "arguments" in obj:
        return obj["name"], decode_tool_arguments(obj["arguments"]), None

    if "tool_name" in obj:
        return obj["tool_name"], decode_tool_arguments(obj.get("tool_args", obj.get("arguments", {}))), None

    if "d435_coords" in obj or "d405_coords" in obj:
        return "get_xyz_fused", obj, None

    if "coords" in obj:
        return f"get_xyz_{default_camera}", obj, None

    raise ValueError("Could not find a tool call or localization arguments in the response.")


def validate_tool_call(tool_name, tool_args):
    if tool_name == "get_xyz_fused":
        d435_coords, error = dispatcher._optional_coords(tool_args, "d435_coords")
        if error:
            return {"valid": False, "error": error}
        d405_coords, error = dispatcher._optional_coords(tool_args, "d405_coords")
        if error:
            return {"valid": False, "error": error}
        if not d435_coords and not d405_coords:
            return {
                "valid": False,
                "error": "No coordinates provided to get_xyz_fused.",
            }
        return {"valid": True, "d435_coords": d435_coords, "d405_coords": d405_coords}

    if tool_name in ("get_xyz_d435", "get_xyz_d405"):
        coords = tool_args.get("coords", [])
        error = dispatcher._validate_coord_pairs(coords, "coords", require_nonempty=True)
        if error:
            return {"valid": False, "error": error}
        camera_name = "d435" if tool_name == "get_xyz_d435" else "d405"
        return {
            "valid": True,
            "d435_coords": coords if camera_name == "d435" else [],
            "d405_coords": coords if camera_name == "d405" else [],
        }

    return {
        "valid": False,
        "error": f"Tool '{tool_name}' is not a localization tool.",
    }


def fake_translate_point_to_robot_frame(point_xyz, source_camera, robot_ee_pose=None):
    try:
        point = np.array(point_xyz, dtype=float)
    except (TypeError, ValueError):
        return {
            "xyz": None,
            "valid": False,
            "source": source_camera,
            "reason": f"Expected numeric point, got {point_xyz}",
        }
    if point.shape != (3,):
        return {
            "xyz": None,
            "valid": False,
            "source": source_camera,
            "reason": f"Expected 3D point, got {point_xyz}",
        }

    offset = CAMERA_OFFSETS_M.get(source_camera, np.zeros(3, dtype=float))
    return {
        "xyz": (point + offset).tolist(),
        "valid": True,
        "source": source_camera,
        "reason": "offline fake camera-to-robot transform",
    }


def depth_window_stats(depth_m, pixel, radius):
    clean = coerce_pixel(pixel)
    if clean is None:
        return {"valid": False, "reason": f"Pixel is not integer [u, v]: {pixel}"}

    height, width = depth_m.shape
    u, v = clean
    if not (0 <= u < width and 0 <= v < height):
        return {
            "valid": False,
            "pixel": clean,
            "reason": f"Pixel out of bounds for depth image {width}x{height}",
        }

    x0 = max(0, u - radius)
    x1 = min(width, u + radius + 1)
    y0 = max(0, v - radius)
    y1 = min(height, v + radius + 1)
    window = depth_m[y0:y1, x0:x1]
    valid = window[window > 0.0]
    stats = {
        "valid": bool(depth_m[v, u] > 0.0),
        "pixel": clean,
        "center_depth_m": float(depth_m[v, u]),
        "window_radius_px": int(radius),
        "valid_window_px": int(valid.size),
    }
    if valid.size:
        stats.update(
            {
                "window_min_m": float(valid.min()),
                "window_median_m": float(np.median(valid)),
                "window_max_m": float(valid.max()),
            }
        )
    return stats


def process_points(camera_name, points, resolution, depth_frame):
    return [
        localization.process_pixel(
            point=point,
            resolution=resolution,
            depth_rs=depth_frame,
            depth_scale=1.0,
            source_camera=camera_name,
            robot_ee_pose=np.eye(4).tolist(),
        )
        for point in points
    ]


def pixel_delta_px(first_pixel, second_pixel):
    a = coerce_pixel(first_pixel)
    b = coerce_pixel(second_pixel)
    if a is None or b is None:
        return None
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def draw_square(image, u, v, color, radius=2):
    height, width = image.shape[:2]
    x0 = max(0, int(u) - radius)
    x1 = min(width, int(u) + radius + 1)
    y0 = max(0, int(v) - radius)
    y1 = min(height, int(v) + radius + 1)
    image[y0:y1, x0:x1] = color


def draw_cross(image, u, v, color, size=12):
    height, width = image.shape[:2]
    u = int(u)
    v = int(v)
    if not (0 <= u < width and 0 <= v < height):
        return
    x0 = max(0, u - size)
    x1 = min(width, u + size + 1)
    y0 = max(0, v - size)
    y1 = min(height, v + size + 1)
    image[v : v + 1, x0:x1] = color
    image[y0:y1, u : u + 1] = color


def draw_rect(image, bbox, color, thickness=2):
    if not bbox:
        return
    x, y, w, h = [int(value) for value in bbox]
    height, width = image.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width - 1, x + w)
    y1 = min(height - 1, y + h)
    if x0 >= x1 or y0 >= y1:
        return
    image[y0 : min(height, y0 + thickness), x0 : x1 + 1] = color
    image[max(0, y1 - thickness + 1) : y1 + 1, x0 : x1 + 1] = color
    image[y0 : y1 + 1, x0 : min(width, x0 + thickness)] = color
    image[y0 : y1 + 1, max(0, x1 - thickness + 1) : x1 + 1] = color


def save_overlay(path, image, segmentation, llm_coords, localization_results):
    cv2 = load_cv2()

    overlay = image.copy()
    mask = segmentation.get("mask")
    if mask is not None:
        tint = overlay.copy()
        tint[mask.astype(bool)] = (0, 210, 255)
        if cv2 is not None:
            overlay = cv2.addWeighted(tint, 0.28, overlay, 0.72, 0)
        else:
            overlay = (tint.astype(float) * 0.28 + overlay.astype(float) * 0.72).astype(np.uint8)

    if segmentation.get("valid"):
        u, v = segmentation["pixel"]
        bbox = segmentation.get("bbox")
        if cv2 is not None and bbox:
            x, y, w, h = bbox
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 210, 255), 2)
            cv2.drawMarker(overlay, (u, v), (0, 210, 255), cv2.MARKER_TILTED_CROSS, 24, 2)
            cv2.putText(overlay, "SEG", (u + 10, v + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 255), 2)
        else:
            draw_rect(overlay, bbox, (0, 210, 255))
            draw_cross(overlay, u, v, (0, 210, 255), size=13)

    for index, point in enumerate(llm_coords):
        pixel = coerce_pixel(point)
        if pixel is None:
            continue
        color = (0, 255, 0)
        if index < len(localization_results) and localization_results[index].get("status") != "ok":
            color = (0, 0, 255)
        u, v = pixel
        if cv2 is not None:
            cv2.drawMarker(overlay, (u, v), color, cv2.MARKER_CROSS, 24, 2)
            cv2.circle(overlay, (u, v), 5, color, -1)
            cv2.putText(overlay, f"LLM{index}", (u + 10, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            draw_cross(overlay, u, v, color, size=15)
            draw_square(overlay, u, v, color, radius=3)

    return write_image(path, overlay)


def camera_report(camera_name, image, depth_m, segmentation, llm_coords, output_dir):
    resolution = (image.shape[1], image.shape[0])
    depth_frame = FakeDepthFrame(depth_m)
    segmentation_points = [segmentation["pixel"]] if segmentation.get("valid") else []
    llm_results = process_points(camera_name, llm_coords, resolution, depth_frame)
    segmentation_results = process_points(camera_name, segmentation_points, resolution, depth_frame)

    mask_path = output_dir / f"{camera_name}_segmentation_mask.png"
    overlay_path = output_dir / f"{camera_name}_segmentation_llm_overlay.png"
    depth_path = output_dir / f"{camera_name}_depth_colormap.png"

    mask_saved = False
    if segmentation.get("mask") is not None:
        mask_saved = write_image(mask_path, segmentation["mask"].astype(np.uint8) * 255)
    overlay_saved = save_overlay(overlay_path, image, segmentation, llm_coords, llm_results)
    depth_saved = save_depth_colormap(depth_path, depth_m)

    working_style_saved = localization.save_debug_image(image, llm_results, camera_name, "offline")

    first_llm_pixel = llm_coords[0] if llm_coords else None
    return {
        "camera": camera_name,
        "resolution": [int(resolution[0]), int(resolution[1])],
        "segmentation": {
            key: value
            for key, value in segmentation.items()
            if key != "mask"
        },
        "llm_coords": llm_coords,
        "llm_to_segmentation_delta_px": pixel_delta_px(first_llm_pixel, segmentation.get("pixel")),
        "llm_depth_stats": [depth_window_stats(depth_m, point, radius=5) for point in llm_coords],
        "segmentation_depth_stats": [
            depth_window_stats(depth_m, point, radius=5) for point in segmentation_points
        ],
        "llm_localization_results": llm_results,
        "segmentation_localization_results": segmentation_results,
        "artifacts": {
            "input_image": str(output_dir / f"{camera_name}_input.png"),
            "segmentation_mask": str(mask_path) if mask_saved else None,
            "overlay": str(overlay_path) if overlay_saved else None,
            "depth_colormap": str(depth_path) if depth_saved else None,
            "working_style_localization_debug": (
                str(output_dir / "debug_images" / f"{camera_name}_offline_debug.jpg")
                if working_style_saved
                else None
            ),
        },
    }


def extract_coords_from_validation(validation):
    return {
        "d435": validation.get("d435_coords", []) if validation.get("valid") else [],
        "d405": validation.get("d405_coords", []) if validation.get("valid") else [],
    }


def load_or_create_camera_data(args, camera_name, output_dir):
    resolution = cfg.D435_RESOLUTION if camera_name == "d435" else cfg.D405_RESOLUTION
    image_path = args.d435_image if camera_name == "d435" else args.d405_image
    depth_path = args.d435_depth if camera_name == "d435" else args.d405_depth

    synthetic_meta = {}
    if image_path:
        image = load_image(image_path)
        synthetic_depth = None
    else:
        image, synthetic_depth, synthetic_meta = synthetic_scene(camera_name, resolution, args.target_color)

    segmentation = segment_image(image, args.target_color)

    if depth_path:
        depth_m = load_depth(depth_path, args.depth_scale)
    elif synthetic_depth is not None:
        depth_m = synthetic_depth
    else:
        depth_m = depth_from_segmentation(
            image,
            args.target_color,
            target_depth_m=args.synthetic_target_depth_m,
            background_depth_m=args.synthetic_background_depth_m,
        )

    if depth_m.shape[:2] != image.shape[:2]:
        raise RuntimeError(
            f"{camera_name} depth shape {depth_m.shape[:2]} does not match image shape {image.shape[:2]}"
        )

    write_image(output_dir / f"{camera_name}_input.png", image)
    return image, depth_m, segmentation, synthetic_meta


def default_tool_call(d435_segmentation, d405_segmentation, d435_resolution, d405_resolution):
    d435_pixel = d435_segmentation.get("pixel") or [d435_resolution[0] // 2, d435_resolution[1] // 2]
    d405_pixel = d405_segmentation.get("pixel") or [d405_resolution[0] // 2, d405_resolution[1] // 2]
    return "get_xyz_fused", {
        "d435_coords": [offset_pixel(d435_pixel, 6, -5, d435_resolution)],
        "d405_coords": [offset_pixel(d405_pixel, -7, 4, d405_resolution)],
    }


def apply_coord_overrides(tool_name, tool_args, args):
    d435_override = parse_coord_list(args.d435_coords) if args.d435_coords else None
    d405_override = parse_coord_list(args.d405_coords) if args.d405_coords else None

    if d435_override is None and d405_override is None:
        return tool_name, tool_args

    if tool_name == "get_xyz_fused":
        updated = dict(tool_args)
        if d435_override is not None:
            updated["d435_coords"] = d435_override
        if d405_override is not None:
            updated["d405_coords"] = d405_override
        return tool_name, updated

    return "get_xyz_fused", {
        "d435_coords": d435_override,
        "d405_coords": d405_override,
    }


def summarize_camera(report):
    seg = report["segmentation"]
    seg_pixel = seg.get("pixel") if seg.get("valid") else None
    first_llm = report["llm_coords"][0] if report["llm_coords"] else None
    first_depth = report["llm_depth_stats"][0].get("center_depth_m") if report["llm_depth_stats"] else None
    ok_count = sum(1 for item in report["llm_localization_results"] if item.get("status") == "ok")
    return (
        f"{report['camera'].upper()}: seg={seg_pixel} llm={first_llm} "
        f"delta_px={report['llm_to_segmentation_delta_px']} depth_m={first_depth} ok={ok_count}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Debug LLM pixel output, color segmentation, and depth-to-XYZ coordination offline."
    )
    parser.add_argument("--llm-response-file", type=Path, help="Saved LLM/tool-call response JSON or text.")
    parser.add_argument("--d435-image", type=Path, help="Optional saved D435 BGR/RGB image file.")
    parser.add_argument("--d405-image", type=Path, help="Optional saved D405 BGR/RGB image file.")
    parser.add_argument("--d435-depth", type=Path, help="Optional D435 depth .npy/.npz/image file.")
    parser.add_argument("--d405-depth", type=Path, help="Optional D405 depth .npy/.npz/image file.")
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="Scale applied to loaded depth files. Use 0.001 for raw RealSense millimeters.",
    )
    parser.add_argument(
        "--target-color",
        choices=["red", "green", "blue", "yellow"],
        default="red",
        help="Color used by the simple segmentation debugger.",
    )
    parser.add_argument("--d435-coords", help="Override D435 LLM coords, format 'u,v' or 'u,v;u,v'.")
    parser.add_argument("--d405-coords", help="Override D405 LLM coords, format 'u,v' or 'u,v;u,v'.")
    parser.add_argument(
        "--default-camera",
        choices=["d435", "d405"],
        default="d435",
        help="Camera to assume when a raw response only has {'coords': ...}.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--synthetic-target-depth-m", type=float, default=0.72)
    parser.add_argument("--synthetic-background-depth-m", type=float, default=1.15)
    args = parser.parse_args()

    run_dir = create_run_dir(args.output_dir)
    debug_image_dir = run_dir / "debug_images"
    debug_image_dir.mkdir(parents=True, exist_ok=True)

    d435_image, d435_depth, d435_segmentation, d435_meta = load_or_create_camera_data(args, "d435", run_dir)
    d405_image, d405_depth, d405_segmentation, d405_meta = load_or_create_camera_data(args, "d405", run_dir)

    if args.llm_response_file:
        response_text = args.llm_response_file.read_text(encoding="utf-8")
        response_obj = load_jsonish_text(response_text)
        tool_name, tool_args, selected_index = normalize_tool_call(response_obj, args.default_camera)
    else:
        tool_name, tool_args = default_tool_call(
            d435_segmentation,
            d405_segmentation,
            (d435_image.shape[1], d435_image.shape[0]),
            (d405_image.shape[1], d405_image.shape[0]),
        )
        selected_index = None

    tool_name, tool_args = apply_coord_overrides(tool_name, tool_args, args)
    validation = validate_tool_call(tool_name, tool_args)
    coords_by_camera = extract_coords_from_validation(validation)

    with fake_realsense_module(), patched_attr(
        localization,
        "translate_point_to_robot_frame",
        fake_translate_point_to_robot_frame,
    ), patched_attr(cfg, "DEBUG_IMAGE_DIR", str(debug_image_dir)):
        d435_report = camera_report(
            "d435",
            d435_image,
            d435_depth,
            d435_segmentation,
            coords_by_camera["d435"],
            run_dir,
        )
        d405_report = camera_report(
            "d405",
            d405_image,
            d405_depth,
            d405_segmentation,
            coords_by_camera["d405"],
            run_dir,
        )

        llm_fusion = translate_points_fused(
            localization.valid_robot_points(d435_report["llm_localization_results"], "d435"),
            localization.valid_robot_points(d405_report["llm_localization_results"], "d405"),
        )
        segmentation_fusion = translate_points_fused(
            localization.valid_robot_points(d435_report["segmentation_localization_results"], "d435"),
            localization.valid_robot_points(d405_report["segmentation_localization_results"], "d405"),
        )

    report = {
        "run_dir": str(run_dir),
        "purpose": "offline debug for LLM response pixels, segmentation pixels, and fake-depth XYZ conversion",
        "runtime_debug_image_writer": "Working/vision/tools/localization.py::save_debug_image",
        "runtime_debug_image_setting": "Working/config.py::DEBUG_IMAGE_DIR",
        "offline_transform_note": (
            "xyz_robot values use a fake transform in this harness. Use xyz_camera/depth stats "
            "to debug pixel-depth coordination independent of robot calibration."
        ),
        "llm_response": {
            "source": str(args.llm_response_file) if args.llm_response_file else "synthetic mock tool call",
            "selected_tool_call_index": selected_index,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "validation": validation,
        },
        "synthetic_scene": {
            "d435": d435_meta,
            "d405": d405_meta,
        },
        "cameras": {
            "d435": d435_report,
            "d405": d405_report,
        },
        "fusion": {
            "llm_pixels": llm_fusion,
            "segmentation_pixels": segmentation_fusion,
        },
    }
    report_path = run_dir / "report.json"
    write_json(report_path, report)

    print("--- Offline LLM / Segmentation / Depth Debug ---")
    print(f"Output folder: {run_dir}")
    print(f"Tool call: {tool_name} {json.dumps(tool_args)}")
    if not validation["valid"]:
        print(f"Validation: FAIL - {validation['error']}")
    else:
        print("Validation: PASS")
    print(summarize_camera(d435_report))
    print(summarize_camera(d405_report))
    print(f"LLM fusion valid: {llm_fusion.get('valid')} xyz={llm_fusion.get('xyz')}")
    print(f"Segmentation fusion valid: {segmentation_fusion.get('valid')} xyz={segmentation_fusion.get('xyz')}")
    print(f"Report: {report_path}")

    return 0 if validation["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
