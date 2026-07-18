"""
Live LLM / Camera / Segmentation Debug Harness
----------------------------------------------
Uses the configured RealSense camera hardware and local OpenAI-compatible LLM
server, but never connects to the Franka robot and never commands motion.

This diagnostic answers:
- Can the D435/D405 produce aligned RGB/depth frames?
- Can the LLM inspect those real camera images and return object pixels?
- Do color-segmentation pixels agree with the LLM-selected pixels?
- Is there valid depth at the selected pixels, and what is the camera-frame XYZ?

Example:
    python3 Module_Tests/Working_Tests/live_llm_camera_seg_debug/live_llm_camera_seg_debug.py \
        --target "red block" --target-color red
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
import json
import math
from pathlib import Path
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
WORKING_TESTS_DIR = SCRIPT_DIR.parent
if str(WORKING_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(WORKING_TESTS_DIR))

from _working_test_utils import TEST_OUTPUTS_DIR  # noqa: E402

import config as cfg  # noqa: E402
from hardware.camera import RealSense  # noqa: E402
from utilities.coordinates import pixel_to_xyz  # noqa: E402


DEFAULT_OUTPUT_DIR = TEST_OUTPUTS_DIR / "live_llm_camera_seg_debug"
CAMERA_INFO = {
    "d435": {
        "label": "D435 overhead bird's-eye camera",
        "serial": lambda: cfg.D435_SERIAL,
        "resolution": lambda: cfg.D435_RESOLUTION,
    },
    "d405": {
        "label": "D405 wrist/eye-in-hand camera",
        "serial": lambda: cfg.D405_SERIAL,
        "resolution": lambda: cfg.D405_RESOLUTION,
    },
}


def configured_serial(value):
    if not value or str(value).startswith("YOUR_"):
        return None
    return value


def load_cv2():
    try:
        import cv2

        return cv2
    except ModuleNotFoundError:
        return None


def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", checksum)
    )


def bgr_to_png_bytes(bgr_image) -> bytes:
    image = np.asarray(bgr_image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 BGR image, got {image.shape}")

    height, width = image.shape[:2]
    rgb = image[:, :, ::-1]
    raw_rows = b"".join(b"\x00" + np.ascontiguousarray(row).tobytes() for row in rgb)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(raw_rows, 6))
        + png_chunk(b"IEND", b"")
    )


def write_png(path: Path, image) -> bool:
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
        raise ValueError(f"Unsupported PNG image shape: {image.shape}")

    raw_rows = b"".join(b"\x00" + np.ascontiguousarray(row).tobytes() for row in rows)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(raw_rows, 6))
        + png_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return True


def write_image(path: Path, image) -> bool:
    cv2 = load_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    if cv2 is None:
        return write_png(path, image)
    return bool(cv2.imwrite(str(path), image))


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def create_run_dir(base_dir: Path) -> Path:
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


def image_url_content(bgr_image) -> dict:
    encoded = base64.b64encode(bgr_to_png_bytes(bgr_image)).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{encoded}"},
    }


def build_prompt(target: str, cameras: list[str], target_color: str) -> str:
    camera_lines = []
    for index, camera_name in enumerate(cameras, start=1):
        width, height = CAMERA_INFO[camera_name]["resolution"]()
        camera_lines.append(
            f"Image {index}: {camera_name.upper()} ({CAMERA_INFO[camera_name]['label']}), "
            f"resolution {width}x{height}."
        )

    return (
        "You are debugging a robotic vision system. Inspect the real camera image(s) "
        "and identify the requested target object. Return JSON only, no markdown.\n\n"
        + "\n".join(camera_lines)
        + "\n\n"
        f"Requested target: {target}\n"
        f"Segmentation color being checked separately: {target_color}\n\n"
        "Use pixel coordinates [u, v] with origin at the top-left of each image. "
        "If the target is not visible in a camera, set visible=false and center_pixel=null.\n\n"
        "Return exactly this JSON shape:\n"
        "{\n"
        '  "target_label": "short object name",\n'
        '  "d435": {"visible": true, "center_pixel": [u, v], "bbox_xywh": [x, y, w, h], "confidence": 0.0, "evidence": "brief visual evidence"},\n'
        '  "d405": {"visible": true, "center_pixel": [u, v], "bbox_xywh": [x, y, w, h], "confidence": 0.0, "evidence": "brief visual evidence"}\n'
        "}\n"
        "Only include d435/d405 keys for cameras that were provided."
    )


def build_messages(target: str, captures: dict, target_color: str) -> list[dict]:
    cameras = list(captures.keys())
    content = [{"type": "text", "text": build_prompt(target, cameras, target_color)}]
    for camera_name in cameras:
        content.append({"type": "text", "text": f"Next image is {camera_name.upper()}."})
        content.append(image_url_content(captures[camera_name]["rgb"]))

    return [
        {
            "role": "system",
            "content": "You are a strict visual perception diagnostic. Return compact JSON only.",
        },
        {
            "role": "user",
            "content": content,
        },
    ]


def build_chat_payload(messages: list[dict]) -> dict:
    return {
        "model": cfg.QWEN_MODEL_PATH,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 700,
    }


def chat_completion(payload: dict, timeout_s: float) -> tuple[str, dict]:
    endpoint = cfg.LLM_API_URL.rstrip("/") + "/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.LLM_API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {endpoint}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach LLM server at {endpoint}: {exc}") from exc

    data = json.loads(response_body)
    reply = data["choices"][0]["message"].get("content") or ""
    return reply, data


def parse_jsonish(text: str):
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    block = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if block:
        try:
            return json.loads(block.group(1).strip())
        except json.JSONDecodeError:
            pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return json.loads(stripped[start : end + 1])

    raise ValueError("Could not parse JSON object from LLM reply.")


def target_mask_for_color(image, color_name: str):
    if color_name == "none":
        return np.zeros(image.shape[:2], dtype=bool)

    b = image[:, :, 0].astype(float)
    g = image[:, :, 1].astype(float)
    r = image[:, :, 2].astype(float)

    if color_name == "red":
        return (r > 110.0) & (r > g * 1.35) & (r > b * 1.35)
    if color_name == "green":
        return (g > 100.0) & (g > r * 1.25) & (g > b * 1.25)
    if color_name == "blue":
        return (b > 100.0) & (b > r * 1.25) & (b > g * 1.25)
    if color_name == "yellow":
        return (r > 125.0) & (g > 115.0) & (b < 120.0)
    if color_name == "orange":
        return (r > 130.0) & (g > 70.0) & (g < r * 0.85) & (b < 100.0)
    if color_name == "white":
        return (r > 180.0) & (g > 180.0) & (b > 180.0)
    if color_name == "black":
        return (r < 60.0) & (g < 60.0) & (b < 60.0)

    raise ValueError(f"Unsupported target color: {color_name}")


def segment_image(image, color_name: str) -> dict:
    mask = target_mask_for_color(image, color_name).astype(np.uint8)
    if color_name == "none":
        return {"valid": False, "reason": "Segmentation disabled.", "mask": mask}

    cv2 = load_cv2()
    if cv2 is not None:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        if count <= 1:
            return {"valid": False, "reason": f"No {color_name} segment found.", "mask": mask}
        areas = stats[1:, cv2.CC_STAT_AREA]
        label = int(np.argmax(areas)) + 1
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label]
        return {
            "valid": True,
            "pixel": [int(round(cx)), int(round(cy))],
            "bbox_xywh": [x, y, w, h],
            "area_px": int(stats[label, cv2.CC_STAT_AREA]),
            "mask": (labels == label).astype(np.uint8),
        }

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {"valid": False, "reason": f"No {color_name} segment found.", "mask": mask}
    return {
        "valid": True,
        "pixel": [int(round(float(xs.mean()))), int(round(float(ys.mean())))],
        "bbox_xywh": [
            int(xs.min()),
            int(ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        ],
        "area_px": int(len(xs)),
        "mask": mask,
        "reason": "OpenCV unavailable; using centroid of all matching color pixels.",
    }


def coerce_pixel(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        return None
    u = int(round(float(value[0])))
    v = int(round(float(value[1])))
    return [u, v]


def llm_camera_result(parsed_reply, camera_name: str) -> dict:
    if not isinstance(parsed_reply, dict):
        return {"visible": False, "center_pixel": None, "reason": "Parsed reply is not a JSON object."}

    value = parsed_reply.get(camera_name)
    if isinstance(value, dict):
        value = dict(value)
        value["center_pixel"] = coerce_pixel(value.get("center_pixel"))
        return value

    for key in (f"{camera_name}_center_pixel", f"{camera_name}_pixel"):
        if key in parsed_reply:
            return {
                "visible": True,
                "center_pixel": coerce_pixel(parsed_reply.get(key)),
                "reason": f"Read fallback key {key}.",
            }

    objects = parsed_reply.get("objects")
    if isinstance(objects, list):
        for item in objects:
            if not isinstance(item, dict):
                continue
            if str(item.get("camera", "")).lower() == camera_name:
                item = dict(item)
                item["center_pixel"] = coerce_pixel(item.get("center_pixel") or item.get("pixel"))
                item.setdefault("visible", item["center_pixel"] is not None)
                return item

    return {"visible": False, "center_pixel": None, "reason": f"No {camera_name} object in LLM reply."}


def depth_window_stats(depth_m, pixel, radius=5):
    clean = coerce_pixel(pixel)
    if clean is None:
        return {"valid": False, "reason": f"Invalid pixel: {pixel}"}

    height, width = depth_m.shape
    u, v = clean
    if not (0 <= u < width and 0 <= v < height):
        return {"valid": False, "pixel": clean, "reason": f"Out of bounds for {width}x{height}"}

    x0 = max(0, u - radius)
    x1 = min(width, u + radius + 1)
    y0 = max(0, v - radius)
    y1 = min(height, v + radius + 1)
    window = depth_m[y0:y1, x0:x1]
    valid = window[window > 0.0]
    result = {
        "valid": bool(depth_m[v, u] > 0.0),
        "pixel": clean,
        "center_depth_m": float(depth_m[v, u]),
        "valid_window_px": int(valid.size),
        "window_radius_px": int(radius),
    }
    if valid.size:
        result.update(
            {
                "window_min_m": float(valid.min()),
                "window_median_m": float(np.median(valid)),
                "window_max_m": float(valid.max()),
            }
        )
    return result


def pixel_to_camera_xyz_report(pixel, depth_rs, depth_scale):
    clean = coerce_pixel(pixel)
    if clean is None:
        return {"valid": False, "reason": f"Invalid pixel: {pixel}", "pixel": None}
    data = pixel_to_xyz(clean[0], clean[1], depth_rs, depth_scale)
    return {
        "pixel": clean,
        "valid": bool(data.get("valid")),
        "xyz_camera_m": [data.get("x"), data.get("y"), data.get("z")] if data.get("valid") else None,
        "reason": None if data.get("valid") else "No valid depth/deprojection at this pixel.",
    }


def pixel_delta(a, b):
    pa = coerce_pixel(a)
    pb = coerce_pixel(b)
    if pa is None or pb is None:
        return None
    return float(math.hypot(pa[0] - pb[0], pa[1] - pb[1]))


def draw_square(image, u, v, color, radius=3):
    height, width = image.shape[:2]
    x0 = max(0, int(u) - radius)
    x1 = min(width, int(u) + radius + 1)
    y0 = max(0, int(v) - radius)
    y1 = min(height, int(v) + radius + 1)
    image[y0:y1, x0:x1] = color


def draw_cross(image, pixel, color, size=16):
    clean = coerce_pixel(pixel)
    if clean is None:
        return
    u, v = clean
    height, width = image.shape[:2]
    if not (0 <= u < width and 0 <= v < height):
        return
    x0 = max(0, u - size)
    x1 = min(width, u + size + 1)
    y0 = max(0, v - size)
    y1 = min(height, v + size + 1)
    image[v : v + 1, x0:x1] = color
    image[y0:y1, u : u + 1] = color
    draw_square(image, u, v, color, radius=3)


def draw_rect(image, bbox, color, thickness=2):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return
    x, y, w, h = [int(round(float(value))) for value in bbox]
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


def save_overlay(path: Path, image, llm_result: dict, segmentation: dict) -> bool:
    overlay = image.copy()
    mask = segmentation.get("mask")
    if mask is not None:
        tint = overlay.copy()
        tint[mask.astype(bool)] = (0, 210, 255)
        overlay = (tint.astype(float) * 0.28 + overlay.astype(float) * 0.72).astype(np.uint8)

    if segmentation.get("valid"):
        draw_rect(overlay, segmentation.get("bbox_xywh"), (0, 210, 255), thickness=2)
        draw_cross(overlay, segmentation.get("pixel"), (0, 210, 255), size=14)

    if llm_result.get("center_pixel") is not None:
        draw_rect(overlay, llm_result.get("bbox_xywh"), (0, 255, 0), thickness=2)
        draw_cross(overlay, llm_result.get("center_pixel"), (0, 255, 0), size=18)

    return write_image(path, overlay)


def colorize_depth(depth_m):
    valid = depth_m > 0.0
    normalized = np.zeros(depth_m.shape, dtype=np.uint8)
    if np.any(valid):
        min_depth = float(np.min(depth_m[valid]))
        max_depth = float(np.max(depth_m[valid]))
        if max_depth > min_depth:
            clipped = np.clip(depth_m, min_depth, max_depth)
            normalized = ((clipped - min_depth) / (max_depth - min_depth) * 255.0).astype(np.uint8)
            normalized[~valid] = 0

    ramp = normalized.astype(float) / 255.0
    colorized = np.zeros((*normalized.shape, 3), dtype=np.uint8)
    colorized[:, :, 0] = (255.0 * (1.0 - ramp)).astype(np.uint8)
    colorized[:, :, 1] = (180.0 * (1.0 - np.abs(ramp - 0.5) * 2.0)).astype(np.uint8)
    colorized[:, :, 2] = (255.0 * ramp).astype(np.uint8)
    colorized[~valid] = (0, 0, 0)
    return colorized


def start_camera(camera_name: str) -> RealSense:
    serial = configured_serial(CAMERA_INFO[camera_name]["serial"]())
    resolution = CAMERA_INFO[camera_name]["resolution"]()
    if serial is None:
        print(f"Warning: {camera_name.upper()} serial is not configured; opening by stream type.")
    print(f"Starting {camera_name.upper()} at {resolution}...")
    return RealSense(serial_number=serial, resolution=resolution, fps=cfg.CAMERA_FPS)


def stop_camera(camera, camera_name: str) -> None:
    if camera is None:
        return
    try:
        camera.stop()
    except Exception as exc:
        print(f"Warning: failed to stop {camera_name}: {exc}")


def capture_live_frames(cameras: dict, camera_names: list[str]):
    if len(camera_names) == 2:
        synced = cameras["d435"].grab_synced_snapshot(
            cameras["d405"],
            max_delta_ms=cfg.CAMERA_SYNC_TOLERANCE_MS,
        )
        if synced is None:
            raise RuntimeError(
                f"D435/D405 failed to synchronize within {cfg.CAMERA_SYNC_TOLERANCE_MS}ms."
            )
        return {
            "d435": {
                "rgb": synced[0],
                "depth_array": synced[1],
                "depth_rs": synced[2],
                "depth_scale": cameras["d435"].depth_scale,
            },
            "d405": {
                "rgb": synced[3],
                "depth_array": synced[4],
                "depth_rs": synced[5],
                "depth_scale": cameras["d405"].depth_scale,
            },
        }

    camera_name = camera_names[0]
    rgb, depth_array, depth_rs = cameras[camera_name].get_frames()
    if rgb is None or depth_array is None or depth_rs is None:
        raise RuntimeError(f"{camera_name.upper()} did not return a complete RGB/depth frame.")
    return {
        camera_name: {
            "rgb": rgb,
            "depth_array": depth_array,
            "depth_rs": depth_rs,
            "depth_scale": cameras[camera_name].depth_scale,
        }
    }


def save_capture_artifacts(run_dir: Path, captures: dict) -> None:
    for camera_name, capture in captures.items():
        write_image(run_dir / f"{camera_name}_rgb.png", capture["rgb"])
        depth_m = capture["depth_array"].astype(float) * float(capture["depth_scale"])
        np.save(run_dir / f"{camera_name}_depth_raw.npy", capture["depth_array"])
        write_image(run_dir / f"{camera_name}_depth_colormap.png", colorize_depth(depth_m))


def camera_analysis(camera_name: str, capture: dict, llm_result: dict, segmentation: dict, run_dir: Path):
    depth_m = capture["depth_array"].astype(float) * float(capture["depth_scale"])
    llm_pixel = llm_result.get("center_pixel")
    seg_pixel = segmentation.get("pixel") if segmentation.get("valid") else None

    mask_path = run_dir / f"{camera_name}_segmentation_mask.png"
    overlay_path = run_dir / f"{camera_name}_llm_vs_segmentation_overlay.png"
    if segmentation.get("mask") is not None:
        write_image(mask_path, segmentation["mask"].astype(np.uint8) * 255)
    overlay_saved = save_overlay(overlay_path, capture["rgb"], llm_result, segmentation)

    return {
        "camera": camera_name,
        "resolution": [int(capture["rgb"].shape[1]), int(capture["rgb"].shape[0])],
        "llm": llm_result,
        "segmentation": {
            key: value
            for key, value in segmentation.items()
            if key != "mask"
        },
        "llm_to_segmentation_delta_px": pixel_delta(llm_pixel, seg_pixel),
        "llm_depth": depth_window_stats(depth_m, llm_pixel),
        "segmentation_depth": depth_window_stats(depth_m, seg_pixel),
        "llm_xyz_camera": pixel_to_camera_xyz_report(llm_pixel, capture["depth_rs"], capture["depth_scale"]),
        "segmentation_xyz_camera": pixel_to_camera_xyz_report(
            seg_pixel,
            capture["depth_rs"],
            capture["depth_scale"],
        ),
        "artifacts": {
            "rgb": str(run_dir / f"{camera_name}_rgb.png"),
            "depth_raw_npy": str(run_dir / f"{camera_name}_depth_raw.npy"),
            "depth_colormap": str(run_dir / f"{camera_name}_depth_colormap.png"),
            "segmentation_mask": str(mask_path) if segmentation.get("mask") is not None else None,
            "overlay": str(overlay_path) if overlay_saved else None,
        },
    }


def run_llm_once(target: str, captures: dict, target_color: str, timeout_s: float, run_dir: Path):
    messages = build_messages(target, captures, target_color)
    payload = build_chat_payload(messages)
    write_json(run_dir / "llm_request.json", payload)
    reply, response_data = chat_completion(payload, timeout_s)
    write_json(run_dir / "llm_response.json", response_data)
    (run_dir / "llm_reply.txt").write_text(reply, encoding="utf-8")
    parsed = parse_jsonish(reply)
    write_json(run_dir / "llm_reply_parsed.json", parsed)
    return reply, response_data, parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Debug real camera frames, local LLM object identification, color segmentation, and depth pixels without robot motion."
    )
    parser.add_argument("--target", default="the most visually distinct object", help="Object for the LLM to identify.")
    parser.add_argument(
        "--target-color",
        choices=["red", "green", "blue", "yellow", "orange", "white", "black", "none"],
        default="red",
        help="Color mask used for segmentation comparison. Use 'none' to disable segmentation.",
    )
    parser.add_argument(
        "--camera",
        choices=["both", "d435", "d405"],
        default="both",
        help="Which RealSense camera(s) to capture. Default: both.",
    )
    parser.add_argument("--warmup-s", type=float, default=2.0, help="Camera warmup before capture.")
    parser.add_argument("--timeout-s", type=float, default=90.0, help="HTTP timeout for the LLM request.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    camera_names = ["d435", "d405"] if args.camera == "both" else [args.camera]
    run_dir = create_run_dir(args.output_dir)
    cameras = {}

    print("--- Live LLM / Camera / Segmentation Debug ---")
    print("Robot connection: disabled. Robot motion: disabled.")
    print(f"LLM endpoint: {cfg.LLM_API_URL.rstrip('/')}/chat/completions")
    print(f"Model: {cfg.QWEN_MODEL_PATH}")
    print(f"Run folder: {run_dir}")

    try:
        for camera_name in camera_names:
            cameras[camera_name] = start_camera(camera_name)

        print(f"Warming cameras for {args.warmup_s:.1f}s...")
        time.sleep(max(0.0, args.warmup_s))

        print("Capturing live RGB/depth frame(s)...")
        captures = capture_live_frames(cameras, camera_names)
        save_capture_artifacts(run_dir, captures)

        print("Sending captured frame(s) to the configured LLM server...")
        _, _, parsed_reply = run_llm_once(args.target, captures, args.target_color, args.timeout_s, run_dir)

        analyses = {}
        for camera_name, capture in captures.items():
            segmentation = segment_image(capture["rgb"], args.target_color)
            llm_result = llm_camera_result(parsed_reply, camera_name)
            analyses[camera_name] = camera_analysis(
                camera_name,
                capture,
                llm_result,
                segmentation,
                run_dir,
            )

        summary = {
            "run_dir": str(run_dir),
            "robot_connection": "disabled",
            "robot_motion": "disabled",
            "target": args.target,
            "target_color": args.target_color,
            "endpoint": cfg.LLM_API_URL.rstrip("/") + "/chat/completions",
            "model": cfg.QWEN_MODEL_PATH,
            "camera_count": len(captures),
            "llm_reply_parsed": parsed_reply,
            "cameras": analyses,
        }
        write_json(run_dir / "summary.json", summary)

        print("\nResults:")
        for camera_name, analysis in analyses.items():
            llm_pixel = analysis["llm"].get("center_pixel")
            seg_pixel = analysis["segmentation"].get("pixel")
            print(
                f"{camera_name.upper()}: llm={llm_pixel} seg={seg_pixel} "
                f"delta_px={analysis['llm_to_segmentation_delta_px']} "
                f"depth_m={analysis['llm_depth'].get('center_depth_m')} "
                f"xyz_camera={analysis['llm_xyz_camera'].get('xyz_camera_m')}"
            )

        print(f"\nSaved summary: {run_dir / 'summary.json'}")
        return 0

    except Exception as exc:
        error_summary = {
            "run_dir": str(run_dir),
            "robot_connection": "disabled",
            "robot_motion": "disabled",
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_json(run_dir / "summary.json", error_summary)
        print(f"FAIL: {exc}")
        print(f"Saved failure summary: {run_dir / 'summary.json'}")
        return 1
    finally:
        for camera_name, camera in cameras.items():
            stop_camera(camera, camera_name)


if __name__ == "__main__":
    raise SystemExit(main())
