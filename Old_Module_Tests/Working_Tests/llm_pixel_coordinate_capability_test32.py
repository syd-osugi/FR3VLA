"""
Test 32: LLM Pixel Coordinate Capability
----------------------------------------
Checks whether the configured local multimodal LLM can look at image inputs and
return pixel coordinates for a prompted target through a tool call.

This test does not need RealSense cameras or robot hardware. It creates simple
synthetic images with known target centers, sends them to the same
OpenAI-compatible `/chat/completions` endpoint used by the runtime, and verifies
that the model calls a coordinate-reporting tool with pixels close to ground
truth.

Why this is separate from llm_pixel_annotation_test30.py:
  - Test 32 answers: can the LLM see an image and produce usable pixel coords?
  - Test 30 answers: does the live camera/tool runtime produce pixel coords for
    a real operator prompt?
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
from io import BytesIO
import json
import math
from pathlib import Path
import struct
import urllib.error
import urllib.request
import zlib

from _working_test_utils import TEST_OUTPUTS_DIR, add_working_to_path


add_working_to_path()

import config as cfg


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return (
        struct.pack("!I", len(data))
        + chunk_type
        + data
        + struct.pack("!I", checksum)
    )


def make_png(width: int, height: int, objects: list[dict]) -> bytes:
    """Create an RGB PNG with filled rectangles using only the stdlib."""
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            rgb = (255, 255, 255)
            for obj in objects:
                left, top, right, bottom = obj["bbox"]
                if left <= x < right and top <= y < bottom:
                    rgb = obj["rgb"]
                    break
            row.extend(rgb)
        rows.append(bytes(row))

    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(b"".join(rows), level=9))
        + _png_chunk(b"IEND", b"")
    )


def make_jpeg(width: int, height: int, objects: list[dict]) -> bytes | None:
    """Create a JPEG when Pillow is available, matching the runtime image path."""
    try:
        from PIL import Image, ImageDraw
    except ModuleNotFoundError:
        return None

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for obj in objects:
        left, top, right, bottom = obj["bbox"]
        draw.rectangle((left, top, right - 1, bottom - 1), fill=obj["rgb"])

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def rect_center(bbox):
    left, top, right, bottom = bbox
    return [int(round((left + right - 1) / 2)), int(round((top + bottom - 1) / 2))]


def make_image_input(name: str, objects: list[dict], image_format: str):
    """Return image bytes plus metadata and expected centers."""
    if image_format not in {"auto", "jpeg", "png"}:
        raise ValueError("image_format must be auto, jpeg, or png")

    image_bytes = None
    mime_type = "image/png"
    extension = "png"

    if image_format in {"auto", "jpeg"}:
        image_bytes = make_jpeg(IMAGE_WIDTH, IMAGE_HEIGHT, objects)
        if image_bytes is not None:
            mime_type = "image/jpeg"
            extension = "jpg"
        elif image_format == "jpeg":
            raise RuntimeError("Pillow is required for --image-format jpeg")

    if image_bytes is None:
        image_bytes = make_png(IMAGE_WIDTH, IMAGE_HEIGHT, objects)

    expected_centers = {
        obj["label"]: rect_center(obj["bbox"])
        for obj in objects
        if obj.get("label")
    }
    return {
        "name": name,
        "bytes": image_bytes,
        "mime_type": mime_type,
        "extension": extension,
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
        "expected_centers": expected_centers,
    }


def image_url_content(image_input: dict) -> dict:
    encoded = base64.b64encode(image_input["bytes"]).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{image_input['mime_type']};base64,{encoded}",
        },
    }


def coordinate_tool_schema():
    return [
        {
            "type": "function",
            "function": {
                "name": "report_pixel_coordinates",
                "description": (
                    "Report the center pixel of the prompted target after inspecting "
                    "the provided image or images."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "camera": {
                            "type": "string",
                            "enum": ["d435", "d405"],
                            "description": "Camera/image where the target is visible.",
                        },
                        "target": {
                            "type": "string",
                            "description": "Short name of the target object.",
                        },
                        "coords": {
                            "type": "array",
                            "description": "Target center pixel as [u, v], origin at top-left.",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                    },
                    "required": ["camera", "target", "coords"],
                },
            },
        }
    ]


def build_payload(messages):
    return {
        "model": cfg.QWEN_MODEL_PATH,
        "messages": messages,
        "tools": coordinate_tool_schema(),
        "tool_choice": "auto",
        "temperature": 0.0,
        "max_tokens": 256,
    }


def chat_completion(payload, timeout_s):
    endpoint = cfg.LLM_API_URL.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
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

    return json.loads(response_body)


def parse_tool_args(response_data):
    """Extract report_pixel_coordinates arguments from OpenAI-style responses."""
    message = response_data["choices"][0]["message"]
    tool_calls = message.get("tool_calls") or []
    for call in tool_calls:
        function = call.get("function") or {}
        if function.get("name") != "report_pixel_coordinates":
            continue
        raw_args = function.get("arguments") or "{}"
        if isinstance(raw_args, dict):
            return raw_args
        return json.loads(raw_args)

    function_call = message.get("function_call")
    if isinstance(function_call, dict) and function_call.get("name") == "report_pixel_coordinates":
        raw_args = function_call.get("arguments") or "{}"
        if isinstance(raw_args, dict):
            return raw_args
        return json.loads(raw_args)

    return None


def evaluate_tool_args(tool_args, expected_camera, expected_pixel, tolerance_px):
    failures = []
    if not isinstance(tool_args, dict):
        return {
            "passed": False,
            "failures": ["No report_pixel_coordinates tool call was returned."],
            "tool_args": tool_args,
            "distance_px": None,
        }

    if tool_args.get("camera") != expected_camera:
        failures.append(
            f"Expected camera {expected_camera!r}, got {tool_args.get('camera')!r}."
        )

    coords = tool_args.get("coords")
    if not isinstance(coords, list) or len(coords) != 2:
        failures.append(f"coords must be [u, v], got {coords!r}.")
        distance_px = None
    elif not all(isinstance(value, int) and not isinstance(value, bool) for value in coords):
        failures.append(f"coords must contain integer pixels, got {coords!r}.")
        distance_px = None
    else:
        distance_px = float(math.dist(coords, expected_pixel))
        if distance_px > tolerance_px:
            failures.append(
                f"Pixel {coords!r} is {distance_px:.1f}px from expected {expected_pixel!r}; "
                f"limit is {tolerance_px:.1f}px."
            )

    return {
        "passed": not failures,
        "failures": failures,
        "tool_args": tool_args,
        "expected_camera": expected_camera,
        "expected_pixel": expected_pixel,
        "distance_px": distance_px,
        "tolerance_px": tolerance_px,
    }


def system_message():
    return {
        "role": "system",
        "content": (
            "You are a strict visual pixel-coordinate diagnostic. Inspect the image "
            "evidence and call report_pixel_coordinates exactly once. Use [u, v] "
            "pixel coordinates with origin at the top-left. Do not answer in prose."
        ),
    }


def run_single_image_check(red_image, timeout_s, tolerance_px):
    expected_pixel = red_image["expected_centers"]["red_square"]
    messages = [
        system_message(),
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"D435 overhead image, {IMAGE_WIDTH}x{IMAGE_HEIGHT}. "
                        "Locate the center pixel of the red square."
                    ),
                },
                image_url_content(red_image),
            ],
        },
    ]
    payload = build_payload(messages)
    response_data = chat_completion(payload, timeout_s)
    tool_args = parse_tool_args(response_data)
    evaluation = evaluate_tool_args(
        tool_args,
        expected_camera="d435",
        expected_pixel=expected_pixel,
        tolerance_px=tolerance_px,
    )
    return {
        "name": "single_image_pixel_tool",
        "request_payload": payload,
        "response_data": response_data,
        "evaluation": evaluation,
    }


def run_separate_camera_check(d435_image, d405_image, timeout_s, tolerance_px):
    expected_pixel = d405_image["expected_centers"]["green_square"]
    messages = [
        system_message(),
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"D435 overhead image, {IMAGE_WIDTH}x{IMAGE_HEIGHT}. "
                        "Remember this image label as d435."
                    ),
                },
                image_url_content(d435_image),
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"D405 wrist image, {IMAGE_WIDTH}x{IMAGE_HEIGHT}. "
                        "Remember this image label as d405."
                    ),
                },
                image_url_content(d405_image),
            ],
        },
        {
            "role": "user",
            "content": (
                "User prompt: locate the green square. Call report_pixel_coordinates "
                "with the camera label where that target appears and its center pixel."
            ),
        },
    ]
    payload = build_payload(messages)
    response_data = chat_completion(payload, timeout_s)
    tool_args = parse_tool_args(response_data)
    evaluation = evaluate_tool_args(
        tool_args,
        expected_camera="d405",
        expected_pixel=expected_pixel,
        tolerance_px=tolerance_px,
    )
    return {
        "name": "separate_camera_messages_pixel_tool",
        "request_payload": payload,
        "response_data": response_data,
        "evaluation": evaluation,
    }


def create_run_folder(output_root: Path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{timestamp}"
    folders = {
        "run": run_dir,
        "inputs": run_dir / "inputs",
        "requests": run_dir / "requests",
        "outputs": run_dir / "outputs",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_image(image_input, folder: Path):
    path = folder / f"{image_input['name']}.{image_input['extension']}"
    path.write_bytes(image_input["bytes"])
    return path


def save_check_artifacts(result, folders):
    name = result["name"]
    write_json(folders["requests"] / f"{name}_request.json", result["request_payload"])
    write_json(folders["outputs"] / f"{name}_response.json", result["response_data"])
    write_json(folders["outputs"] / f"{name}_evaluation.json", result["evaluation"])


def save_summary(folders, results=None, error=None):
    results = results or []
    summary = {
        "endpoint": cfg.LLM_API_URL.rstrip("/") + "/chat/completions",
        "model": cfg.QWEN_MODEL_PATH,
        "run_folder": str(folders["run"]),
        "error": str(error) if error else None,
        "checks": [
            {
                "name": result["name"],
                "passed": result["evaluation"]["passed"],
                "failures": result["evaluation"]["failures"],
                "tool_args": result["evaluation"]["tool_args"],
                "expected_camera": result["evaluation"].get("expected_camera"),
                "expected_pixel": result["evaluation"].get("expected_pixel"),
                "distance_px": result["evaluation"].get("distance_px"),
            }
            for result in results
        ],
    }
    summary["passed"] = bool(results) and all(
        result["evaluation"]["passed"] for result in results
    )
    if error:
        summary["interpretation"] = (
            "The request did not complete. Check the local LLM server, model path, "
            "vision projector, and OpenAI-compatible image/tool support."
        )
    elif summary["passed"]:
        summary["interpretation"] = (
            "The LLM saw the synthetic image input and produced usable pixel tool arguments."
        )
    else:
        summary["interpretation"] = (
            "The LLM did not reliably produce in-range pixel coordinates through tool calls. "
            "Check multimodal model setup and function-calling behavior."
        )
    write_json(folders["outputs"] / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether the LLM can output prompted target pixel coordinates."
    )
    parser.add_argument(
        "--mode",
        choices=["single", "separate", "both"],
        default="both",
        help="Which coordinate capability check to run. Default: both.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=60.0,
        help="HTTP timeout per LLM request. Default: 60 seconds.",
    )
    parser.add_argument(
        "--pixel-tolerance",
        type=float,
        default=60.0,
        help="Allowed distance from known target center in pixels. Default: 60.",
    )
    parser.add_argument(
        "--image-format",
        choices=["auto", "jpeg", "png"],
        default="auto",
        help="Synthetic image format. 'auto' uses JPEG if Pillow is available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TEST_OUTPUTS_DIR / "llm_pixel_coordinate_capability",
        help="Root directory for per-run artifacts.",
    )
    args = parser.parse_args()

    folders = create_run_folder(args.output_dir)

    red_square = {
        "label": "red_square",
        "rgb": (220, 0, 0),
        "bbox": (250, 170, 390, 310),
    }
    d435_red_square = {
        "label": "red_square",
        "rgb": (220, 0, 0),
        "bbox": (80, 90, 220, 230),
    }
    d405_green_square = {
        "label": "green_square",
        "rgb": (0, 170, 0),
        "bbox": (410, 260, 550, 400),
    }

    red_image = make_image_input("single_red_square", [red_square], args.image_format)
    d435_image = make_image_input("d435_red_square", [d435_red_square], args.image_format)
    d405_image = make_image_input("d405_green_square", [d405_green_square], args.image_format)

    input_paths = [
        save_image(red_image, folders["inputs"]),
        save_image(d435_image, folders["inputs"]),
        save_image(d405_image, folders["inputs"]),
    ]

    print("--- LLM Pixel Coordinate Capability Test ---")
    print(f"Endpoint: {cfg.LLM_API_URL.rstrip('/')}/chat/completions")
    print(f"Model: {cfg.QWEN_MODEL_PATH}")
    print(f"Run folder: {folders['run']}")
    for path in input_paths:
        print(f"Saved input image: {path}")

    results = []
    try:
        if args.mode in {"single", "both"}:
            result = run_single_image_check(red_image, args.timeout_s, args.pixel_tolerance)
            save_check_artifacts(result, folders)
            results.append(result)

        if args.mode in {"separate", "both"}:
            result = run_separate_camera_check(
                d435_image,
                d405_image,
                args.timeout_s,
                args.pixel_tolerance,
            )
            save_check_artifacts(result, folders)
            results.append(result)
    except Exception as exc:
        summary = save_summary(folders, results, error=exc)
        print(f"FAIL: could not complete coordinate capability test: {exc}")
        print(f"Saved summary: {folders['outputs'] / 'summary.json'}")
        return 1

    summary = save_summary(folders, results)
    for result in results:
        evaluation = result["evaluation"]
        print(f"\n{result['name']}: {'PASS' if evaluation['passed'] else 'FAIL'}")
        print(f"  Tool args: {evaluation['tool_args']}")
        print(f"  Expected: camera={evaluation.get('expected_camera')} pixel={evaluation.get('expected_pixel')}")
        if evaluation.get("distance_px") is not None:
            print(f"  Pixel error: {evaluation['distance_px']:.1f}px")
        for failure in evaluation["failures"]:
            print(f"  - {failure}")

    print(f"\nSaved request/response artifacts: {folders['run']}")
    print(summary["interpretation"])
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
