"""
Test 29: Local LLM image-input capability check.

This is an integration test for the local OpenAI-compatible LLM server. It does
not require RealSense cameras or robot hardware. Instead, it creates simple
synthetic images in memory and sends them to the same chat/completions endpoint
that main.py uses.

WHY THIS TEST EXISTS:
The robot runtime currently depends on the LLM to visually inspect camera
images and choose object pixels. If the local server ignores `image_url` content,
uses a text-only model, or has a mismatched multimodal projector, the model will
sound confused and say things like "no specific object was provided." This test
separates "camera/tool bug" from "LLM cannot actually see images."

WHAT PASSING MEANS:
- The server accepted an OpenAI-style `image_url` data URL.
- The loaded model/projector could identify a simple red square.
- The two-image check can tell whether one request can contain multiple images.

WHAT FAILING MEANS:
- If both image checks fail, the local model/server probably is not receiving
  or using images.
- If the single-image check passes but the two-image check fails, the model can
  see images, but multiple images in one request may be unsupported or unreliable.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
import struct
import urllib.error
import urllib.request
import zlib

from _working_test_utils import TEST_OUTPUTS_DIR, add_working_to_path


add_working_to_path()

import config as cfg


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """
    Build one PNG chunk using only the Python standard library.

    WHY NOT USE PILLOW HERE:
    The real image encoder uses Pillow/OpenCV, but this test is specifically
    about LLM image intake. Using a tiny standard-library PNG writer keeps the
    test focused on the LLM server and avoids failing just because a local image
    package is missing.
    """
    checksum = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return (
        struct.pack("!I", len(data))
        + chunk_type
        + data
        + struct.pack("!I", checksum)
    )


def square_bounds(width, height):
    """Return the rectangle used for the synthetic square in every image format."""
    margin_x = width // 4
    margin_y = height // 5
    return margin_x, margin_y, width - margin_x, height - margin_y


def make_colored_square_png(square_rgb, width=192, height=144) -> bytes:
    """
    Create a simple PNG: white background with one large colored square.

    The image is intentionally boring. A working vision-language model should
    not need scene context, OCR, or object priors to answer this correctly. If it
    cannot identify this square, it will not reliably identify a real red block
    from the robot cameras.
    """
    square_rgb = tuple(int(value) for value in square_rgb)
    square_left, square_top, square_right, square_bottom = square_bounds(width, height)

    rows = []
    for y in range(height):
        row = bytearray()
        # PNG filter byte 0 means this row is stored without prediction/filtering.
        # That makes the image easy for us to write by hand.
        row.append(0)
        for x in range(width):
            in_square = square_left <= x < square_right and square_top <= y < square_bottom
            if in_square:
                row.extend(square_rgb)
            else:
                row.extend((255, 255, 255))
        rows.append(bytes(row))

    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    compressed_pixels = zlib.compress(b"".join(rows), level=9)
    return (
        PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", compressed_pixels)
        + _png_chunk(b"IEND", b"")
    )


def make_colored_square_jpeg(square_rgb, width=192, height=144):
    """
    Create a JPEG version of the test image when Pillow is available.

    WHY JPEG FIRST:
    The robot runtime sends camera frames as JPEG data URLs through
    `encode_image_for_llm()`. Using JPEG here makes the diagnostic match the
    production path more closely. PNG remains available as a standard-library
    fallback so the test can still run on minimal environments.
    """
    try:
        from PIL import Image, ImageDraw
    except ModuleNotFoundError:
        return None

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle(square_bounds(width, height), fill=tuple(int(value) for value in square_rgb))

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def make_colored_square_input(square_rgb, image_format="auto"):
    """
    Create one synthetic image input and describe its transport metadata.

    `image_format="auto"` prefers JPEG to mirror the runtime camera path. If
    Pillow is unavailable, it falls back to PNG so the test can still diagnose
    whether the LLM server handles image_url data at all.
    """
    if image_format not in {"auto", "jpeg", "png"}:
        raise ValueError("image_format must be one of: auto, jpeg, png")

    if image_format in {"auto", "jpeg"}:
        jpeg_bytes = make_colored_square_jpeg(square_rgb)
        if jpeg_bytes is not None:
            return {
                "bytes": jpeg_bytes,
                "mime_type": "image/jpeg",
                "extension": "jpg",
            }
        if image_format == "jpeg":
            raise RuntimeError("Pillow is required for --image-format jpeg")

    png_bytes = make_colored_square_png(square_rgb)
    return {
        "bytes": png_bytes,
        "mime_type": "image/png",
        "extension": "png",
    }


def image_url_content(image_input: dict) -> dict:
    """Wrap image bytes in the OpenAI-style image_url object used by the runtime."""
    encoded = base64.b64encode(image_input["bytes"]).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{image_input['mime_type']};base64,{encoded}",
        },
    }


def build_chat_payload(messages):
    """
    Build the exact OpenAI-compatible JSON request body.

    Keeping request construction in a helper matters for debugging: the same
    payload object is both sent to the server and saved to disk. If the model
    fails to identify the square, the saved JSON lets us answer "did we actually
    send an image_url?" without guessing from console output.
    """
    return {
        "model": cfg.QWEN_MODEL_PATH,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 128,
    }


def chat_completion(payload, timeout_s):
    """
    Send a chat completion request using standard-library HTTP.

    WHY NOT IMPORT THE OPENAI SDK:
    The robot runtime uses the SDK, but this diagnostic should be runnable even
    in environments where the SDK is not installed yet. The local server still
    receives the same OpenAI-compatible JSON shape.
    """
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


def normalize(text):
    return " ".join(str(text).upper().replace("_", " ").replace("=", " ").split())


def contains_in_order(text, expected_words):
    """Return true when all expected words appear in response order."""
    normalized = normalize(text)
    search_start = 0
    for word in expected_words:
        index = normalized.find(word.upper(), search_start)
        if index < 0:
            return False
        search_start = index + len(word)
    return True


def run_single_image_check(red_image, timeout_s):
    """
    Check whether the model can identify one image at all.

    The prompt does not reveal the answer. A text-only model usually responds
    that it cannot see an image, guesses, or ignores the visual question.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a strict visual diagnostic. Answer only from image evidence.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Look at this image. What color is the large filled square? "
                        "Respond with exactly one word: RED, GREEN, BLUE, or UNKNOWN."
                    ),
                },
                image_url_content(red_image),
            ],
        },
    ]
    payload = build_chat_payload(messages)
    reply, response_data = chat_completion(payload, timeout_s)
    passed = "RED" in normalize(reply) and "UNKNOWN" not in normalize(reply)
    return {
        "name": "single_image",
        "passed": passed,
        "reply": reply,
        "request_payload": payload,
        "response_data": response_data,
    }


def run_two_image_check(red_image, green_image, timeout_s):
    """
    Check whether the server/model can handle two image_url items in one message.

    This is slightly different from the current robot runtime: main.py normally
    lets the model request D435 and D405 images as separate tool calls. This test
    answers the user's practical question: can the local LLM receive both camera
    images in a single request if we later change the runtime to do that?
    """
    messages = [
        {
            "role": "system",
            "content": "You are a strict visual diagnostic. Answer only from image evidence.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You will receive two images. Each image has one large filled square. "
                        "Report the color in image 1 and image 2. Respond exactly like: "
                        "FIRST=RED; SECOND=GREEN. Use RED, GREEN, BLUE, or UNKNOWN."
                    ),
                },
                image_url_content(red_image),
                image_url_content(green_image),
            ],
        },
    ]
    payload = build_chat_payload(messages)
    reply, response_data = chat_completion(payload, timeout_s)
    passed = contains_in_order(reply, ["RED", "GREEN"]) and "UNKNOWN" not in normalize(reply)
    return {
        "name": "two_image",
        "passed": passed,
        "reply": reply,
        "request_payload": payload,
        "response_data": response_data,
    }


def create_run_folder(output_root: Path) -> dict:
    """
    Create one self-contained output folder for this diagnostic run.

    WHY A NEW FOLDER EVERY TIME:
    LLM server behavior can change when you swap models, restart llama-server,
    or change mmproj files. Timestamped folders preserve each run's exact inputs
    and outputs so we can compare failures instead of overwriting evidence.
    """
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


def save_debug_images(red_image, green_image, input_dir: Path):
    """
    Save the synthetic image inputs so the operator can inspect exactly what was sent.

    These are not camera images; they are controlled test inputs. If the LLM
    cannot read them, the failure is almost certainly in the multimodal server
    setup rather than RealSense capture quality.
    """
    red_path = input_dir / f"red_square_input.{red_image['extension']}"
    green_path = input_dir / f"green_square_input.{green_image['extension']}"
    red_path.write_bytes(red_image["bytes"])
    green_path.write_bytes(green_image["bytes"])
    return red_path, green_path


def write_json(path: Path, data):
    """Write readable JSON artifacts for requests, responses, and summary files."""
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_check_artifacts(check_result, folders):
    """
    Save one check's request and response artifacts.

    The request JSON includes the base64 data URL. That is intentional: it makes
    the saved file a faithful copy of what the LLM server received. The PNG files
    in `inputs/` are easier for humans to inspect, while `requests/` proves the
    transport format.
    """
    name = check_result["name"]
    write_json(folders["requests"] / f"{name}_request.json", check_result["request_payload"])
    write_json(folders["outputs"] / f"{name}_response.json", check_result["response_data"])
    (folders["outputs"] / f"{name}_reply.txt").write_text(
        check_result["reply"],
        encoding="utf-8",
    )


def save_summary(folders, single_result=None, two_result=None, error=None):
    """
    Save a compact summary that explains the result without opening every artifact.

    This file is the first place to look after a failure. If the input images are
    correct but the replies are wrong, the issue is almost certainly the model,
    projector, or server image-input support, not the synthetic test image.
    """
    summary = {
        "endpoint": cfg.LLM_API_URL.rstrip("/") + "/chat/completions",
        "model": cfg.QWEN_MODEL_PATH,
        "run_folder": str(folders["run"]),
        "error": str(error) if error else None,
        "single_image": None,
        "two_image": None,
        "interpretation": None,
    }

    if single_result is not None:
        summary["single_image"] = {
            "passed": single_result["passed"],
            "reply": single_result["reply"],
        }
    if two_result is not None:
        summary["two_image"] = {
            "passed": two_result["passed"],
            "reply": two_result["reply"],
        }

    if error:
        summary["interpretation"] = (
            "The request did not complete. Check whether the local LLM server is "
            "running at LLM_API_URL and whether it accepts OpenAI-compatible chat requests."
        )
    elif single_result and two_result and single_result["passed"] and two_result["passed"]:
        summary["interpretation"] = (
            "The local LLM can receive and reason over image_url inputs, including two images."
        )
    elif single_result and single_result["passed"]:
        summary["interpretation"] = (
            "The local LLM can see one image, but two images in one request failed or were unreliable."
        )
    else:
        summary["interpretation"] = (
            "The local LLM did not reliably identify a simple red square. Check that llama-server "
            "was started with a compatible vision model and mmproj, and that the server supports "
            "OpenAI-style image_url content."
        )

    write_json(folders["outputs"] / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=60.0,
        help="HTTP timeout per LLM request. Default: 60 seconds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TEST_OUTPUTS_DIR / "llm_image_capability",
        help=(
            "Root directory for per-run input/output folders. "
            "Default: Module_Tests/Test_Outputs/llm_image_capability."
        ),
    )
    parser.add_argument(
        "--image-format",
        choices=["auto", "jpeg", "png"],
        default="auto",
        help=(
            "Synthetic image format. 'auto' uses JPEG when Pillow is available "
            "and falls back to PNG. Default: auto."
        ),
    )
    args = parser.parse_args()

    folders = create_run_folder(args.output_dir)
    red_image = make_colored_square_input((220, 0, 0), args.image_format)
    green_image = make_colored_square_input((0, 170, 0), args.image_format)
    red_path, green_path = save_debug_images(red_image, green_image, folders["inputs"])

    print("--- LLM Image Capability Test ---")
    print(f"Endpoint: {cfg.LLM_API_URL.rstrip('/')}/chat/completions")
    print(f"Model: {cfg.QWEN_MODEL_PATH}")
    print(f"Image MIME type: {red_image['mime_type']}")
    print(f"Run folder: {folders['run']}")
    print(f"Saved red test image: {red_path}")
    print(f"Saved green test image: {green_path}")

    try:
        single_result = run_single_image_check(red_image, args.timeout_s)
        save_check_artifacts(single_result, folders)

        two_result = run_two_image_check(red_image, green_image, args.timeout_s)
        save_check_artifacts(two_result, folders)
    except Exception as exc:
        summary = save_summary(folders, error=exc)
        print(f"FAIL: could not complete image capability test: {exc}")
        print(f"Saved failure summary: {folders['outputs'] / 'summary.json'}")
        return 1

    summary = save_summary(folders, single_result, two_result)
    single_passed = single_result["passed"]
    two_passed = two_result["passed"]
    single_reply = single_result["reply"]
    two_reply = two_result["reply"]

    print("\nSingle-image response:")
    print(single_reply)
    print(f"Single-image check: {'PASS' if single_passed else 'FAIL'}")

    print("\nTwo-image response:")
    print(two_reply)
    print(f"Two-image check: {'PASS' if two_passed else 'FAIL'}")
    print(f"\nSaved request/response artifacts: {folders['run']}")
    print(f"Saved summary: {folders['outputs'] / 'summary.json'}")

    print("\nInterpretation:")
    print(summary["interpretation"])
    if single_passed and two_passed:
        return 0
    if single_passed:
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
