###########################
# Test 24: Live ChArUco board detection.
#
# This opens a live RealSense color view, detects the ChArUco board configured
# in Working/config.py, and overlays recognized marker/corner points.
#
# Usage:
#   python3 Module_Tests/Working_Tests/charuco_detection_video_test24.py
#   python3 Module_Tests/Working_Tests/charuco_detection_video_test24.py --camera D405
#
# Keys in the video window:
#   s: save the current annotated frame to Module_Tests/Test_Outputs
#   q or ESC: close the view
###########################

from pathlib import Path
import argparse
import time

from _working_test_utils import TEST_OUTPUTS_DIR, add_working_to_path, unique_output_path

add_working_to_path()

import numpy as np

import config as cfg
from camera_calibration.charuco_utils import (
    create_charuco_board,
    detect_charuco_corners,
    draw_charuco_detection,
    get_aruco_dictionary,
    require_charuco_support,
)
from hardware.camera import RealSense


SCRIPT_NAME = Path(__file__).stem
OUTPUT_DIR = TEST_OUTPUTS_DIR / SCRIPT_NAME
QUIT_KEYS = {ord("q"), ord("Q"), 27}
SNAPSHOT_KEYS = {ord("s"), ord("S")}
cv2 = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show live ChArUco board recognition with annotated points."
    )
    parser.add_argument(
        "--camera",
        default="D435",
        help="Configured RealSense camera to use. Default: D435",
    )
    parser.add_argument(
        "--min-markers",
        type=int,
        default=4,
        help="Minimum ArUco markers required before interpolation. Default: 4",
    )
    parser.add_argument(
        "--min-corners",
        type=int,
        default=4,
        help="Minimum ChArUco corners required for recognized status. Default: 4",
    )
    args = parser.parse_args()
    args.camera = args.camera.upper()
    if args.camera not in {"D435", "D405"}:
        parser.error("--camera must be D435 or D405")
    if args.min_markers < 1:
        parser.error("--min-markers must be at least 1")
    if args.min_corners < 1:
        parser.error("--min-corners must be at least 1")
    return args


def is_configured_serial(serial_number):
    return bool(serial_number) and not str(serial_number).startswith("YOUR_")


def camera_config(camera_name):
    if camera_name == "D435":
        return cfg.D435_SERIAL, cfg.D435_RESOLUTION
    return cfg.D405_SERIAL, cfg.D405_RESOLUTION


def create_detector_params():
    if hasattr(cv2.aruco, "DetectorParameters"):
        return cv2.aruco.DetectorParameters()
    return cv2.aruco.DetectorParameters_create()


def make_board_model(label, board_corners, aruco_dict, legacy_pattern=False):
    board = create_charuco_board(
        cv2,
        board_corners,
        cfg.INTRINSIC_SQUARE_SIZE,
        cfg.INTRINSIC_MARKER_SIZE,
        aruco_dict,
        legacy_pattern=legacy_pattern,
    )

    return {
        "label": label,
        "corners": board_corners,
        "legacy_pattern": legacy_pattern,
        "board": board,
        "successes": 0,
        "max_markers": 0,
        "max_charuco": 0,
    }


def make_board_models(aruco_dict):
    """
    Build board candidates that diagnose common ChArUco mismatch cases.

    Seeing every ArUco marker but zero ChArUco corners usually means the marker
    dictionary is fine, while the board model disagrees with the printed target.
    """
    configured = tuple(cfg.INTRINSIC_BOARD_CORNERS)
    swapped = (configured[1], configured[0])
    corner_options = [("configured", configured)]
    if swapped != configured:
        corner_options.append(("swapped", swapped))

    models = []
    for source, board_corners in corner_options:
        standard_label = f"{source} {board_corners[0]}x{board_corners[1]} standard"
        models.append(make_board_model(standard_label, board_corners, aruco_dict))

        legacy_label = f"{source} {board_corners[0]}x{board_corners[1]} legacy"
        legacy_model = make_board_model(legacy_label, board_corners, aruco_dict, legacy_pattern=True)
        if legacy_model is not None:
            models.append(legacy_model)

    return [model for model in models if model is not None]


def detect_best_board_model(gray, aruco_dict, board_models, detector_params, min_markers, min_corners):
    best_detection = None
    for model in board_models:
        detection = detect_charuco_corners(
            cv2,
            gray,
            aruco_dict,
            model["board"],
            detector_params=detector_params,
            min_markers=min_markers,
            min_corners=min_corners,
        )
        detection["board_model"] = model

        model["max_markers"] = max(model["max_markers"], detection.get("marker_count", 0))
        model["max_charuco"] = max(model["max_charuco"], detection.get("charuco_count", 0))
        if detection["success"]:
            model["successes"] += 1

        if best_detection is None:
            best_detection = detection
            continue

        best_score = (
            int(best_detection["success"]),
            best_detection.get("charuco_count", 0),
            best_detection.get("marker_count", 0),
        )
        candidate_score = (
            int(detection["success"]),
            detection.get("charuco_count", 0),
            detection.get("marker_count", 0),
        )
        if candidate_score > best_score:
            best_detection = detection

    return best_detection


def print_board_model_summary(board_models):
    print("\nBoard model diagnosis:")
    for model in board_models:
        print(
            "  "
            f"{model['label']}: "
            f"success_frames={model['successes']}, "
            f"max_markers={model['max_markers']}, "
            f"max_charuco_corners={model['max_charuco']}"
        )

    successful = [model for model in board_models if model["successes"] > 0]
    if not successful:
        print("  No tested board model produced ChArUco corners.")
        return

    best = max(successful, key=lambda model: model["successes"])
    print(f"  Best model: {best['label']}")
    if "swapped" in best["label"]:
        print(f"  ACTION: set INTRINSIC_BOARD_CORNERS to {best['corners']} in Working/config.py.")
    if best["legacy_pattern"]:
        print("  ACTION: set INTRINSIC_CHARUCO_LEGACY_PATTERN to True in Working/config.py.")


def draw_text(image, text, position, color, scale=0.7, thickness=2):
    x, y = position
    cv2.putText(
        image,
        text,
        (x + 1, y + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
    )
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
    )


def draw_charuco_point_labels(image, detection):
    charuco_corners = detection.get("charuco_corners")
    charuco_ids = detection.get("charuco_ids")
    if charuco_corners is None or charuco_ids is None:
        return

    for corner, corner_id in zip(charuco_corners, charuco_ids.flatten()):
        x, y = corner.reshape(2)
        point = (int(round(x)), int(round(y)))
        cv2.circle(image, point, 8, (0, 0, 0), 2)
        cv2.circle(image, point, 5, (0, 255, 255), -1)
        cv2.putText(
            image,
            str(int(corner_id)),
            (point[0] + 7, point[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            image,
            str(int(corner_id)),
            (point[0] + 7, point[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
        )


def render_preview(frame, camera_name, detection, recognized_frames, fps):
    display = frame.copy()
    draw_charuco_detection(cv2, display, detection)
    draw_charuco_point_labels(display, detection)

    marker_count = detection.get("marker_count", 0)
    charuco_count = detection.get("charuco_count", 0)
    if detection["success"]:
        status = f"CHARUCO BOARD RECOGNIZED: {charuco_count} corners"
        color = (0, 255, 0)
    elif marker_count:
        status = f"PARTIAL BOARD: {marker_count} markers, {charuco_count} corners"
        color = (0, 220, 255)
    else:
        status = "SEARCHING FOR CHARUCO BOARD"
        color = (0, 0, 255)

    board_text = (
        f"Board {cfg.INTRINSIC_BOARD_CORNERS[0]}x{cfg.INTRINSIC_BOARD_CORNERS[1]} "
        f"corners | square {cfg.INTRINSIC_SQUARE_SIZE * 1000:.1f}mm | "
        f"marker {cfg.INTRINSIC_MARKER_SIZE * 1000:.1f}mm | "
        f"legacy {cfg.INTRINSIC_CHARUCO_LEGACY_PATTERN}"
    )
    model_text = f"best model: {detection['board_model']['label']}"

    draw_text(display, f"{camera_name} ChArUco detection test", (20, 35), (255, 255, 255), 0.75)
    draw_text(display, status, (20, 72), color, 0.75)
    draw_text(display, board_text, (20, 106), (255, 255, 255), 0.58)
    draw_text(display, model_text, (20, 138), (255, 255, 255), 0.58)
    draw_text(display, f"recognized frames: {recognized_frames} | fps: {fps:.1f}", (20, 170), (255, 255, 255), 0.58)
    draw_text(display, "s: save annotated frame | q/ESC: quit", (20, display.shape[0] - 20), (255, 255, 255), 0.62)
    return display


def save_snapshot(display):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = unique_output_path(OUTPUT_DIR / "charuco_detection.png")
    if not cv2.imwrite(str(path), display):
        raise RuntimeError(f"Could not save annotated frame to {path}")
    return path


def run_detection_view(
    camera,
    camera_name,
    resolution,
    aruco_dict,
    board_models,
    detector_params,
    min_markers,
    min_corners,
):
    window_name = f"{camera_name} ChArUco Detection Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    recognized_frames = 0
    saved_frames = 0
    last_frame_time = time.perf_counter()
    no_frame_started = None
    fps = 0.0

    while True:
        frame, _, _ = camera.get_frames()
        if frame is None:
            if no_frame_started is None:
                no_frame_started = time.perf_counter()
            if time.perf_counter() - no_frame_started > 5.0:
                raise RuntimeError(f"No color frames received from {camera_name}.")

            display = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
            draw_text(display, f"Waiting for {camera_name} color frames...", (20, 40), (0, 0, 255), 0.75)
            draw_text(display, "q/ESC: quit", (20, display.shape[0] - 20), (255, 255, 255), 0.62)
            cv2.imshow(window_name, display)
            key = cv2.waitKey(30) & 0xFF
            if key in QUIT_KEYS:
                break
            continue
        no_frame_started = None

        now = time.perf_counter()
        delta = now - last_frame_time
        if delta > 0:
            fps = 1.0 / delta
        last_frame_time = now

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detection = detect_best_board_model(
            gray,
            aruco_dict,
            board_models,
            detector_params,
            min_markers,
            min_corners,
        )

        if detection["success"]:
            recognized_frames += 1

        display = render_preview(frame, camera_name, detection, recognized_frames, fps)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key in QUIT_KEYS:
            break
        if key in SNAPSHOT_KEYS:
            saved_path = save_snapshot(display)
            saved_frames += 1
            print(f"Saved annotated ChArUco frame: {saved_path}")

    return recognized_frames, saved_frames


def main():
    global cv2

    args = parse_args()
    camera = None

    print("--- Testing Live ChArUco Detection ---")
    print(f"Camera: {args.camera}")
    print(f"Board inner corners: {cfg.INTRINSIC_BOARD_CORNERS}")
    print(f"Square size: {cfg.INTRINSIC_SQUARE_SIZE * 1000:.1f} mm")
    print(f"Marker size: {cfg.INTRINSIC_MARKER_SIZE * 1000:.1f} mm")
    print(f"ArUco dictionary: {cfg.INTRINSIC_ARUCO_DICT_NAME}")

    try:
        try:
            import cv2 as cv2_module
        except ModuleNotFoundError:
            print("FAIL: OpenCV is required. Install/use an environment with opencv-contrib-python.")
            return 1
        cv2 = cv2_module

        require_charuco_support(cv2)
        print(f"OpenCV version: {cv2.__version__}")
        if cfg.INTRINSIC_MARKER_SIZE >= cfg.INTRINSIC_SQUARE_SIZE:
            print("FAIL: INTRINSIC_MARKER_SIZE must be smaller than INTRINSIC_SQUARE_SIZE.")
            return 1

        serial_number, resolution = camera_config(args.camera)
        if not is_configured_serial(serial_number):
            print(f"FAIL: {args.camera} serial number is not configured in Working/config.py")
            return 1

        aruco_dict = get_aruco_dictionary(cv2, cfg.INTRINSIC_ARUCO_DICT_NAME)
        board_models = make_board_models(aruco_dict)
        detector_params = create_detector_params()
        print("Trying board models:")
        for model in board_models:
            print(f"  - {model['label']}")

        print(f"Starting {args.camera} ({serial_number}) at {resolution}...")
        camera = RealSense(serial_number=serial_number, resolution=resolution, fps=cfg.CAMERA_FPS)
        print(f"Warming up for {cfg.CAMERA_WARMUP_SECONDS:.1f} seconds...")
        time.sleep(cfg.CAMERA_WARMUP_SECONDS)
        print("Show the printed ChArUco board to the camera. Press q or ESC to close.")

        recognized_frames, saved_frames = run_detection_view(
            camera,
            args.camera,
            resolution,
            aruco_dict,
            board_models,
            detector_params,
            args.min_markers,
            args.min_corners,
        )

        print_board_model_summary(board_models)
        print(f"Recognized frames: {recognized_frames}")
        print(f"Annotated frames saved: {saved_frames}")
        if recognized_frames == 0:
            print("FAIL: ChArUco board was never recognized before the view closed.")
            return 1

        print("PASS: ChArUco board was recognized and annotated in the video view.")
        return 0
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1
    finally:
        if cv2 is not None:
            cv2.destroyAllWindows()
        if camera is not None:
            camera.stop()


if __name__ == "__main__":
    raise SystemExit(main())
