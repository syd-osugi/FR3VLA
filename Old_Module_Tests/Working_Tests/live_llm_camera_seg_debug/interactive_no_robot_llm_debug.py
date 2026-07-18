"""
Interactive No-Robot Main Debug
-------------------------------
Runs as close to Working/main.py as possible without robot access.

This starts the real D435/D405 cameras, uses the real LLMinterface, exposes the
normal camera/localization/fusion/trajectory-planning tools, and removes only
physical robot execution. A static EE pose is supplied for math paths that need
robot_ee_pose, so D405 transforms and trajectory planning can be tested without
connecting to the Franka.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from datetime import datetime

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
WORKING_TESTS_DIR = SCRIPT_DIR.parent
if str(WORKING_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(WORKING_TESTS_DIR))

from _working_test_utils import TEST_OUTPUTS_DIR  # noqa: E402

import config as cfg  # noqa: E402
from hardware.camera import RealSense  # noqa: E402
from robot.trajectory import get_d405_hand_eye_transform, get_d435_transform  # noqa: E402
from vision.camera_viewer import CameraViewer  # noqa: E402
from vision.llm_interface import LLMinterface  # noqa: E402
import vision.tools as tools  # noqa: E402


OUTPUT_DIR = TEST_OUTPUTS_DIR / "interactive_no_robot_llm_debug"
DEFAULT_POSE_PLAN_PATH = Path(cfg.CALIBRATION_DIR) / "d405_hand_eye_pose_plan.json"


def configured_serial(value):
    if not value or str(value).startswith("YOUR_"):
        return None
    return value


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
    raise RuntimeError(f"Could not create unique output folder under {base_dir}")


def stop_camera(camera, name):
    if camera is None:
        return
    try:
        camera.stop()
    except Exception as exc:
        print(f"Warning: failed to stop {name}: {exc}")


def stop_camera_viewer(camera_viewer):
    if camera_viewer is None:
        return
    try:
        camera_viewer.stop()
    except Exception as exc:
        print(f"Warning: failed to stop camera viewer: {exc}")


def identity_pose():
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def validate_pose(matrix, source):
    pose = np.array(matrix, dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{source} must contain a finite 4x4 transform, got shape {pose.shape}")
    return pose.tolist()


def load_pose_json(path: Path, pose_index: int):
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict) and "matrix" in data:
        return validate_pose(data["matrix"], str(path)), f"matrix in {path}"

    if isinstance(data, dict) and "T_ee_to_base" in data:
        return validate_pose(data["T_ee_to_base"], str(path)), f"T_ee_to_base in {path}"

    if isinstance(data, dict) and isinstance(data.get("poses"), list):
        poses = data["poses"]
        if not poses:
            raise ValueError(f"pose plan has no poses: {path}")
        index = max(1, min(int(pose_index), len(poses))) - 1
        pose_record = poses[index]
        return (
            validate_pose(pose_record.get("T_ee_to_base"), f"{path} pose {index + 1}"),
            f"pose {index + 1} from {path}",
        )

    raise ValueError(f"Could not find matrix/T_ee_to_base/poses in {path}")


def choose_static_ee_pose(source: str, pose_file: Path | None, pose_index: int):
    if source == "none":
        return None, "none: D405 robot-frame localization may report robot_ee_pose required"

    if pose_file is not None:
        return load_pose_json(pose_file, pose_index)

    if source in ("auto", "pose-plan") and DEFAULT_POSE_PLAN_PATH.exists():
        return load_pose_json(DEFAULT_POSE_PLAN_PATH, pose_index)

    if source == "pose-plan":
        raise FileNotFoundError(f"pose plan not found: {DEFAULT_POSE_PLAN_PATH}")

    return identity_pose(), "identity fallback pose"


def no_robot_tool_schemas():
    """
    Keep every normal Working tool except physical waypoint execution.

    Planning stays available because it is pure math. Localization/fusion stay
    available because they use cameras, depth, calibration, and the static EE
    pose supplied by this no-robot runner.
    """
    return [
        schema
        for schema in tools.tool_json_list
        if schema.get("function", {}).get("name") != "execute_robot_waypoints"
    ]


def no_robot_system_note(pose_source: str):
    return {
        "role": "system",
        "content": (
            "NO-ROBOT DEBUG MODE is active. Real cameras and normal vision/math "
            "tools are available, including get_xyz_fused and plan_robot_trajectory. "
            "Physical robot execution is unavailable and execute_robot_waypoints was "
            "not provided. Do not claim motion was executed. D405 and trajectory math "
            f"use a static robot_ee_pose from: {pose_source}. Treat planned waypoints "
            "as math/debug output only."
        ),
    }


def sanitized_messages(messages):
    clean = []
    for message in messages:
        copied = copy.deepcopy(message)
        content = copied.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    item["image_url"] = {"url": "<image_url omitted from saved log>"}
        clean.append(copied)
    return clean


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_cv2():
    try:
        import cv2

        return cv2
    except ModuleNotFoundError:
        return None


def parse_json_result(result_text):
    try:
        data = json.loads(result_text)
    except (TypeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def short_text(value, max_chars=96):
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def format_xyz(xyz):
    if not isinstance(xyz, list) or len(xyz) != 3:
        return None
    try:
        return f"{float(xyz[0]):.3f}, {float(xyz[1]):.3f}, {float(xyz[2]):.3f}"
    except (TypeError, ValueError):
        return None


class ToolResultRecorder:
    def __init__(self, viewer_callback=None):
        self.viewer_callback = viewer_callback
        self.records = []

    def reset(self):
        self.records = []

    def handle_tool_result(self, tool_name, tool_args, result_text, user_prompt=None):
        parsed = parse_json_result(result_text)
        self.records.append(
            {
                "tool_name": tool_name,
                "tool_args": copy.deepcopy(tool_args),
                "result_text": result_text,
                "parsed": parsed,
                "user_prompt": user_prompt,
            }
        )
        if self.viewer_callback is not None:
            self.viewer_callback(tool_name, tool_args, result_text, user_prompt)

    def latest_plan(self):
        for record in reversed(self.records):
            if record["tool_name"] == "plan_robot_trajectory" and isinstance(record["parsed"], dict):
                return record["parsed"]
        return None

    def latest_localization_results(self):
        latest = {"d435": [], "d405": []}
        for record in self.records:
            tool_name = record["tool_name"]
            data = record["parsed"]
            if not isinstance(data, dict):
                continue

            if tool_name == "get_xyz_fused":
                for camera_name in ("d435", "d405"):
                    camera_data = data.get(camera_name, {})
                    results = camera_data.get("results") if isinstance(camera_data, dict) else None
                    if isinstance(results, list) and results:
                        latest[camera_name] = results
            elif tool_name in ("get_xyz_d435", "get_xyz_d405"):
                camera_name = data.get("source_camera")
                if camera_name not in ("d435", "d405"):
                    camera_name = "d435" if tool_name == "get_xyz_d435" else "d405"
                results = data.get("points")
                if isinstance(results, list) and results:
                    latest[camera_name] = results

        return latest

    def summary(self):
        return [
            {
                "tool_name": record["tool_name"],
                "tool_args": record["tool_args"],
                "parsed": record["parsed"],
            }
            for record in self.records
        ]


def get_camera_frame_bundle(camera):
    try:
        rgb, depth_array, depth_rs = camera.get_frames()
    except Exception:
        return None
    if rgb is None:
        return None
    return {"rgb": rgb, "depth_array": depth_array, "depth_rs": depth_rs}


def resize_to_height(cv2, image, target_height):
    height, width = image.shape[:2]
    if height == target_height:
        return image.copy(), 1.0, 1.0
    scale = target_height / max(height, 1)
    target_width = max(1, int(round(width * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (target_width, target_height), interpolation=interpolation)
    return resized, target_width / max(width, 1), target_height / max(height, 1)


def draw_label_box(cv2, image, x, y, lines, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    padding = 6
    line_height = 17
    clean_lines = [short_text(line, 56) for line in lines if line]
    if not clean_lines:
        return

    text_sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in clean_lines]
    box_width = min(image.shape[1] - 2, max(width for width, _ in text_sizes) + padding * 2)
    box_height = line_height * len(clean_lines) + padding * 2
    x = max(0, min(image.shape[1] - box_width - 1, int(x)))
    y = max(box_height + 1, min(image.shape[0] - 1, int(y)))
    top = y - box_height
    bottom = y
    right = x + box_width

    overlay = image.copy()
    cv2.rectangle(overlay, (x, top), (right, bottom), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, image, 0.38, 0, image)
    cv2.rectangle(image, (x, top), (right, bottom), color, 1)

    text_y = top + padding + 12
    for line in clean_lines:
        cv2.putText(image, line, (x + padding, text_y), font, scale, color, thickness, cv2.LINE_AA)
        text_y += line_height


def draw_marker(cv2, image, point, color, lines):
    try:
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
    except (TypeError, ValueError, IndexError):
        return
    if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
        return
    cv2.drawMarker(image, (x, y), color, cv2.MARKER_CROSS, 28, 2)
    cv2.circle(image, (x, y), 6, color, -1)
    draw_label_box(cv2, image, x + 12, y - 12, lines, color)


def depth_intrinsics(depth_rs):
    if depth_rs is None:
        return None
    try:
        return depth_rs.profile.as_video_stream_profile().intrinsics
    except Exception:
        return None


def project_camera_point_to_pixel(point_camera, intrinsics):
    if intrinsics is None:
        return None
    point = np.array(point_camera, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)) or point[2] <= 0.0:
        return None
    u = (point[0] / point[2]) * intrinsics.fx + intrinsics.ppx
    v = (point[1] / point[2]) * intrinsics.fy + intrinsics.ppy
    if not np.isfinite(u) or not np.isfinite(v):
        return None
    return [float(u), float(v)]


def project_robot_point_to_camera_pixel(point_robot, camera_name, intrinsics, static_pose):
    point_base = np.array([point_robot[0], point_robot[1], point_robot[2], 1.0], dtype=float)
    if camera_name == "d435":
        T_cam_to_base = get_d435_transform()
        if T_cam_to_base is None:
            return None
        T_base_to_cam = np.linalg.inv(T_cam_to_base)
        point_camera = T_base_to_cam @ point_base
        return project_camera_point_to_pixel(point_camera[:3], intrinsics)

    if camera_name == "d405":
        if static_pose is None:
            return None
        T_cam_to_ee = get_d405_hand_eye_transform()
        if T_cam_to_ee is None:
            return None
        T_ee_to_base = np.array(static_pose, dtype=float)
        T_cam_to_base = T_ee_to_base @ T_cam_to_ee
        T_base_to_cam = np.linalg.inv(T_cam_to_base)
        point_camera = T_base_to_cam @ point_base
        return project_camera_point_to_pixel(point_camera[:3], intrinsics)

    return None


def draw_trajectory_projection(cv2, panel, camera_name, plan, frame_bundle, static_pose, scale_x, scale_y):
    if not isinstance(plan, dict):
        return []
    waypoints = plan.get("waypoints")
    target_xyz = plan.get("target_xyz")
    if not isinstance(waypoints, list):
        return []

    intrinsics = depth_intrinsics(frame_bundle.get("depth_rs") if frame_bundle else None)
    projected_waypoints = []
    projected_target = []
    waypoint_points = []
    target_points = []
    if isinstance(target_xyz, list) and len(target_xyz) == 3:
        target_points.append(("target", target_xyz))
    for index, waypoint in enumerate(waypoints):
        if isinstance(waypoint, list) and len(waypoint) == 3:
            waypoint_points.append((f"wp{index + 1}", waypoint))

    def project_points(points):
        projected = []
        for label, point_robot in points:
            try:
                pixel = project_robot_point_to_camera_pixel(point_robot, camera_name, intrinsics, static_pose)
            except Exception:
                pixel = None
            if pixel is None:
                continue
            x = int(round(pixel[0] * scale_x))
            y = int(round(pixel[1] * scale_y))
            if 0 <= x < panel.shape[1] and 0 <= y < panel.shape[0]:
                projected.append((label, x, y))
        return projected

    projected_target = project_points(target_points)
    projected_waypoints = project_points(waypoint_points)

    if len(projected_waypoints) >= 2:
        points = np.array([[x, y] for _, x, y in projected_waypoints], dtype=np.int32)
        cv2.polylines(panel, [points], isClosed=False, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    for label, x, y in projected_waypoints:
        cv2.circle(panel, (x, y), 7, (255, 0, 255), -1)
        cv2.putText(panel, label, (x + 8, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)

    for label, x, y in projected_target:
        cv2.drawMarker(panel, (x, y), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 2)
        cv2.putText(panel, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)

    return projected_target + projected_waypoints


def render_camera_panel(cv2, camera_name, frame_bundle, localization_results, plan, static_pose, target_height):
    if frame_bundle is None or frame_bundle.get("rgb") is None:
        panel = np.zeros((target_height, int(target_height * 4 / 3), 3), dtype=np.uint8)
        draw_label_box(cv2, panel, 12, 42, [f"{camera_name.upper()}: no frame"], (0, 255, 255))
        return panel, []

    frame = frame_bundle["rgb"]
    panel, scale_x, scale_y = resize_to_height(cv2, frame, target_height)
    draw_label_box(cv2, panel, 12, 34, [f"{camera_name.upper()} latest frame"], (255, 255, 0))

    for result in localization_results:
        if not isinstance(result, dict):
            continue
        pixel = result.get("pixel")
        if not isinstance(pixel, list) or len(pixel) != 2:
            continue
        scaled_pixel = [float(pixel[0]) * scale_x, float(pixel[1]) * scale_y]
        ok = result.get("status") == "ok"
        color = (0, 255, 0) if ok else (0, 0, 255)
        label_lines = ["LLM pixel"]
        xyz_text = format_xyz(result.get("xyz_robot"))
        if xyz_text:
            label_lines.append(f"xyz {xyz_text} m")
        elif result.get("depth_m") is not None:
            label_lines.append(f"depth {float(result['depth_m']):.3f} m")
        elif result.get("reason"):
            label_lines.append(result.get("reason"))
        draw_marker(cv2, panel, scaled_pixel, color, label_lines)

    projected = draw_trajectory_projection(
        cv2,
        panel,
        camera_name,
        plan,
        frame_bundle,
        static_pose,
        scale_x,
        scale_y,
    )
    return panel, projected


def add_output_bands(cv2, image, command, plan, projections_by_camera):
    width = image.shape[1]
    header = np.zeros((54, width, 3), dtype=np.uint8)
    footer = np.zeros((88, width, 3), dtype=np.uint8)

    cv2.putText(
        header,
        short_text(f"Command: {command}", 160),
        (14, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    footer_lines = []
    if isinstance(plan, dict):
        target_text = format_xyz(plan.get("target_xyz"))
        waypoints = plan.get("waypoints") if isinstance(plan.get("waypoints"), list) else []
        footer_lines.append(f"Trajectory: {len(waypoints)} planned waypoint(s); target={target_text or 'n/a'}")
        warning = plan.get("trajectory_metadata", {}).get("assumption") if isinstance(plan.get("trajectory_metadata"), dict) else None
        if warning:
            footer_lines.append(short_text(warning, 150))
    else:
        footer_lines.append("Trajectory: no plan_robot_trajectory result recorded for this command.")
    footer_lines.append(
        "Projection: "
        + ", ".join(f"{camera.upper()} {len(points)} point(s)" for camera, points in projections_by_camera.items())
    )

    for index, line in enumerate(footer_lines[:3]):
        cv2.putText(
            footer,
            short_text(line, 170),
            (14, 24 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return np.vstack([header, image, footer])


def save_side_by_side_marker_trajectory_image(command_dir, command, d435, d405, recorder, static_pose):
    cv2 = load_cv2()
    if cv2 is None:
        write_json(
            command_dir / "side_by_side_marker_trajectory_unavailable.json",
            {"reason": "OpenCV is required to render the side-by-side marker/trajectory image."},
        )
        return None

    frame_bundles = {
        "d435": get_camera_frame_bundle(d435),
        "d405": get_camera_frame_bundle(d405),
    }
    localization = recorder.latest_localization_results()
    plan = recorder.latest_plan()

    heights = [
        bundle["rgb"].shape[0]
        for bundle in frame_bundles.values()
        if bundle is not None and bundle.get("rgb") is not None
    ]
    target_height = min(max(heights), 720) if heights else 480

    panels = {}
    projections = {}
    for camera_name in ("d435", "d405"):
        panel, projected = render_camera_panel(
            cv2,
            camera_name,
            frame_bundles[camera_name],
            localization.get(camera_name, []),
            plan,
            static_pose,
            target_height,
        )
        panels[camera_name] = panel
        projections[camera_name] = projected

    combined = cv2.hconcat([panels["d435"], panels["d405"]])
    output = add_output_bands(cv2, combined, command, plan, projections)
    output_path = command_dir / "side_by_side_marker_trajectory.png"
    if not cv2.imwrite(str(output_path), output):
        return None

    write_json(
        command_dir / "side_by_side_marker_trajectory_summary.json",
        {
            "image": str(output_path),
            "localization": localization,
            "trajectory": plan,
            "trajectory_projection_points": {
                camera: [{"label": label, "display_x": x, "display_y": y} for label, x, y in points]
                for camera, points in projections.items()
            },
            "tool_results": recorder.summary(),
        },
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a main.py-like LLM/camera runtime with all Franka robot access disabled."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--ee-pose-source",
        choices=["auto", "pose-plan", "identity", "none"],
        default="auto",
        help=(
            "Static EE pose source for D405 transforms and trajectory math. "
            "auto uses d405_hand_eye_pose_plan.json when available, else identity."
        ),
    )
    parser.add_argument("--ee-pose-file", type=Path, help="Optional JSON file containing matrix/T_ee_to_base/poses.")
    parser.add_argument("--pose-index", type=int, default=1, help="1-based pose index when using a pose-plan JSON.")
    parser.add_argument("--no-viewer", action="store_true", help="Do not start the OpenCV camera viewer.")
    args = parser.parse_args()

    run_dir = create_run_dir(args.output_dir)
    static_pose, pose_source = choose_static_ee_pose(args.ee_pose_source, args.ee_pose_file, args.pose_index)

    d435 = None
    d405 = None
    camera_viewer = None
    command_index = 0

    old_debug_dir = cfg.DEBUG_IMAGE_DIR
    cfg.DEBUG_IMAGE_DIR = str(run_dir / "debug_images")

    write_json(
        run_dir / "no_robot_runtime_config.json",
        {
            "robot_connection": "disabled",
            "robot_motion": "disabled",
            "tools_removed": ["execute_robot_waypoints"],
            "static_robot_ee_pose_source": pose_source,
            "static_robot_ee_pose": static_pose,
            "debug_image_dir": cfg.DEBUG_IMAGE_DIR,
            "llm_api_url": cfg.LLM_API_URL,
            "model": cfg.QWEN_MODEL_PATH,
        },
    )

    print("--- Main.py No-Robot Debug Runtime ---")
    print("Robot connection: disabled. Robot motion: disabled.")
    print("Available runtime behavior: cameras, LLM tools, pixel localization, fusion, trajectory planning.")
    print("Unavailable behavior: physical waypoint execution.")
    print(f"Static EE pose source: {pose_source}")
    print(f"Debug/run folder: {run_dir}")

    try:
        print("Starting Cameras...")
        d435_serial = configured_serial(cfg.D435_SERIAL)
        d405_serial = configured_serial(cfg.D405_SERIAL)
        if d435_serial is None or d405_serial is None:
            print("Warning: Camera serial numbers are not fully configured. Check Working/config.py.")

        d435 = RealSense(serial_number=d435_serial, resolution=cfg.D435_RESOLUTION, fps=cfg.CAMERA_FPS)
        d405 = RealSense(serial_number=d405_serial, resolution=cfg.D405_RESOLUTION, fps=cfg.CAMERA_FPS)

        if not args.no_viewer:
            camera_viewer = CameraViewer(d435, d405)
            camera_viewer.start()

        llm = LLMinterface(
            model=cfg.QWEN_MODEL_PATH,
            tools_json=no_robot_tool_schemas(),
            api_url=cfg.LLM_API_URL,
            api_key=cfg.LLM_API_KEY,
        )
        llm.messages.append(no_robot_system_note(pose_source))

        def static_pose_provider():
            return copy.deepcopy(static_pose)

        recorder = ToolResultRecorder(
            camera_viewer.handle_tool_result
            if camera_viewer is not None
            else None
        )

        while True:
            command = llm.get_text()
            if command.strip().lower() in {"exit", "quit"}:
                break

            command_index += 1
            command_dir = run_dir / f"command_{command_index:03d}"
            command_dir.mkdir(parents=True, exist_ok=True)
            (command_dir / "command.txt").write_text(command, encoding="utf-8")

            if camera_viewer is not None:
                camera_viewer.set_prompt(command)

            recorder.reset()
            llm.send_message_with_tools(
                d435,
                d405,
                robot_ee_pose=copy.deepcopy(static_pose),
                robot_pose_provider=static_pose_provider,
                robot_interface=None,
                tool_result_callback=recorder.handle_tool_result,
            )

            llm.print_message()
            write_json(command_dir / "conversation_sanitized.json", sanitized_messages(llm.messages))
            output_image = save_side_by_side_marker_trajectory_image(
                command_dir,
                command,
                d435,
                d405,
                recorder,
                static_pose,
            )
            if output_image is not None:
                print(f"Saved marker/trajectory output image: {output_image}")

            if camera_viewer is not None:
                camera_viewer.set_status("Ready for next command.")

            llm.prune_image_history()

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as exc:
        write_json(run_dir / "fatal_error.json", {"error": f"{type(exc).__name__}: {exc}"})
        print(f"FAIL: {exc}")
        return 1
    finally:
        cfg.DEBUG_IMAGE_DIR = old_debug_dir
        print("Shutting down hardware safely...")
        stop_camera_viewer(camera_viewer)
        stop_camera(d435, "D435")
        stop_camera(d405, "D405")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
