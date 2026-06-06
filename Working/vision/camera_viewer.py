"""
Live Camera Viewer
------------------
Shows the D435 and D405 streams during main.py and overlays the latest
LLM-selected localization pixels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import threading
import time

import numpy as np

import config as cfg


def _load_cv2():
    """Lazy-load OpenCV so imports still work on machines without GUI support."""
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "opencv-python is required for the live camera viewer. "
            "Install it or set CAMERA_VIEWER_ENABLED=false."
        ) from exc
    return cv2


@dataclass(frozen=True)
class CameraAnnotation:
    """One marker to draw on one camera feed."""

    camera_name: str
    pixel: tuple[int, int]
    status: str
    label: str
    xyz_robot: list[float] | None = None
    depth_m: float | None = None
    reason: str | None = None
    tool_name: str | None = None
    created_s: float = 0.0


def _short_text(value, max_chars=72):
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _format_xyz(xyz):
    if not xyz or len(xyz) != 3:
        return None
    try:
        return f"xyz {float(xyz[0]):.3f}, {float(xyz[1]):.3f}, {float(xyz[2]):.3f} m"
    except (TypeError, ValueError):
        return None


def _parse_json_result(result_text):
    try:
        data = json.loads(result_text)
    except (TypeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _has_gui_display():
    if os.name != "posix":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class CameraViewer:
    """
    Background OpenCV viewer for the live RealSense streams.

    The terminal prompt blocks the main thread while waiting for commands, so the
    viewer owns a small display thread. OpenCV GUI support can vary across
    machines; failures here disable only the viewer, not the robot/LLM loop.
    """

    def __init__(
        self,
        d435_cam,
        d405_cam,
        window_name=None,
        max_width=None,
        max_height=None,
    ):
        self.d435_cam = d435_cam
        self.d405_cam = d405_cam
        self.window_name = window_name or cfg.CAMERA_VIEWER_WINDOW_NAME
        self.max_width = int(max_width or cfg.CAMERA_VIEWER_MAX_DISPLAY_WIDTH)
        self.max_height = int(max_height or cfg.CAMERA_VIEWER_MAX_DISPLAY_HEIGHT)

        self._cv2 = None
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._enabled = False
        self._closed_by_user = False
        self._prompt = ""
        self._status = "Waiting for command."
        self._annotations = {"d435": [], "d405": []}

    def start(self):
        """Start the viewer thread if the feature is enabled and OpenCV loads."""
        if not cfg.CAMERA_VIEWER_ENABLED:
            print("Camera viewer disabled by CAMERA_VIEWER_ENABLED=false.")
            return False
        if not _has_gui_display():
            print("Warning: camera viewer disabled because no GUI display is available.")
            return False

        try:
            self._cv2 = _load_cv2()
        except RuntimeError as exc:
            print(f"Warning: camera viewer disabled: {exc}")
            return False

        self._enabled = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"Camera viewer opened: {self.window_name} (press q or Esc in the window to close it).")
        return True

    def stop(self):
        """Signal the viewer thread to stop and wait briefly for cleanup."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._enabled = False

    def is_running(self):
        return self._enabled and not self._stop_event.is_set() and not self._closed_by_user

    def set_prompt(self, prompt):
        """Show the active user command and clear stale localization overlays."""
        with self._lock:
            self._prompt = _short_text(prompt, 96)
            self._status = "LLM processing command."
            self._annotations = {"d435": [], "d405": []}

    def set_status(self, status):
        with self._lock:
            self._status = _short_text(status, 120)

    def handle_tool_result(self, tool_name, tool_args, result_text, user_prompt=None):
        """
        Update overlays from one LLM tool result.

        Localization tools produce JSON with pixels and robot-frame XYZ values.
        Camera-view tools and planning/execution tools only update the status line.
        """
        del tool_args  # The result JSON contains the validated pixels we should draw.

        if user_prompt:
            prompt = _short_text(user_prompt, 96)
        else:
            prompt = None

        annotations, status = self._annotations_from_result(tool_name, result_text)
        with self._lock:
            if prompt is not None:
                self._prompt = prompt
            if annotations is not None:
                self._annotations = {"d435": [], "d405": []}
                for annotation in annotations:
                    self._annotations[annotation.camera_name].append(annotation)
            if status:
                self._status = status

    def snapshot_state(self):
        """Return a serializable copy of viewer state for tests and debugging."""
        with self._lock:
            return {
                "prompt": self._prompt,
                "status": self._status,
                "annotations": {
                    camera_name: [asdict(annotation) for annotation in annotations]
                    for camera_name, annotations in self._annotations.items()
                },
            }

    def _annotations_from_result(self, tool_name, result_text):
        if tool_name in ("get_birds_eye_view", "get_eye_in_hand_view"):
            camera = "D435" if tool_name == "get_birds_eye_view" else "D405"
            return None, f"{camera} image sent to LLM."

        if tool_name == "plan_robot_trajectory":
            return None, "Robot trajectory planned."

        if tool_name == "execute_robot_waypoints":
            data = _parse_json_result(result_text)
            if data is not None:
                status = data.get("execution_status") or "complete"
                return None, f"Robot motion {status}."
            return None, _short_text(result_text, 120)

        if tool_name == "get_xyz_fused":
            data = _parse_json_result(result_text)
            if data is None:
                return None, _short_text(result_text, 120)
            annotations = []
            annotations.extend(
                self._annotations_for_camera(
                    "d435",
                    data.get("d435", {}).get("results", []),
                    tool_name,
                )
            )
            annotations.extend(
                self._annotations_for_camera(
                    "d405",
                    data.get("d405", {}).get("results", []),
                    tool_name,
                )
            )
            status = self._status_from_fused_result(data)
            return annotations, status

        if tool_name in ("get_xyz_d435", "get_xyz_d405"):
            data = _parse_json_result(result_text)
            if data is None:
                return None, _short_text(result_text, 120)
            camera_name = data.get("source_camera")
            if camera_name not in ("d435", "d405"):
                camera_name = "d435" if tool_name == "get_xyz_d435" else "d405"
            annotations = self._annotations_for_camera(
                camera_name,
                data.get("points", []),
                tool_name,
            )
            status = self._status_from_single_result(camera_name, data)
            return annotations, status

        return None, _short_text(result_text, 120)

    def _annotations_for_camera(self, camera_name, results, tool_name):
        if not isinstance(results, list):
            return []

        with self._lock:
            prompt = self._prompt or "LLM target"

        annotations = []
        for result in results:
            if not isinstance(result, dict):
                continue
            pixel = result.get("pixel")
            if not isinstance(pixel, list) or len(pixel) != 2:
                continue
            try:
                u = int(pixel[0])
                v = int(pixel[1])
            except (TypeError, ValueError):
                continue

            status = str(result.get("status") or "unknown")
            annotations.append(
                CameraAnnotation(
                    camera_name=camera_name,
                    pixel=(u, v),
                    status=status,
                    label=_short_text(prompt, 44),
                    xyz_robot=result.get("xyz_robot"),
                    depth_m=result.get("depth_m"),
                    reason=result.get("reason"),
                    tool_name=tool_name,
                    created_s=time.monotonic(),
                )
            )
        return annotations

    def _status_from_single_result(self, camera_name, data):
        points = data.get("points", [])
        ok_points = [
            point for point in points
            if isinstance(point, dict) and point.get("status") == "ok"
        ]
        if ok_points:
            xyz_text = _format_xyz(ok_points[0].get("xyz_robot"))
            if xyz_text:
                return f"{camera_name.upper()} identified target: {xyz_text}."
            return f"{camera_name.upper()} identified target."

        if isinstance(points, list) and points:
            reason = points[0].get("reason") if isinstance(points[0], dict) else None
            return _short_text(f"{camera_name.upper()} localization failed: {reason}", 120)
        return f"{camera_name.upper()} localization returned no points."

    def _status_from_fused_result(self, data):
        fused = data.get("fused_result", {})
        if isinstance(fused, dict) and fused.get("valid"):
            xyz_text = _format_xyz(fused.get("xyz"))
            sources = ", ".join(fused.get("sources_used", []))
            if xyz_text and sources:
                return f"Fused target identified from {sources}: {xyz_text}."
            if xyz_text:
                return f"Fused target identified: {xyz_text}."
            return "Fused target identified."

        reason = fused.get("reason") if isinstance(fused, dict) else None
        return _short_text(f"Fused localization failed: {reason}", 120)

    def _run(self):
        cv2 = self._cv2
        window_created = False
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            window_created = True

            while not self._stop_event.is_set():
                display = self._compose_display(cv2)
                cv2.imshow(self.window_name, display)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    self._closed_by_user = True
                    self._stop_event.set()
                    break

                time.sleep(0.01)
        except Exception as exc:
            print(f"Warning: camera viewer stopped: {exc}")
            self._enabled = False
        finally:
            if window_created:
                try:
                    cv2.destroyWindow(self.window_name)
                except Exception:
                    pass

    def _compose_display(self, cv2):
        d435_frame = self._get_color_frame(self.d435_cam)
        d405_frame = self._get_color_frame(self.d405_cam)
        target_height = self._target_panel_height(d435_frame, d405_frame)

        with self._lock:
            prompt = self._prompt
            status = self._status
            d435_annotations = list(self._annotations["d435"])
            d405_annotations = list(self._annotations["d405"])

        d435_panel = self._render_camera_panel(
            cv2,
            d435_frame,
            "D435 overhead",
            "d435",
            d435_annotations,
            target_height,
        )
        d405_panel = self._render_camera_panel(
            cv2,
            d405_frame,
            "D405 wrist",
            "d405",
            d405_annotations,
            target_height,
        )

        combined = np.hstack([d435_panel, d405_panel])
        display = self._add_status_bands(cv2, combined, prompt, status)
        return self._fit_to_bounds(cv2, display)

    def _get_color_frame(self, camera):
        if camera is None:
            return None
        try:
            frame, _, _ = camera.get_frames()
        except Exception:
            return None
        return frame

    def _target_panel_height(self, *frames):
        heights = [frame.shape[0] for frame in frames if frame is not None]
        native_height = max(heights) if heights else 480
        reserved_for_text = 92
        max_panel_height = max(160, self.max_height - reserved_for_text)
        return max(160, min(native_height, max_panel_height))

    def _render_camera_panel(self, cv2, frame, title, camera_name, annotations, target_height):
        if frame is None:
            panel = np.zeros((target_height, int(target_height * 4 / 3), 3), dtype=np.uint8)
            self._draw_label_box(cv2, panel, 12, 42, [f"{title}: waiting for frames"], (0, 255, 255))
            return panel

        raw_height, raw_width = frame.shape[:2]
        scale = target_height / max(raw_height, 1)
        target_width = max(1, int(round(raw_width * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        panel = cv2.resize(frame.copy(), (target_width, target_height), interpolation=interpolation)

        self._draw_label_box(cv2, panel, 12, 34, [title], (255, 255, 0))

        scale_x = target_width / max(raw_width, 1)
        scale_y = target_height / max(raw_height, 1)
        for annotation in annotations:
            if annotation.camera_name != camera_name:
                continue
            x = int(round(annotation.pixel[0] * scale_x))
            y = int(round(annotation.pixel[1] * scale_y))
            self._draw_annotation(cv2, panel, annotation, x, y)

        return panel

    def _draw_annotation(self, cv2, image, annotation, x, y):
        ok = annotation.status == "ok"
        color = (0, 255, 0) if ok else (0, 0, 255)

        x = max(0, min(image.shape[1] - 1, x))
        y = max(0, min(image.shape[0] - 1, y))
        cv2.drawMarker(image, (x, y), color, cv2.MARKER_CROSS, 28, 2)
        cv2.circle(image, (x, y), 6, color, -1)

        lines = [annotation.label or "LLM target"]
        xyz_text = _format_xyz(annotation.xyz_robot)
        if ok and xyz_text:
            lines.append(xyz_text)
        elif ok and annotation.depth_m is not None:
            lines.append(f"depth {float(annotation.depth_m):.3f} m")
        elif annotation.reason:
            lines.append(_short_text(annotation.reason, 48))
        else:
            lines.append(annotation.status)

        self._draw_label_box(cv2, image, x + 12, y - 12, lines, color)

    def _draw_label_box(self, cv2, image, x, y, lines, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thickness = 1
        padding = 6
        line_height = 17
        clean_lines = [_short_text(line, 54) for line in lines if line]
        if not clean_lines:
            return

        text_sizes = [
            cv2.getTextSize(line, font, scale, thickness)[0]
            for line in clean_lines
        ]
        box_width = max(width for width, _ in text_sizes) + padding * 2
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

    def _add_status_bands(self, cv2, image, prompt, status):
        header_height = 54
        footer_height = 38
        width = image.shape[1]
        header = np.zeros((header_height, width, 3), dtype=np.uint8)
        footer = np.zeros((footer_height, width, 3), dtype=np.uint8)

        prompt_text = f"Prompt: {prompt or 'waiting for command'}"
        status_text = f"Status: {status or 'idle'}"
        self._put_fitted_text(cv2, header, prompt_text, (14, 34), width - 28, (255, 255, 255), 0.62, 1)
        self._put_fitted_text(cv2, footer, status_text, (14, 25), width - 28, (0, 255, 255), 0.52, 1)
        return np.vstack([header, image, footer])

    def _put_fitted_text(self, cv2, image, text, origin, max_width, color, scale, thickness):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fitted = _short_text(text, 180)
        while fitted:
            text_width = cv2.getTextSize(fitted, font, scale, thickness)[0][0]
            if text_width <= max_width:
                break
            fitted = _short_text(fitted, max(4, len(fitted) - 6))
        if fitted:
            cv2.putText(image, fitted, origin, font, scale, color, thickness, cv2.LINE_AA)

    def _fit_to_bounds(self, cv2, image):
        height, width = image.shape[:2]
        scale = min(self.max_width / width, self.max_height / height, 1.0)
        if scale >= 1.0:
            return image
        target_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
