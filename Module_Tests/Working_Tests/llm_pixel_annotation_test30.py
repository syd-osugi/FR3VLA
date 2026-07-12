"""
LLM Pixel Annotation Test
-------------------------
Runs a single main.py-style LLM interaction that captures live RealSense frames,
asks the configured model to localize an operator-described object, and saves an
annotated image with the returned pixel coordinates.

Outputs are written to `Module_Tests/Test_Outputs/llm_pixel_annotation_test/<timestamp>/`
with one `command_001` folder containing:
  - `command.txt`: original prompt
  - `conversation_sanitized.json`: chat history with image payloads stripped
  - `side_by_side_marker_trajectory.png`: annotated D435/D405 composite image
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from _working_test_utils import TEST_OUTPUTS_DIR, add_working_to_path

# Ensure the Working package is importable before pulling in runtime modules.
add_working_to_path()

import config as cfg  # noqa: E402
from hardware.camera import RealSense  # noqa: E402
from vision.llm_interface import LLMinterface  # noqa: E402
import vision.tools as tools  # noqa: E402

# Reuse the well-tested helpers from the no-robot interactive debugger so the
# test matches main.py behaviour without having to reimplement its plumbing.
from live_llm_camera_seg_debug.interactive_no_robot_llm_debug import (  # noqa: E402
    ToolResultRecorder,
    choose_static_ee_pose,
    create_run_dir,
    no_robot_system_note,
    no_robot_tool_schemas,
    sanitized_messages,
    save_side_by_side_marker_trajectory_image,
    stop_camera,
    stop_camera_viewer,
    write_json,
)


def parse_args() -> argparse.Namespace:
    """Set up CLI flags mirroring the interactive debug runner for consistency."""
    parser = argparse.ArgumentParser(description="Run a single LLM pixel annotation command.")
    parser.add_argument(
        "--prompt",
        help="Object instruction to send to the LLM. Defaults to interactive input if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TEST_OUTPUTS_DIR / "llm_pixel_annotation_test",
        help=(
            "Root folder for run outputs (default: Module_Tests/Test_Outputs/"
            "llm_pixel_annotation_test)."
        ),
    )
    parser.add_argument(
        "--ee-pose-source",
        choices=["auto", "pose-plan", "identity", "none"],
        default="auto",
        help="Static robot EE pose source for D405 math (default: auto).",
    )
    parser.add_argument(
        "--ee-pose-file",
        type=Path,
        help="Optional JSON file providing matrix/T_ee_to_base/poses for the static EE pose.",
    )
    parser.add_argument(
        "--pose-index",
        type=int,
        default=1,
        help="1-based pose index when using a pose-plan JSON (default: 1).",
    )
    parser.add_argument(
        "--no-require-pixel-output",
        action="store_true",
        help="Save artifacts even if the LLM never provides pixel coordinates.",
    )
    parser.add_argument(
        "--require-valid-depth",
        action="store_true",
        help="Also require at least one returned pixel to produce a valid depth/XYZ result.",
    )
    return parser.parse_args()


def _pixel_in_bounds(camera_name: str, pixel) -> tuple[bool, str | None]:
    """Check whether [u, v] falls inside the configured camera resolution."""
    if camera_name == "d435":
        width, height = cfg.D435_RESOLUTION
    elif camera_name == "d405":
        width, height = cfg.D405_RESOLUTION
    else:
        return False, f"unknown camera {camera_name!r}"

    if not isinstance(pixel, (list, tuple)) or len(pixel) != 2:
        return False, f"pixel is not [u, v]: {pixel!r}"
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in pixel):
        return False, f"pixel values must be integers: {pixel!r}"

    u, v = pixel
    if not (0 <= u < width and 0 <= v < height):
        return False, f"{camera_name} pixel {pixel!r} outside {width}x{height}"
    return True, None


def collect_requested_pixels(recorder: ToolResultRecorder):
    """
    Return pixel coordinates the LLM supplied to localization tools.

    These are the coordinates we care about for this test: they are not produced
    by deterministic code; they come from the model's tool arguments after it
    inspected camera images.
    """
    pixels = []
    for record in recorder.records:
        tool_name = record.get("tool_name")
        args = record.get("tool_args") or {}
        if tool_name == "get_xyz_fused":
            for camera_name, key in (("d435", "d435_coords"), ("d405", "d405_coords")):
                coords = args.get(key)
                if isinstance(coords, list):
                    for pixel in coords:
                        pixels.append(
                            {
                                "tool_name": tool_name,
                                "camera": camera_name,
                                "pixel": pixel,
                            }
                        )
        elif tool_name in ("get_xyz_d435", "get_xyz_d405"):
            camera_name = "d435" if tool_name == "get_xyz_d435" else "d405"
            coords = args.get("coords")
            if isinstance(coords, list):
                for pixel in coords:
                    pixels.append(
                        {
                            "tool_name": tool_name,
                            "camera": camera_name,
                            "pixel": pixel,
                        }
                    )
    return pixels


def collect_valid_depth_results(recorder: ToolResultRecorder):
    """Return localization result records that reached a valid robot-frame XYZ."""
    valid = []
    for camera_name, results in recorder.latest_localization_results().items():
        for result in results:
            if isinstance(result, dict) and result.get("status") == "ok":
                valid.append({"camera": camera_name, "result": result})
    return valid


def evaluate_pixel_output(recorder: ToolResultRecorder, require_valid_depth: bool):
    """Build a pass/fail report for the LLM's pixel-coordinate output."""
    requested_pixels = collect_requested_pixels(recorder)
    invalid_pixels = []
    for item in requested_pixels:
        ok, reason = _pixel_in_bounds(item["camera"], item["pixel"])
        if not ok:
            invalid_pixels.append({**item, "reason": reason})

    valid_depth_results = collect_valid_depth_results(recorder)
    failures = []
    if not requested_pixels:
        failures.append("LLM did not call a pixel-localization tool with coordinates.")
    if invalid_pixels:
        failures.append("LLM returned one or more malformed or out-of-bounds pixels.")
    if require_valid_depth and not valid_depth_results:
        failures.append("No returned pixel produced a valid depth/XYZ localization.")

    return {
        "passed": not failures,
        "failures": failures,
        "requested_pixels": requested_pixels,
        "invalid_pixels": invalid_pixels,
        "valid_depth_results": valid_depth_results,
        "require_valid_depth": require_valid_depth,
    }


def main() -> int:
    args = parse_args()

    # Create a timestamped run folder so multiple invocations keep their evidence.
    run_dir = create_run_dir(args.output_dir)
    command_dir = run_dir / "command_001"
    command_dir.mkdir(parents=True, exist_ok=True)

    # Determine which static robot pose to use so D405 transforms stay reproducible.
    static_pose, pose_source = choose_static_ee_pose(
        args.ee_pose_source,
        args.ee_pose_file,
        args.pose_index,
    )

    d435 = None
    d405 = None
    camera_viewer = None

    try:
        # Respect any serial overrides so each camera maps to the correct USB device.
        configured_serial = getattr(tools, "configured_serial", None)
        d435_serial = configured_serial(cfg.D435_SERIAL) if callable(configured_serial) else cfg.D435_SERIAL
        d405_serial = configured_serial(cfg.D405_SERIAL) if callable(configured_serial) else cfg.D405_SERIAL

        # Spin up both RealSense pipelines exactly as main.py does (live RGB + depth feeds).
        d435 = RealSense(serial_number=d435_serial, resolution=cfg.D435_RESOLUTION, fps=cfg.CAMERA_FPS)
        d405 = RealSense(serial_number=d405_serial, resolution=cfg.D405_RESOLUTION, fps=cfg.CAMERA_FPS)

        # Instantiate the normal LLM interface with the no-robot tool list.
        llm = LLMinterface(
            model=cfg.QWEN_MODEL_PATH,
            tools_json=no_robot_tool_schemas(),
            api_url=cfg.LLM_API_URL,
            api_key=cfg.LLM_API_KEY,
        )
        llm.messages.append(no_robot_system_note(pose_source))

        # Either take the CLI-supplied prompt or ask the operator interactively.
        prompt = args.prompt or input("Enter object request for the LLM:\n> ").strip()
        if not prompt:
            raise ValueError("Prompt must not be empty.")

        # Persist the raw prompt for audit/debugging alongside other artifacts.
        (command_dir / "command.txt").write_text(prompt, encoding="utf-8")

        llm.text = prompt  # Mimic get_text(); send_message_with_tools reads this.

        # Record every tool call result so the final overlay can pull localization data.
        recorder = ToolResultRecorder(None)

        def static_pose_provider():
            """Return a fresh copy of the static pose so downstream code can mutate safely."""
            return copy.deepcopy(static_pose)

        # The test mimics the operator issuing a single message; send_message_with_tools
        # appends that user message internally, matching main.py.
        llm.send_message_with_tools(
            d435,
            d405,
            robot_ee_pose=copy.deepcopy(static_pose),
            robot_pose_provider=static_pose_provider,
            robot_interface=None,
            tool_result_callback=recorder.handle_tool_result,
        )

        # Echo the final LLM reply for immediate CLI feedback.
        llm.print_message()

        # Save the conversation minus base64 blobs to keep logs light and readable.
        write_json(
            command_dir / "conversation_sanitized.json",
            sanitized_messages(llm.messages),
        )

        # Generate the same side-by-side diagnostic image main.py produces after each command.
        annotated_path = save_side_by_side_marker_trajectory_image(
            command_dir,
            prompt,
            d435,
            d405,
            recorder,
            static_pose,
        )
        pixel_check = evaluate_pixel_output(
            recorder,
            require_valid_depth=args.require_valid_depth,
        )
        write_json(command_dir / "pixel_output_check.json", pixel_check)

        # Summarize run metadata so reviewers can trace which assets were produced and why.
        summary = {
            "prompt": prompt,
            "pose_source": pose_source,
            "annotated_image": str(annotated_path) if annotated_path is not None else None,
            "conversation_file": str(command_dir / "conversation_sanitized.json"),
            "pixel_output_check": pixel_check,
            "tools_removed": ["execute_robot_waypoints"],
        }
        write_json(command_dir / "summary.json", summary)

        print("--- LLM Pixel Annotation Test ---")
        print(f"Run folder: {run_dir}")
        if annotated_path:
            print(f"Annotated image saved: {annotated_path}")
        else:
            print("Annotated image not generated (no markers returned).")

        if pixel_check["passed"]:
            print("Pixel output check: PASS")
        else:
            print("Pixel output check: FAIL")
            for failure in pixel_check["failures"]:
                print(f"  - {failure}")
            if not args.no_require_pixel_output:
                return 1

        return 0

    except Exception as exc:  # noqa: BLE001
        # Preserve the failure reason so investigators can see what went wrong post-run.
        write_json(command_dir / "fatal_error.json", {"error": f"{type(exc).__name__}: {exc}"})
        print(f"FAIL: {exc}")
        return 1
    finally:
        # Always release hardware resources to avoid leaving the cameras in a locked state.
        stop_camera_viewer(camera_viewer)
        stop_camera(d435, "D435")
        stop_camera(d405, "D405")


if __name__ == "__main__":
    raise SystemExit(main())
