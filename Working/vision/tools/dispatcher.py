"""
Tool Dispatcher
---------------
Routes LLM tool calls to camera capture, 3D localization, fusion, and trajectory
planning helpers.

Runtime data flow:
    image tool -> fresh synchronized D435/D405 capture
    localization tool -> fresh synchronized D435/D405 capture -> robot-frame XYZ
    fusion tool -> uses whichever cameras can see the object in that fresh pair
    trajectory tool -> turns the latest robot-frame target into EE waypoints

Important limitation:
    The trajectory tool plans only. It does not execute robot motion and it does
    not continuously update while the robot is moving. To update the trajectory
    after any motion, call a camera/localization tool again and then call
    plan_robot_trajectory again with the updated target.
"""

import json
import math
from numbers import Integral, Real

import config as cfg
from robot.trajectory import get_robot_trajectory_to_point, translate_points_fused

from .camera_frames import capture_synced_frames, frames_for_camera
from .image_utils import encode_image_for_llm
from .localization import process_pixel, save_debug_image, valid_robot_points


CAMERAS = {
    "d435": {
        "label": "D435",
        "view_name": "overhead bird's eye",
        "resolution": cfg.D435_RESOLUTION,
    },
    "d405": {
        "label": "D405",
        "view_name": "eye-in-hand wrist",
        "resolution": cfg.D405_RESOLUTION,
    },
}


def _json(data):
    return json.dumps(data, indent=2)


def _camera_from_name(camera_name, d435_cam, d405_cam):
    if camera_name == "d435":
        return d435_cam
    if camera_name == "d405":
        return d405_cam
    raise ValueError(f"Unknown camera: {camera_name}")


def _is_integer_pixel(value):
    return isinstance(value, Integral) and not isinstance(value, bool)


def _validate_coord_pairs(coords, name, require_nonempty=False):
    if not isinstance(coords, list):
        return (
            f"Error: '{name}' must be a list of [u, v] pixel pairs. "
            "Example: {\"coords\": [[320, 240]]}"
        )

    if require_nonempty and len(coords) == 0:
        return (
            f"Error: '{name}' must be a non-empty list of [u, v] pixel pairs. "
            "Example: {\"coords\": [[320, 240]]}"
        )

    for index, point in enumerate(coords):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return (
                f"Error: invalid '{name}' item {index}: expected [u, v] pixel pair, "
                f"got {point!r}."
            )
        if not all(_is_integer_pixel(value) for value in point):
            return (
                f"Error: invalid '{name}' item {index}: pixel values must be integers, "
                f"got {point!r}."
            )
    return None


def _optional_coords(tool_args, name):
    coords = tool_args.get(name)
    if coords is None:
        return [], None
    error = _validate_coord_pairs(coords, name)
    if error:
        return [], error + " Use null only when that camera cannot see the object."
    return coords, None


def _parse_xyz(value):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("target_xyz must be [x, y, z] in robot base frame meters.")
    return [float(value[0]), float(value[1]), float(value[2])]


def _parse_waypoints(value):
    if not isinstance(value, list) or len(value) == 0:
        raise ValueError(
            "waypoints must be a non-empty list of [x, y, z] robot-base points."
        )

    waypoints = []
    for index, waypoint in enumerate(value):
        if not isinstance(waypoint, list) or len(waypoint) != 3:
            raise ValueError(
                f"waypoints[{index}] must be [x, y, z] in robot base frame meters."
            )
        if not all(isinstance(coord, Real) and not isinstance(coord, bool) for coord in waypoint):
            raise ValueError(f"waypoints[{index}] contains a non-numeric coordinate.")
        parsed = [float(waypoint[0]), float(waypoint[1]), float(waypoint[2])]
        if not all(math.isfinite(coord) for coord in parsed):
            raise ValueError(f"waypoints[{index}] contains a non-finite coordinate.")
        waypoints.append(parsed)
    return waypoints


def _localize_points(camera_name, coords, frames, camera, robot_ee_pose):
    robot_pose = robot_ee_pose if camera_name == "d405" else None
    return [
        process_pixel(
            point=point,
            resolution=CAMERAS[camera_name]["resolution"],
            depth_rs=frames.depth_rs,
            depth_scale=camera.depth_scale,
            source_camera=camera_name,
            robot_ee_pose=robot_pose,
        )
        for point in coords
    ]


def handle_get_camera_view(camera_name, d435_cam, d405_cam):
    """
    Captures a synced D435/D405 pair and returns one RGB image to the LLM.

    Even when the LLM asks for only one camera image, both cameras are captured
    together. This keeps later pixel localization aligned with the same physical
    moment if the user asks for both cameras.
    """
    synced, error = capture_synced_frames(d435_cam, d405_cam)
    if error:
        return f"Error: {error} Please try again.", None

    frames = frames_for_camera(synced, camera_name)
    if frames.rgb is None:
        return f"Error: {CAMERAS[camera_name]['label']} camera is not ready.", None

    try:
        image_msg = encode_image_for_llm(frames.rgb)
    except RuntimeError as exc:
        return f"Error encoding image: {exc}", None

    label = CAMERAS[camera_name]["label"]
    view = CAMERAS[camera_name]["view_name"]
    return (
        f"Synchronized {label} {view} image captured. "
        "Identify the target object, estimate its center pixel [u, v], then call "
        "get_xyz_fused with every camera that can see the object.",
        image_msg,
    )


def handle_get_xyz_single(tool_name, tool_args, d435_cam, d405_cam, robot_ee_pose=None):
    """
    Localizes pixels from one camera using a fresh synchronized capture.

    Single-camera localization is useful when only one camera can see the
    object. Fusion remains preferred when both cameras have a valid view.
    """
    camera_name = "d435" if tool_name == "get_xyz_d435" else "d405"
    coords = tool_args.get("coords", [])
    error = _validate_coord_pairs(coords, "coords", require_nonempty=True)
    if error:
        return error, None

    synced, error = capture_synced_frames(d435_cam, d405_cam)
    if error:
        return f"Error: {error} Single-camera localization still requires synced captures.", None

    frames = frames_for_camera(synced, camera_name)
    if frames.depth_rs is None:
        return f"Error: No depth data available from {camera_name.upper()}.", None

    camera = _camera_from_name(camera_name, d435_cam, d405_cam)
    results = _localize_points(camera_name, coords, frames, camera, robot_ee_pose)
    debug_saved = save_debug_image(frames.rgb, results, camera_name, "single")

    return _json({
        "units": "meters",
        "source_camera": camera_name,
        "snapshot": "D435 and D405 were timestamp-synchronized for this measurement.",
        "points": results,
        "successful_count": sum(1 for result in results if result["status"] == "ok"),
        "debug_image_saved": debug_saved,
    }), None


def handle_get_xyz_fused(tool_args, d435_cam, d405_cam, robot_ee_pose=None):
    """
    Localizes the object from whichever cameras see it, then fuses valid results.

    If both cameras provide valid robot-frame points, translate_points_fused()
    averages their estimates and reports camera disagreement. If only one camera
    can see the object, the same tool still returns that single-camera estimate.
    """
    d435_coords, error = _optional_coords(tool_args, "d435_coords")
    if error:
        return error, None

    d405_coords, error = _optional_coords(tool_args, "d405_coords")
    if error:
        return error, None

    if not d435_coords and not d405_coords:
        return (
            "Error: provide coordinates from at least one camera. Use null only for "
            "a camera that cannot see the object.",
            None,
        )

    synced, error = capture_synced_frames(d435_cam, d405_cam)
    if error:
        return f"Error: {error} Fusion requires synchronized captures.", None

    d435_results = []
    if d435_coords:
        d435_results = _localize_points("d435", d435_coords, synced.d435, d435_cam, robot_ee_pose)

    d405_results = []
    if d405_coords:
        d405_results = _localize_points("d405", d405_coords, synced.d405, d405_cam, robot_ee_pose)

    fusion_result = translate_points_fused(
        d435_points=valid_robot_points(d435_results, "d435"),
        d405_points=valid_robot_points(d405_results, "d405"),
        robot_ee_pose=robot_ee_pose,
    )

    d435_debug = save_debug_image(synced.d435.rgb, d435_results, "d435", "fused")
    d405_debug = save_debug_image(synced.d405.rgb, d405_results, "d405", "fused")

    return _json({
        "units": "meters",
        "snapshot": "D435 and D405 were timestamp-synchronized for this fusion.",
        "fused_result": {
            "xyz": fusion_result.get("xyz"),
            "valid": fusion_result.get("valid", False),
            "sources_used": fusion_result.get("sources_used", []),
            "reason": fusion_result.get("reason"),
            "d435_estimate": fusion_result.get("d435_estimate"),
            "d405_estimate": fusion_result.get("d405_estimate"),
            "camera_disagreement_mm": fusion_result.get("camera_disagreement_mm"),
            "quality_warning": fusion_result.get("quality_warning"),
        },
        "d435": {
            "coords_provided": bool(d435_coords),
            "results": d435_results,
            "successful_count": sum(1 for result in d435_results if result["status"] == "ok"),
            "debug_image_saved": d435_debug,
        },
        "d405": {
            "coords_provided": bool(d405_coords),
            "results": d405_results,
            "successful_count": sum(1 for result in d405_results if result["status"] == "ok"),
            "debug_image_saved": d405_debug,
        },
    }), None


def handle_plan_robot_trajectory(tool_args, robot_ee_pose=None):
    """
    Builds robot-base EE-origin waypoints for the current target position.

    This intentionally does not execute motion. A motion controller should consume
    these waypoints and re-run vision after each move or correction.

    The input target is interpreted as the desired gripper/TCP position. The
    trajectory helper subtracts the configured gripper offset so the returned
    waypoints describe where the Franka EE origin should go.
    """
    try:
        target_xyz = _parse_xyz(tool_args.get("target_xyz"))
        approach_direction = tool_args.get("approach_direction", "z")
        approach_height = tool_args.get("approach_height_m", None)
        if approach_height is None:
            waypoints, metadata = get_robot_trajectory_to_point(
                target_xyz,
                approach_direction=approach_direction,
                robot_ee_pose=robot_ee_pose,
                return_metadata=True,
            )
        else:
            waypoints, metadata = get_robot_trajectory_to_point(
                target_xyz,
                approach_height=float(approach_height),
                approach_direction=approach_direction,
                robot_ee_pose=robot_ee_pose,
                return_metadata=True,
            )
    except (TypeError, ValueError) as exc:
        return f"Error: {exc}", None

    return _json({
        "units": "meters",
        "target_xyz": target_xyz,
        "waypoints": waypoints,
        "trajectory_metadata": metadata,
        "frame": "robot_base",
        "waypoints_are": "end_effector_origin_positions",
        "target_xyz_is": "desired_gripper_tcp_position",
        "execution_status": "planned_only",
        "auto_updates_after_motion": False,
        "update_method": (
            "After robot motion, call get_birds_eye_view/get_eye_in_hand_view as needed, "
            "call get_xyz_fused with fresh pixels, then call plan_robot_trajectory again."
        ),
        "trajectory_update_hint": (
            "Before executing or after any robot motion, capture fresh synced images, "
            "localize again, and re-plan from the updated target estimate."
        ),
    }), None


def handle_execute_robot_waypoints(tool_args, robot_interface=None):
    """
    Executes pre-planned robot-base EE-origin waypoints.

    This is intentionally separate from plan_robot_trajectory. Planning can be
    inspected safely; execution moves hardware and therefore must be requested as
    a distinct tool call. FrankaRobotInterface still enforces workspace checks,
    speed clamping, collision behavior, and its configured confirmation prompt.
    """
    if robot_interface is None:
        return (
            "Error: robot motion is not available because main.py did not provide "
            "a robot interface.",
            None,
        )

    try:
        waypoints = _parse_waypoints(tool_args.get("waypoints"))
        speed_mps = tool_args.get("speed_mps")
        if speed_mps is not None:
            if isinstance(speed_mps, bool):
                raise ValueError("speed_mps must be numeric and positive")
            speed_mps = float(speed_mps)
            if not math.isfinite(speed_mps) or speed_mps <= 0.0:
                raise ValueError("speed_mps must be numeric and positive")
        result = robot_interface.move_to_waypoints(
            waypoints,
            speed_mps=speed_mps,
            source="execute_robot_waypoints_tool",
        )
    except (TypeError, ValueError) as exc:
        return f"Error: {exc}", None
    except RuntimeError as exc:
        return f"Error: robot motion was not executed: {exc}", None
    except Exception as exc:
        return f"Error: robot motion failed: {exc}", None

    return _json({
        "execution_status": result.get("status"),
        "hardware_motion_enabled": result.get("hardware_motion_enabled", False),
        "robot_ip": result.get("robot_ip"),
        "waypoints": result.get("waypoints"),
        "speed_mps": result.get("speed_mps"),
        "motion_summary": result.get("motion_summary"),
        "segments_executed": result.get("segments_executed"),
        "distance_m": result.get("distance_m"),
        "final_xyz": result.get("final_xyz"),
        "warnings": result.get("warnings", []),
        "next_step": (
            "After motion, capture fresh synchronized images, localize again, "
            "and re-plan before making another correction."
        ),
    }), None


def dispatch(tool_name, tool_args, d435_cam, d405_cam, robot_ee_pose=None, robot_interface=None):
    """
    Main entry point called by the LLM interface.

    Returns:
        tuple: (text_result, optional_image_message)
    """
    if not isinstance(tool_args, dict):
        return f"Error: Tool arguments must be a JSON object, got {type(tool_args).__name__}", None
    if not isinstance(tool_name, str):
        return f"Error: Tool name must be a string, got {type(tool_name).__name__}", None

    if tool_name == "get_birds_eye_view":
        return handle_get_camera_view("d435", d435_cam, d405_cam)
    if tool_name == "get_eye_in_hand_view":
        return handle_get_camera_view("d405", d435_cam, d405_cam)
    if tool_name in ("get_xyz_d435", "get_xyz_d405"):
        return handle_get_xyz_single(tool_name, tool_args, d435_cam, d405_cam, robot_ee_pose)
    if tool_name == "get_xyz_fused":
        return handle_get_xyz_fused(tool_args, d435_cam, d405_cam, robot_ee_pose)
    if tool_name == "plan_robot_trajectory":
        return handle_plan_robot_trajectory(tool_args, robot_ee_pose=robot_ee_pose)
    if tool_name == "execute_robot_waypoints":
        return handle_execute_robot_waypoints(tool_args, robot_interface=robot_interface)

    available_tools = [
        "get_birds_eye_view",
        "get_eye_in_hand_view",
        "get_xyz_d435",
        "get_xyz_d405",
        "get_xyz_fused",
        "plan_robot_trajectory",
        "execute_robot_waypoints",
    ]
    return f"Error: Unknown tool '{tool_name}'. Available tools: {available_tools}", None
