"""
Robot Interface Layer
---------------------
Defines the boundary between controllers/policies and physical robot hardware.

The current project reads a Franka pose in main.py but does not have a persistent
motion command layer. A working feedback controller should use an implementation
of RobotInterface instead of calling pylibfranka directly from policy code.

Expected future work:
- Implement a persistent Franka connection with lifecycle management.
- Add gripper support.
- Convert PolicyAction objects into libfranka motion commands.
- Report commanded vs measured motion for logging and debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional

import numpy as np

import config as cfg
from policy.actions import (
    ACTION_CARTESIAN_DELTA,
    ACTION_NO_OP,
    ACTION_WAYPOINT,
    PolicyAction,
)
from robot.franka_setup import apply_franka_control_config
from robot.trajectory import get_robot_trajectory_to_point


@dataclass
class RobotState:
    """Snapshot of robot state used by controllers and dataset logging."""

    timestamp_s: float = field(default_factory=time.time)
    ee_pose: Optional[List[List[float]]] = None
    joint_positions: Optional[List[float]] = None
    joint_velocities: Optional[List[float]] = None
    gripper_width_m: Optional[float] = None
    measured_force_n: Optional[List[float]] = None
    measured_torque_nm: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_s": self.timestamp_s,
            "ee_pose": self.ee_pose,
            "joint_positions": self.joint_positions,
            "joint_velocities": self.joint_velocities,
            "gripper_width_m": self.gripper_width_m,
            "measured_force_n": self.measured_force_n,
            "measured_torque_nm": self.measured_torque_nm,
            "metadata": self.metadata,
        }


def _load_pylibfranka():
    try:
        import pylibfranka
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pylibfranka is required for FrankaRobotInterface hardware motion."
        ) from exc
    return pylibfranka


def _coerce_xyz(point: Any, name: str = "point") -> np.ndarray:
    try:
        xyz = np.array(point, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric [x, y, z] point") from exc

    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise ValueError(f"{name} must be a finite [x, y, z] point, got: {point}")
    return xyz


def _workspace_bounds() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array(cfg.ROBOT_WORKSPACE_MIN_M, dtype=float),
        np.array(cfg.ROBOT_WORKSPACE_MAX_M, dtype=float),
    )


def _validate_workspace_point(point: np.ndarray, name: str) -> None:
    workspace_min, workspace_max = _workspace_bounds()
    if not np.all((workspace_min <= point) & (point <= workspace_max)):
        raise ValueError(
            f"{name} {point.tolist()} is outside ROBOT_WORKSPACE bounds: "
            f"min={workspace_min.tolist()}, max={workspace_max.tolist()}"
        )


def _franka_pose_list_to_matrix(pose_values: Any) -> List[List[float]]:
    pose_array = np.array(pose_values, dtype=float)
    if pose_array.shape != (16,):
        raise ValueError(f"Expected Franka O_T_EE as 16 values, got shape {pose_array.shape}")
    return pose_array.reshape((4, 4), order="F").tolist()


def _extract_xyz_from_franka_pose(pose_values: Any) -> np.ndarray:
    pose_array = np.array(pose_values, dtype=float)
    if pose_array.shape != (16,):
        raise ValueError(f"Expected Franka O_T_EE as 16 values, got shape {pose_array.shape}")
    return pose_array[[12, 13, 14]]


def _duration_to_seconds(duration: Any) -> float:
    to_sec = getattr(duration, "to_sec", None)
    if callable(to_sec):
        return float(to_sec())
    return float(duration)


def _smoothstep(alpha: float) -> float:
    bounded = max(0.0, min(1.0, alpha))
    return bounded * bounded * (3.0 - 2.0 * bounded)


class RobotInterface:
    """
    Base class for robot hardware adapters.

    Subclasses should make stop() fast and reliable. Controllers should call
    stop() from exception handlers and external interrupt paths.
    """

    def connect(self) -> Dict[str, Any]:
        raise NotImplementedError

    def close(self) -> Dict[str, Any]:
        raise NotImplementedError

    def read_state(self) -> RobotState:
        raise NotImplementedError

    def execute_action(self, action: PolicyAction) -> Dict[str, Any]:
        raise NotImplementedError

    def move_to_waypoints(self, waypoints: List[List[float]], **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def move_to_target_xyz(self, target_xyz: List[float], **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def stop(self) -> Dict[str, Any]:
        raise NotImplementedError


class NoOpRobotInterface(RobotInterface):
    """
    Safe placeholder implementation for dry-runs and wiring tests.

    It accepts calls, records the latest action, and never commands hardware.
    """

    def __init__(self) -> None:
        self.connected = False
        self.latest_action: Optional[PolicyAction] = None

    def connect(self) -> Dict[str, Any]:
        self.connected = True
        return {"status": "connected_noop", "hardware_motion_enabled": False}

    def close(self) -> Dict[str, Any]:
        self.connected = False
        return {"status": "closed_noop"}

    def read_state(self) -> RobotState:
        return RobotState(
            metadata={
                "status": "noop_state",
                "todo": "Replace NoOpRobotInterface with FrankaRobotInterface for real motion.",
            }
        )

    def execute_action(self, action: PolicyAction) -> Dict[str, Any]:
        self.latest_action = action
        return {
            "status": "not_executed",
            "hardware_motion_enabled": False,
            "action": action.to_dict(),
        }

    def move_to_waypoints(self, waypoints: List[List[float]], **kwargs: Any) -> Dict[str, Any]:
        return {
            "status": "not_executed",
            "hardware_motion_enabled": False,
            "waypoints": waypoints,
            "kwargs": kwargs,
        }

    def move_to_target_xyz(self, target_xyz: List[float], **kwargs: Any) -> Dict[str, Any]:
        return {
            "status": "not_executed",
            "hardware_motion_enabled": False,
            "target_xyz": target_xyz,
            "kwargs": kwargs,
        }

    def stop(self) -> Dict[str, Any]:
        return {"status": "stopped_noop"}


class FrankaRobotInterface(RobotInterface):
    """
    Franka robot adapter for robot-base Cartesian XYZ commands.

    The motion methods preserve the current EE orientation and only command the
    translation elements of the Franka O_T_EE pose. Waypoints are interpreted as
    end-effector origin positions in the robot base frame. For object positions
    returned by the cameras/LLM, call move_to_target_xyz(); it uses the existing
    trajectory planner to generate approach and target EE waypoints.
    """

    def __init__(
        self,
        robot_ip: Optional[str] = None,
        require_motion_confirmation: Optional[bool] = None,
    ) -> None:
        self.robot_ip = robot_ip or cfg.FRANKA_IP
        self.robot: Optional[Any] = None
        self._franka: Optional[Any] = None
        self.control_config: Optional[Dict[str, Any]] = None
        self.require_motion_confirmation = (
            bool(require_motion_confirmation)
            if require_motion_confirmation is not None
            else bool(getattr(cfg, "FRANKA_REQUIRE_MOTION_CONFIRMATION", True))
        )

    def connect(self) -> Dict[str, Any]:
        if self.robot is not None:
            return {
                "status": "already_connected",
                "robot_ip": self.robot_ip,
                "hardware_motion_enabled": True,
                "franka_control_config": self.control_config,
            }

        self._franka = _load_pylibfranka()
        realtime_config = getattr(getattr(self._franka, "RealtimeConfig", None), "kIgnore", None)
        if realtime_config is None:
            self.robot = self._franka.Robot(self.robot_ip)
        else:
            self.robot = self._franka.Robot(self.robot_ip, realtime_config)

        self._configure_default_behavior()
        return {
            "status": "connected",
            "robot_ip": self.robot_ip,
            "hardware_motion_enabled": True,
            "require_motion_confirmation": self.require_motion_confirmation,
            "franka_control_config": self.control_config,
        }

    def close(self) -> Dict[str, Any]:
        close = getattr(self.robot, "close", None) if self.robot is not None else None
        if callable(close):
            close()
        self.robot = None
        self.control_config = None
        return {"status": "closed", "robot_ip": self.robot_ip}

    def read_state(self) -> RobotState:
        robot = self._require_robot()
        state = robot.read_once()
        return self._convert_franka_state(state)

    def execute_action(self, action: PolicyAction) -> Dict[str, Any]:
        try:
            action = action.validate()
        except ValueError as exc:
            return {"status": "rejected", "reason": f"invalid_action: {exc}"}

        if action.action_type == ACTION_NO_OP:
            return {
                "status": "no_op",
                "hardware_motion_enabled": True,
                "action": action.to_dict(),
            }

        warnings: List[str] = []
        if action.delta_rpy_rad is not None:
            warnings.append(
                "delta_rpy_rad was ignored; FrankaRobotInterface currently preserves EE orientation."
            )
        if action.gripper_command is not None:
            warnings.append(
                "gripper_command was ignored; gripper hardware is not wired in RobotInterface yet."
            )

        if action.action_type == ACTION_WAYPOINT:
            result = self.move_to_waypoints(
                [action.target_waypoint_xyz_m],
                source="policy_waypoint_action",
            )
        elif action.action_type == ACTION_CARTESIAN_DELTA:
            current_state = self.read_state()
            if current_state.ee_pose is None:
                return {"status": "failed", "reason": "Could not read current EE pose"}

            current_pose = np.array(current_state.ee_pose, dtype=float)
            current_xyz = current_pose[:3, 3]
            target_xyz = current_xyz + np.array(action.delta_xyz_m, dtype=float)
            result = self.move_to_waypoints(
                [target_xyz.tolist()],
                source="policy_cartesian_delta_action",
            )
        else:
            return {
                "status": "rejected",
                "reason": f"unsupported_action_type: {action.action_type}",
            }

        if warnings:
            result.setdefault("warnings", []).extend(warnings)
        result["action"] = action.to_dict()
        return result

    def move_to_waypoints(self, waypoints: List[List[float]], **kwargs: Any) -> Dict[str, Any]:
        robot = self._require_robot()
        franka = self._require_franka()

        clean_waypoints = self._coerce_waypoints(waypoints)
        speed_mps, speed_warning = self._resolve_speed(kwargs.get("speed_mps"))
        require_confirmation = bool(
            kwargs.get("require_confirmation", self.require_motion_confirmation)
        )

        if require_confirmation:
            self._confirm_motion(clean_waypoints, speed_mps)

        try:
            motion_result = self._run_cartesian_waypoints(
                robot=robot,
                franka=franka,
                waypoints=clean_waypoints,
                speed_mps=speed_mps,
            )
        except Exception:
            self.stop()
            raise

        result = {
            "status": "executed",
            "hardware_motion_enabled": True,
            "robot_ip": self.robot_ip,
            "waypoints": [waypoint.tolist() for waypoint in clean_waypoints],
            "speed_mps": speed_mps,
            "source": kwargs.get("source", "move_to_waypoints"),
            **motion_result,
        }
        if speed_warning is not None:
            result["warnings"] = [speed_warning]
        return result

    def move_to_target_xyz(self, target_xyz: List[float], **kwargs: Any) -> Dict[str, Any]:
        target = _coerce_xyz(target_xyz, "target_xyz")
        _validate_workspace_point(target, "target_xyz")

        robot_ee_pose = kwargs.get("robot_ee_pose")
        if robot_ee_pose is None:
            robot_ee_pose = self.read_state().ee_pose

        planner_kwargs: Dict[str, Any] = {
            "robot_ee_pose": robot_ee_pose,
            "return_metadata": True,
        }
        if "approach_height" in kwargs:
            planner_kwargs["approach_height"] = kwargs["approach_height"]
        if "approach_direction" in kwargs:
            planner_kwargs["approach_direction"] = kwargs["approach_direction"]

        waypoints, metadata = get_robot_trajectory_to_point(
            target.tolist(),
            **planner_kwargs,
        )
        result = self.move_to_waypoints(
            waypoints,
            speed_mps=kwargs.get("speed_mps"),
            require_confirmation=kwargs.get(
                "require_confirmation",
                self.require_motion_confirmation,
            ),
            source="move_to_target_xyz",
        )
        result["target_xyz"] = target.tolist()
        result["trajectory_metadata"] = metadata
        return result

    def stop(self) -> Dict[str, Any]:
        if self.robot is None:
            return {"status": "not_connected"}
        stop = getattr(self.robot, "stop", None)
        if not callable(stop):
            return {"status": "stop_unavailable"}
        stop()
        return {"status": "stopped"}

    def _require_robot(self) -> Any:
        if self.robot is None:
            self.connect()
        if self.robot is None:
            raise RuntimeError("Franka robot is not connected")
        return self.robot

    def _require_franka(self) -> Any:
        if self._franka is None:
            self._franka = _load_pylibfranka()
        return self._franka

    def _configure_default_behavior(self) -> None:
        robot = self._require_robot()
        self.control_config = apply_franka_control_config(robot)

    def _convert_franka_state(self, state: Any) -> RobotState:
        wrench = getattr(state, "O_F_ext_hat_K", None)
        wrench_values = list(wrench) if wrench is not None else None

        return RobotState(
            ee_pose=_franka_pose_list_to_matrix(getattr(state, "O_T_EE")),
            joint_positions=list(getattr(state, "q", [])) or None,
            joint_velocities=list(getattr(state, "dq", [])) or None,
            measured_force_n=wrench_values[:3] if wrench_values is not None else None,
            measured_torque_nm=wrench_values[3:] if wrench_values is not None else None,
            metadata={
                "robot_mode": str(getattr(state, "robot_mode", "unknown")),
                "tau_ext_hat_filtered": list(getattr(state, "tau_ext_hat_filtered", [])),
            },
        )

    def _coerce_waypoints(self, waypoints: Any) -> List[np.ndarray]:
        if not isinstance(waypoints, list) or len(waypoints) == 0:
            raise ValueError("waypoints must be a non-empty list of [x, y, z] points")

        clean_waypoints = []
        for index, waypoint in enumerate(waypoints):
            xyz = _coerce_xyz(waypoint, f"waypoint[{index}]")
            _validate_workspace_point(xyz, f"waypoint[{index}]")
            clean_waypoints.append(xyz)
        return clean_waypoints

    def _resolve_speed(self, requested_speed: Optional[Any]) -> tuple[float, Optional[str]]:
        max_speed = float(cfg.ROBOT_MAX_CARTESIAN_SPEED_MPS)
        if requested_speed is None:
            return max_speed, None

        try:
            speed = float(requested_speed)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"speed_mps must be numeric, got: {requested_speed}") from exc

        if speed <= 0 or not np.isfinite(speed):
            raise ValueError(f"speed_mps must be finite and positive, got: {requested_speed}")
        if speed <= max_speed:
            return speed, None
        return max_speed, f"speed_mps was clamped to ROBOT_MAX_CARTESIAN_SPEED_MPS={max_speed}"

    def _confirm_motion(self, waypoints: List[np.ndarray], speed_mps: float) -> None:
        print("WARNING: FrankaRobotInterface will move the robot.")
        print("Keep the user stop button within reach.")
        print(f"Robot IP: {self.robot_ip}")
        print(f"Speed: {speed_mps:.4f} m/s")
        print(f"Waypoints: {[waypoint.tolist() for waypoint in waypoints]}")
        answer = input("Type 'move' to execute this motion: ").strip().lower()
        if answer != "move":
            raise RuntimeError("Motion cancelled by operator")

    def _run_cartesian_waypoints(
        self,
        robot: Any,
        franka: Any,
        waypoints: List[np.ndarray],
        speed_mps: float,
    ) -> Dict[str, Any]:
        initial_state = robot.read_once()
        base_pose = list(initial_state.O_T_EE)
        current_xyz = _extract_xyz_from_franka_pose(base_pose)
        motion_tolerance_m = float(getattr(cfg, "FRANKA_CARTESIAN_MOTION_TOLERANCE_M", 1e-4))
        min_segment_duration_s = float(getattr(cfg, "FRANKA_MIN_CARTESIAN_SEGMENT_DURATION_S", 0.10))

        segments = []
        segment_start_xyz = current_xyz.copy()
        for target_xyz in waypoints:
            delta = target_xyz - segment_start_xyz
            distance = float(np.linalg.norm(delta))
            if distance <= motion_tolerance_m:
                segment_start_xyz = target_xyz.copy()
                continue
            segments.append((segment_start_xyz.copy(), target_xyz.copy(), delta, distance))
            segment_start_xyz = target_xyz.copy()

        if not segments:
            return {
                "motion_summary": "already_at_target",
                "segments_executed": 0,
                "distance_m": 0.0,
                "final_xyz": segment_start_xyz.tolist(),
            }

        active_control = robot.start_cartesian_pose_control(franka.ControllerMode.JointImpedance)

        segment_count = 0
        total_distance_m = 0.0
        final_command_xyz = current_xyz.copy()

        for segment_index, (segment_start_xyz, target_xyz, delta, distance) in enumerate(segments):
            segment_count += 1
            total_distance_m += distance
            segment_duration_s = max(distance / speed_mps, min_segment_duration_s)
            elapsed_s = 0.0
            segment_finished = False

            while not segment_finished:
                robot_state, duration = active_control.readOnce()
                elapsed_s += max(0.0, _duration_to_seconds(duration))
                alpha = min(1.0, elapsed_s / segment_duration_s)
                command_xyz = segment_start_xyz + _smoothstep(alpha) * delta

                command_pose = base_pose.copy()
                command_pose[12] = float(command_xyz[0])
                command_pose[13] = float(command_xyz[1])
                command_pose[14] = float(command_xyz[2])

                cartesian_pose = franka.CartesianPose(command_pose)
                if alpha >= 1.0:
                    final_command_xyz = target_xyz.copy()
                    segment_finished = True
                    if segment_index == len(segments) - 1:
                        cartesian_pose.motion_finished = True

                active_control.writeOnce(cartesian_pose)

        return {
            "motion_summary": "cartesian_waypoints_complete",
            "segments_executed": segment_count,
            "distance_m": total_distance_m,
            "final_xyz": final_command_xyz.tolist(),
        }
