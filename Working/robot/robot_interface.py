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

import config as cfg
from policy.actions import PolicyAction


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

    def stop(self) -> Dict[str, Any]:
        return {"status": "stopped_noop"}


class FrankaRobotInterface(RobotInterface):
    """
    Placeholder for a real Franka implementation.

    This class should eventually own the pylibfranka Robot object for the whole
    runtime session. Do not open/close a robot connection for every controller
    timestep.
    """

    def __init__(self, robot_ip: Optional[str] = None) -> None:
        self.robot_ip = robot_ip or cfg.FRANKA_IP
        self.robot: Optional[Any] = None

    def connect(self) -> Dict[str, Any]:
        # TODO: Import pylibfranka here, create the Robot, configure collision
        # behavior from config.py, and keep the connection open until close().
        return {
            "status": "not_implemented",
            "robot_ip": self.robot_ip,
            "message": "Implement persistent pylibfranka connection here.",
        }

    def close(self) -> Dict[str, Any]:
        # TODO: Close the persistent robot connection and gripper connection.
        self.robot = None
        return {"status": "closed_placeholder"}

    def read_state(self) -> RobotState:
        # TODO: Return ee_pose, joints, velocities, force/torque, gripper state.
        raise NotImplementedError("FrankaRobotInterface.read_state() is not implemented yet")

    def execute_action(self, action: PolicyAction) -> Dict[str, Any]:
        # TODO: Convert bounded PolicyAction into a libfranka command.
        raise NotImplementedError("FrankaRobotInterface.execute_action() is not implemented yet")

    def move_to_waypoints(self, waypoints: List[List[float]], **kwargs: Any) -> Dict[str, Any]:
        # TODO: Use a motion generator or Cartesian impedance controller to move
        # through these EE-origin waypoints.
        raise NotImplementedError("FrankaRobotInterface.move_to_waypoints() is not implemented yet")

    def stop(self) -> Dict[str, Any]:
        # TODO: Issue the fastest safe stop supported by the Franka stack.
        return {"status": "stop_placeholder", "message": "Implement hardware stop here."}
