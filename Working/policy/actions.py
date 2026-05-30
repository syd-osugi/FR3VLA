"""
Policy Action Contracts
-----------------------
Defines the action format a trained policy will output at runtime.

Keep this module model-agnostic. A neural network, behavior cloning policy, or
hand-written baseline should all return the same PolicyAction object so the
robot controller and safety supervisor do not need to know how the action was
produced.

Expected future work:
- Decide the final low-level action space used for training.
- Add gripper-specific commands once the gripper hardware API is selected.
- Add conversions from normalized model outputs to physical units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional


ACTION_CARTESIAN_DELTA = "cartesian_delta"
ACTION_WAYPOINT = "waypoint"
ACTION_NO_OP = "no_op"

GRIPPER_OPEN = "open"
GRIPPER_CLOSE = "close"
GRIPPER_HOLD = "hold"


def _coerce_vector(value: Optional[List[float]], name: str, expected_len: int) -> Optional[List[float]]:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != expected_len:
        raise ValueError(f"{name} must be a list of {expected_len} numbers")
    coerced = [float(item) for item in value]
    if not all(math.isfinite(item) for item in coerced):
        raise ValueError(f"{name} must contain only finite numbers")
    return coerced


@dataclass
class PolicyAction:
    """
    One policy command for a single controller timestep.

    Recommended first action space:
        action_type="cartesian_delta"
        delta_xyz_m=[dx, dy, dz]
        delta_rpy_rad=[droll, dpitch, dyaw] or None while orientation is fixed

    A waypoint action is useful for baselines or scripted policies, but a learned
    policy usually behaves better when it outputs small bounded deltas.
    """

    action_type: str = ACTION_NO_OP
    delta_xyz_m: Optional[List[float]] = None
    delta_rpy_rad: Optional[List[float]] = None
    target_waypoint_xyz_m: Optional[List[float]] = None
    gripper_command: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "PolicyAction":
        if self.action_type not in {ACTION_CARTESIAN_DELTA, ACTION_WAYPOINT, ACTION_NO_OP}:
            raise ValueError(f"Unknown action_type: {self.action_type}")

        self.delta_xyz_m = _coerce_vector(self.delta_xyz_m, "delta_xyz_m", 3)
        self.delta_rpy_rad = _coerce_vector(self.delta_rpy_rad, "delta_rpy_rad", 3)
        self.target_waypoint_xyz_m = _coerce_vector(
            self.target_waypoint_xyz_m,
            "target_waypoint_xyz_m",
            3,
        )

        if self.gripper_command is not None and self.gripper_command not in {
            GRIPPER_OPEN,
            GRIPPER_CLOSE,
            GRIPPER_HOLD,
        }:
            raise ValueError(
                "gripper_command must be one of: "
                f"{GRIPPER_OPEN}, {GRIPPER_CLOSE}, {GRIPPER_HOLD}"
            )

        if self.action_type == ACTION_CARTESIAN_DELTA and self.delta_xyz_m is None:
            raise ValueError("cartesian_delta actions require delta_xyz_m")
        if self.action_type == ACTION_WAYPOINT and self.target_waypoint_xyz_m is None:
            raise ValueError("waypoint actions require target_waypoint_xyz_m")

        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "delta_xyz_m": self.delta_xyz_m,
            "delta_rpy_rad": self.delta_rpy_rad,
            "target_waypoint_xyz_m": self.target_waypoint_xyz_m,
            "gripper_command": self.gripper_command,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyAction":
        return cls(
            action_type=data.get("action_type", ACTION_NO_OP),
            delta_xyz_m=data.get("delta_xyz_m"),
            delta_rpy_rad=data.get("delta_rpy_rad"),
            target_waypoint_xyz_m=data.get("target_waypoint_xyz_m"),
            gripper_command=data.get("gripper_command"),
            metadata=data.get("metadata", {}),
        ).validate()


def no_op_action(reason: str = "No policy action requested.") -> PolicyAction:
    return PolicyAction(
        action_type=ACTION_NO_OP,
        metadata={"reason": reason},
    )
