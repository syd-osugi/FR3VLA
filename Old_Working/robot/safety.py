"""
Robot Safety Supervisor
-----------------------
Filters policy actions before they reach hardware.

A learned policy should never command the robot directly. This module is the
guardrail between model output and RobotInterface execution.

Expected future work:
- Use calibrated workspace/table bounds from config.py.
- Add collision checking and force/torque stop conditions.
- Reject stale observations and missing camera/robot state.
- Add unit tests for every limit and failure path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional

import config as cfg
from policy.actions import ACTION_CARTESIAN_DELTA, ACTION_NO_OP, ACTION_WAYPOINT, PolicyAction


def _vector_norm(values: List[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _scale_vector(values: List[float], max_norm: float) -> List[float]:
    norm = _vector_norm(values)
    if norm <= max_norm or norm == 0:
        return values
    scale = max_norm / norm
    return [value * scale for value in values]


def _inside_workspace(point: List[float], min_xyz: List[float], max_xyz: List[float]) -> bool:
    return all(min_xyz[idx] <= point[idx] <= max_xyz[idx] for idx in range(3))


@dataclass
class SafetyLimits:
    """Physical safety limits applied before robot execution."""

    workspace_min_m: List[float] = field(default_factory=lambda: [-0.8, -0.8, 0.0])
    workspace_max_m: List[float] = field(default_factory=lambda: [0.8, 0.8, 0.8])
    max_delta_xyz_m: float = 0.02
    max_delta_rpy_rad: float = 0.10
    max_cartesian_speed_mps: float = 0.10
    max_cartesian_accel_mps2: float = 0.25
    require_robot_state: bool = False

    @classmethod
    def from_config(cls) -> "SafetyLimits":
        return cls(
            workspace_min_m=list(getattr(cfg, "ROBOT_WORKSPACE_MIN_M", (-0.8, -0.8, 0.0))),
            workspace_max_m=list(getattr(cfg, "ROBOT_WORKSPACE_MAX_M", (0.8, 0.8, 0.8))),
            max_delta_xyz_m=float(getattr(cfg, "POLICY_MAX_STEP_TRANSLATION_M", 0.02)),
            max_delta_rpy_rad=float(getattr(cfg, "POLICY_MAX_STEP_ROTATION_RAD", 0.10)),
            max_cartesian_speed_mps=float(getattr(cfg, "ROBOT_MAX_CARTESIAN_SPEED_MPS", 0.10)),
            max_cartesian_accel_mps2=float(getattr(cfg, "ROBOT_MAX_CARTESIAN_ACCEL_MPS2", 0.25)),
            require_robot_state=bool(getattr(cfg, "POLICY_REQUIRE_ROBOT_STATE", False)),
        )


@dataclass
class SafetyCheckResult:
    """Result of filtering one policy action."""

    approved: bool
    action: PolicyAction
    reason: str
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "action": self.action.to_dict(),
            "reason": self.reason,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


class SafetySupervisor:
    """Validates and bounds policy actions before execution."""

    def __init__(self, limits: Optional[SafetyLimits] = None) -> None:
        self.limits = limits or SafetyLimits.from_config()

    def filter_action(self, action: PolicyAction, robot_state: Optional[Any] = None) -> SafetyCheckResult:
        warnings: List[str] = []

        try:
            action.validate()
        except ValueError as exc:
            return SafetyCheckResult(
                approved=False,
                action=PolicyAction(action_type=ACTION_NO_OP),
                reason=f"invalid_action: {exc}",
            )

        if self.limits.require_robot_state and robot_state is None:
            return SafetyCheckResult(
                approved=False,
                action=PolicyAction(action_type=ACTION_NO_OP),
                reason="missing_robot_state",
            )

        if action.action_type == ACTION_NO_OP:
            return SafetyCheckResult(
                approved=True,
                action=action,
                reason="no_op_action",
            )

        if action.action_type == ACTION_CARTESIAN_DELTA:
            bounded = PolicyAction.from_dict(action.to_dict())
            if bounded.delta_xyz_m is not None:
                original_norm = _vector_norm(bounded.delta_xyz_m)
                bounded.delta_xyz_m = _scale_vector(
                    bounded.delta_xyz_m,
                    self.limits.max_delta_xyz_m,
                )
                if original_norm > self.limits.max_delta_xyz_m:
                    warnings.append(
                        "delta_xyz_m was scaled down to POLICY_MAX_STEP_TRANSLATION_M"
                    )

            if bounded.delta_rpy_rad is not None:
                original_norm = _vector_norm(bounded.delta_rpy_rad)
                bounded.delta_rpy_rad = _scale_vector(
                    bounded.delta_rpy_rad,
                    self.limits.max_delta_rpy_rad,
                )
                if original_norm > self.limits.max_delta_rpy_rad:
                    warnings.append(
                        "delta_rpy_rad was scaled down to POLICY_MAX_STEP_ROTATION_RAD"
                    )

            return SafetyCheckResult(
                approved=True,
                action=bounded,
                reason="cartesian_delta_within_limits",
                warnings=warnings,
                metadata={
                    "max_delta_xyz_m": self.limits.max_delta_xyz_m,
                    "max_delta_rpy_rad": self.limits.max_delta_rpy_rad,
                },
            )

        if action.action_type == ACTION_WAYPOINT:
            waypoint = action.target_waypoint_xyz_m
            if waypoint is None or not _inside_workspace(
                waypoint,
                self.limits.workspace_min_m,
                self.limits.workspace_max_m,
            ):
                return SafetyCheckResult(
                    approved=False,
                    action=PolicyAction(action_type=ACTION_NO_OP),
                    reason="waypoint_outside_workspace",
                    metadata={
                        "workspace_min_m": self.limits.workspace_min_m,
                        "workspace_max_m": self.limits.workspace_max_m,
                        "requested_waypoint": waypoint,
                    },
                )

            return SafetyCheckResult(
                approved=True,
                action=action,
                reason="waypoint_within_workspace",
            )

        return SafetyCheckResult(
            approved=False,
            action=PolicyAction(action_type=ACTION_NO_OP),
            reason=f"unsupported_action_type: {action.action_type}",
        )
