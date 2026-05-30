"""
Scripted Policy Adapters
------------------------
Executable proof-of-concept policies that use the same interface a trained
policy will use later.

The important idea is replaceability:
- Today: ScriptedTrajectoryPolicy wraps robot.trajectory.get_robot_trajectory_to_point().
- Later: A learned policy can replace this class while keeping the controller,
  safety checks, robot interface, and logging structure intact.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from policy.actions import ACTION_WAYPOINT, PolicyAction, no_op_action
from policy.observation import PolicyObservation
from robot.trajectory import get_robot_trajectory_to_point


class ScriptedTrajectoryPolicy:
    """
    PolicyInterface-compatible adapter around the existing trajectory planner.

    This is not learned. It turns observation.target_xyz_m into a sequence of
    waypoint actions. That makes the new policy/controller path executable now
    while preserving the module boundary a trained policy will use later.
    """

    def __init__(
        self,
        approach_direction: str = "z",
        approach_height_m: Optional[float] = None,
        planner: Optional[Callable[..., Tuple[List[List[float]], Dict[str, Any]]]] = None,
    ) -> None:
        self.approach_direction = approach_direction
        self.approach_height_m = approach_height_m
        self.planner = planner or get_robot_trajectory_to_point
        self.latest_target: Optional[List[float]] = None
        self.pending_waypoints: List[List[float]] = []
        self.latest_metadata: Dict[str, Any] = {}
        self.checkpoint_path: Optional[str] = None

    def load(self, checkpoint_path: str) -> None:
        """
        Kept for PolicyInterface compatibility.

        A scripted policy has no weights. The argument is recorded so callers can
        use the same lifecycle as a learned policy without special casing.
        """

        self.checkpoint_path = checkpoint_path

    def predict(self, observation: PolicyObservation) -> PolicyAction:
        target_xyz = observation.target_xyz_m
        if target_xyz is None:
            return no_op_action(
                "ScriptedTrajectoryPolicy needs observation.target_xyz_m before it can plan."
            )

        target_xyz = [float(value) for value in target_xyz]
        if target_xyz != self.latest_target:
            self._replan(target_xyz, observation)

        if not self.pending_waypoints:
            return no_op_action("Scripted trajectory for the current target is complete.")

        waypoint = self.pending_waypoints.pop(0)
        return PolicyAction(
            action_type=ACTION_WAYPOINT,
            target_waypoint_xyz_m=waypoint,
            metadata={
                "policy": type(self).__name__,
                "source": "robot.trajectory.get_robot_trajectory_to_point",
                "remaining_waypoints": len(self.pending_waypoints),
                "trajectory_metadata": self.latest_metadata,
            },
        ).validate()

    def _replan(self, target_xyz: List[float], observation: PolicyObservation) -> None:
        robot_ee_pose = None
        if observation.robot is not None:
            robot_ee_pose = observation.robot.ee_pose

        planner_kwargs: Dict[str, Any] = {
            "approach_direction": self.approach_direction,
            "robot_ee_pose": robot_ee_pose,
            "return_metadata": True,
        }
        if self.approach_height_m is not None:
            planner_kwargs["approach_height"] = self.approach_height_m

        waypoints, metadata = self.planner(target_xyz, **planner_kwargs)
        self.latest_target = target_xyz
        self.pending_waypoints = waypoints
        self.latest_metadata = metadata
