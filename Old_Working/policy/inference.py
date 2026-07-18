"""
Policy Inference Runtime
------------------------
Small runtime wrapper that connects observation building, policy prediction,
safety filtering, optional execution, and optional logging.

This module should stay thin. It is not the robot driver and it is not the model
definition. Its job is orchestration for one controller timestep.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from policy.base import PolicyInterface
from policy.dataset import EpisodeLogger
from policy.observation import ObservationBuilder
from robot.robot_interface import RobotInterface
from robot.safety import SafetySupervisor


class PolicyInferenceLoop:
    """
    Executes one safe policy step at a time.

    A future high-rate controller can call step() in a timed loop. Keeping this
    method single-step makes it easier to test and easier to stop immediately.
    """

    def __init__(
        self,
        observation_builder: ObservationBuilder,
        policy: PolicyInterface,
        safety_supervisor: SafetySupervisor,
        robot_interface: Optional[RobotInterface] = None,
        episode_logger: Optional[EpisodeLogger] = None,
    ) -> None:
        self.observation_builder = observation_builder
        self.policy = policy
        self.safety_supervisor = safety_supervisor
        self.robot_interface = robot_interface
        self.episode_logger = episode_logger

    def step(
        self,
        task_instruction: Optional[str] = None,
        execute: bool = False,
    ) -> Dict[str, Any]:
        observation = self.observation_builder.build(task_instruction=task_instruction)
        raw_action = self.policy.predict(observation)
        safety_result = self.safety_supervisor.filter_action(
            raw_action,
            robot_state=observation.robot,
        )

        execution_result = {
            "status": "not_executed",
            "reason": "execute=False or no robot_interface was provided",
        }
        if execute and self.robot_interface is not None and safety_result.approved:
            execution_result = self.robot_interface.execute_action(safety_result.action)
        elif execute and not safety_result.approved:
            execution_result = {
                "status": "blocked_by_safety",
                "reason": safety_result.reason,
            }

        if self.episode_logger is not None:
            self.episode_logger.record_step(
                observation=observation,
                action=raw_action,
                safety_result=safety_result,
                execution_result=execution_result,
            )

        return {
            "observation": observation.to_dict(),
            "raw_action": raw_action.to_dict(),
            "safety_result": safety_result.to_dict(),
            "execution_result": execution_result,
        }
