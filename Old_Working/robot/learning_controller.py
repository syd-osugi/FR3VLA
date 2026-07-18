"""
Learning Feedback Controller
----------------------------
Future closed-loop controller that will run a trained policy against live robot
observations.

This module is the integration outline for the eventual learned policy runtime:
observation -> policy -> safety -> robot command -> log.

The class is deliberately conservative today. It can run a single dry-run step
with placeholder components, but real hardware execution requires implementing
RobotInterface and replacing PlaceholderPolicy with a trained model.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import config as cfg
from policy.base import PlaceholderPolicy, PolicyInterface
from policy.dataset import EpisodeLogger
from policy.inference import PolicyInferenceLoop
from policy.observation import ObservationBuilder
from robot.robot_interface import NoOpRobotInterface, RobotInterface
from robot.safety import SafetySupervisor


class LearningFeedbackController:
    """
    Controller shell for learned closed-loop manipulation.

    Expected final behavior:
    - maintain a timed control loop,
    - build fresh observations each tick,
    - run policy inference,
    - pass actions through safety checks,
    - command the robot,
    - stop immediately on safety violation, exception, or user interrupt.
    """

    def __init__(
        self,
        robot_interface: Optional[RobotInterface] = None,
        observation_builder: Optional[ObservationBuilder] = None,
        policy: Optional[PolicyInterface] = None,
        safety_supervisor: Optional[SafetySupervisor] = None,
        episode_logger: Optional[EpisodeLogger] = None,
        control_rate_hz: Optional[float] = None,
    ) -> None:
        self.robot_interface = robot_interface or NoOpRobotInterface()
        self.observation_builder = observation_builder or ObservationBuilder()
        self.policy = policy or PlaceholderPolicy()
        self.safety_supervisor = safety_supervisor or SafetySupervisor()
        self.episode_logger = episode_logger
        self.control_rate_hz = float(control_rate_hz or getattr(cfg, "POLICY_CONTROL_RATE_HZ", 10.0))
        self.running = False

        self.inference_loop = PolicyInferenceLoop(
            observation_builder=self.observation_builder,
            policy=self.policy,
            safety_supervisor=self.safety_supervisor,
            robot_interface=self.robot_interface,
            episode_logger=self.episode_logger,
        )

    def initialize(self) -> Dict[str, Any]:
        connection = self.robot_interface.connect()
        return {
            "status": "initialized",
            "control_rate_hz": self.control_rate_hz,
            "robot_connection": connection,
            "policy": type(self.policy).__name__,
            "hardware_motion_enabled": not isinstance(self.robot_interface, NoOpRobotInterface),
        }

    def run_once(
        self,
        task_instruction: Optional[str] = None,
        execute: bool = False,
    ) -> Dict[str, Any]:
        """
        Run one policy timestep.

        Keep execute=False while wiring perception/training. Set execute=True only
        after RobotInterface and SafetySupervisor have real hardware coverage.
        """

        return self.inference_loop.step(
            task_instruction=task_instruction,
            execute=execute,
        )

    def run_episode(
        self,
        task_instruction: str,
        max_steps: int,
        execute: bool = False,
    ) -> Dict[str, Any]:
        """
        Run a bounded controller episode.

        This simple loop is a placeholder. A real version should use monotonic
        scheduling, richer termination checks, and external stop signals.
        """

        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        self.running = True
        results = []
        step_period_s = 1.0 / self.control_rate_hz

        try:
            for _ in range(max_steps):
                if not self.running:
                    break
                start_s = time.monotonic()
                result = self.run_once(task_instruction=task_instruction, execute=execute)
                results.append(result)

                if result["safety_result"]["approved"] is False:
                    self.stop()
                    break

                elapsed_s = time.monotonic() - start_s
                sleep_s = max(0.0, step_period_s - elapsed_s)
                if sleep_s > 0:
                    time.sleep(sleep_s)
        finally:
            self.running = False

        return {
            "status": "episode_finished",
            "steps": len(results),
            "execute": execute,
            "results": results,
        }

    def stop(self) -> Dict[str, Any]:
        self.running = False
        return self.robot_interface.stop()

    def close(self) -> Dict[str, Any]:
        self.running = False
        return self.robot_interface.close()
