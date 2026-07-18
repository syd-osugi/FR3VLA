"""
Executable Policy Runtime Proof of Concept
------------------------------------------
Runs the future policy/controller architecture using today's trajectory planner.

Example:
    python3 examples/policy_poc.py --target 0.4 0.1 0.2 --steps 3

This does not move hardware. It uses NoOpRobotInterface by default, so execution
results show what would be sent through the controller path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policy.observation import ObservationBuilder, RobotObservation
from policy.scripted import ScriptedTrajectoryPolicy
from robot.learning_controller import LearningFeedbackController
from robot.robot_interface import NoOpRobotInterface
from robot.safety import SafetySupervisor


def _identity_ee_pose() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _build_robot_observation() -> RobotObservation:
    return RobotObservation(
        ee_pose=_identity_ee_pose(),
        metadata={
            "source": "examples.policy_poc",
            "note": (
                "Identity EE pose is only for proof-of-concept planning. "
                "Use a fresh Franka pose before real execution."
            ),
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a no-hardware policy/controller POC with the scripted trajectory policy."
    )
    parser.add_argument(
        "--target",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        required=True,
        help="Desired gripper/TCP target in robot base frame, meters.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3,
        help="Maximum controller steps to run.",
    )
    parser.add_argument(
        "--approach-direction",
        choices=["x", "y", "z"],
        default="z",
        help="Axis used by the scripted trajectory planner for the approach waypoint.",
    )
    parser.add_argument(
        "--approach-height-m",
        type=float,
        default=None,
        help="Optional approach offset in meters.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Send actions to the configured robot interface. With this POC runner, "
            "the interface is still NoOpRobotInterface, so no hardware moves."
        ),
    )
    return parser


def summarize_step(step_result: Dict) -> Dict:
    raw_action = step_result["raw_action"]
    safety_result = step_result["safety_result"]
    execution_result = step_result["execution_result"]

    return {
        "raw_action_type": raw_action["action_type"],
        "target_waypoint_xyz_m": raw_action.get("target_waypoint_xyz_m"),
        "safety_approved": safety_result["approved"],
        "safety_reason": safety_result["reason"],
        "execution_status": execution_result["status"],
    }


def main() -> int:
    args = build_arg_parser().parse_args()

    target_xyz = list(args.target)
    observation_builder = ObservationBuilder(
        robot_state_provider=_build_robot_observation,
        target_provider=lambda: target_xyz,
    )
    policy = ScriptedTrajectoryPolicy(
        approach_direction=args.approach_direction,
        approach_height_m=args.approach_height_m,
    )
    controller = LearningFeedbackController(
        robot_interface=NoOpRobotInterface(),
        observation_builder=observation_builder,
        policy=policy,
        safety_supervisor=SafetySupervisor(),
    )

    print(json.dumps(controller.initialize(), indent=2))

    for step_idx in range(args.steps):
        result = controller.run_once(
            task_instruction="proof-of-concept scripted trajectory",
            execute=args.execute,
        )
        summary = summarize_step(result)
        summary["step"] = step_idx
        print(json.dumps(summary, indent=2))

        if summary["raw_action_type"] == "no_op":
            break

    print(json.dumps(controller.close(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
