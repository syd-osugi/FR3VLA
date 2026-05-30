"""
Test 19: Policy action contracts, safety filtering, and dry-run inference.
"""

from __future__ import annotations

from _working_test_utils import require, require_close, run_tests

from policy.actions import ACTION_CARTESIAN_DELTA, ACTION_NO_OP, ACTION_WAYPOINT, PolicyAction
from policy.base import PlaceholderPolicy
from policy.inference import PolicyInferenceLoop
from policy.observation import CameraObservation, ObservationBuilder, PolicyObservation, RobotObservation
from policy.scripted import ScriptedTrajectoryPolicy
from robot.robot_interface import NoOpRobotInterface
from robot.safety import SafetyLimits, SafetySupervisor


def test_policy_action_validation():
    action = PolicyAction(action_type=ACTION_CARTESIAN_DELTA, delta_xyz_m=[0.01, 0.0, 0.0]).validate()
    require(action.delta_xyz_m == [0.01, 0.0, 0.0], "valid action was changed unexpectedly")

    try:
        PolicyAction(action_type=ACTION_CARTESIAN_DELTA).validate()
    except ValueError as exc:
        require("require delta_xyz_m" in str(exc), "missing delta error message changed")
    else:
        raise AssertionError("cartesian_delta without delta_xyz_m should raise")

    try:
        PolicyAction(action_type="jump").validate()
    except ValueError as exc:
        require("Unknown action_type" in str(exc), "unknown action error message changed")
    else:
        raise AssertionError("unknown action type should raise")


def test_safety_supervisor_bounds_and_blocks():
    limits = SafetyLimits(
        workspace_min_m=[-1.0, -1.0, 0.0],
        workspace_max_m=[1.0, 1.0, 1.0],
        max_delta_xyz_m=0.02,
        max_delta_rpy_rad=0.10,
        require_robot_state=False,
    )
    supervisor = SafetySupervisor(limits)
    big_delta = PolicyAction(action_type=ACTION_CARTESIAN_DELTA, delta_xyz_m=[0.2, 0.0, 0.0])
    result = supervisor.filter_action(big_delta)
    require(result.approved is True, "cartesian delta should be approved after scaling")
    require_close(result.action.delta_xyz_m, [0.02, 0.0, 0.0], "delta scaling failed")
    require(result.warnings, "scaled action should include a warning")

    outside = PolicyAction(action_type=ACTION_WAYPOINT, target_waypoint_xyz_m=[2.0, 0.0, 0.5])
    result = supervisor.filter_action(outside)
    require(result.approved is False and result.action.action_type == ACTION_NO_OP, "outside waypoint should be blocked")

    requiring_state = SafetySupervisor(SafetyLimits(require_robot_state=True))
    result = requiring_state.filter_action(PolicyAction(action_type=ACTION_NO_OP))
    require(result.approved is False and result.reason == "missing_robot_state", "missing robot state should block")


def test_observation_builder_and_scripted_policy():
    robot = RobotObservation(ee_pose=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    camera = CameraObservation(camera_name="d435", rgb_frame_id="rgb-1")
    builder = ObservationBuilder(
        robot_state_provider=lambda: robot,
        camera_observation_provider=lambda: {"d435": camera},
        target_provider=lambda: [0.1, 0.2, 0.3],
    )
    observation = builder.build("pick object")
    require(observation.robot is robot, "robot provider result not attached")
    require(observation.cameras["d435"] is camera, "camera provider result not attached")
    require(observation.target_xyz_m == [0.1, 0.2, 0.3], "target provider result not attached")

    def fake_planner(target_xyz, **kwargs):
        return ([[0.0, 0.0, 0.2], [0.0, 0.0, 0.1]], {"target": target_xyz, "kwargs": kwargs})

    policy = ScriptedTrajectoryPolicy(planner=fake_planner)
    first = policy.predict(observation)
    second = policy.predict(observation)
    third = policy.predict(observation)
    require(first.action_type == ACTION_WAYPOINT and first.target_waypoint_xyz_m == [0.0, 0.0, 0.2], "first scripted waypoint wrong")
    require(second.action_type == ACTION_WAYPOINT and second.target_waypoint_xyz_m == [0.0, 0.0, 0.1], "second scripted waypoint wrong")
    require(third.action_type == ACTION_NO_OP, "scripted policy should no-op after waypoints")


def test_inference_loop_dry_run():
    builder = ObservationBuilder(target_provider=lambda: [0.0, 0.0, 0.1])
    policy = PlaceholderPolicy()
    supervisor = SafetySupervisor(SafetyLimits(require_robot_state=False))
    robot = NoOpRobotInterface()
    loop = PolicyInferenceLoop(builder, policy, supervisor, robot_interface=robot)

    result = loop.step(task_instruction="dry run", execute=True)
    require(result["raw_action"]["action_type"] == ACTION_NO_OP, "placeholder policy should produce no-op")
    require(result["safety_result"]["approved"] is True, "no-op should pass safety when state is not required")
    require(result["execution_result"]["status"] == "not_executed", "NoOpRobotInterface should never execute hardware")


if __name__ == "__main__":
    raise SystemExit(
        run_tests(
            [
                ("policy action validation", test_policy_action_validation),
                ("safety supervisor bounds and blocks", test_safety_supervisor_bounds_and_blocks),
                ("observation builder and scripted policy", test_observation_builder_and_scripted_policy),
                ("inference loop dry run", test_inference_loop_dry_run),
            ]
        )
    )
