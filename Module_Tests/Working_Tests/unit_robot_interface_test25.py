"""
Test 25: FrankaRobotInterface command generation with fake pylibfranka objects.

This does not connect to or move hardware. It verifies that robot_interface.py
turns robot-base coordinates into Franka CartesianPose commands using the same
API shape as the Franka example scripts.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from _working_test_utils import patched_attr, require, require_close, require_raises, run_tests

from policy.actions import ACTION_CARTESIAN_DELTA, PolicyAction
from robot import robot_interface


def flat_pose(x=0.0, y=0.0, z=0.0):
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [x, y, z]
    return pose.reshape(16, order="F").tolist()


class FakeDuration:
    def __init__(self, seconds):
        self.seconds = seconds

    def to_sec(self):
        return self.seconds


class FakeState:
    def __init__(self, pose):
        self.O_T_EE = list(pose)
        self.q = [0.0] * 7
        self.dq = [0.0] * 7
        self.O_F_ext_hat_K = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
        self.tau_ext_hat_filtered = [0.0] * 7
        self.robot_mode = "Move"


class FakeCartesianPose:
    def __init__(self, pose):
        self.pose = list(pose)
        self.motion_finished = False


class FakeActiveControl:
    def __init__(self, robot):
        self.robot = robot

    def readOnce(self):
        return FakeState(self.robot.pose), FakeDuration(0.05)

    def writeOnce(self, cartesian_pose):
        self.robot.commands.append(cartesian_pose)
        self.robot.pose = list(cartesian_pose.pose)


class FakeRobot:
    instances = []

    def __init__(self, ip, realtime_config=None):
        self.ip = ip
        self.realtime_config = realtime_config
        self.pose = flat_pose()
        self.commands = []
        self.collision_behavior = None
        self.joint_impedance = None
        self.cartesian_impedance = None
        self.load = None
        self.stopped = False
        self.closed = False
        FakeRobot.instances.append(self)

    def set_collision_behavior(self, lower_torque, upper_torque, lower_force, upper_force):
        self.collision_behavior = (lower_torque, upper_torque, lower_force, upper_force)

    def set_joint_impedance(self, values):
        self.joint_impedance = list(values)

    def set_cartesian_impedance(self, values):
        self.cartesian_impedance = list(values)

    def set_load(self, mass_kg, center_of_mass_m, inertia_kgm2):
        self.load = (float(mass_kg), list(center_of_mass_m), list(inertia_kgm2))

    def read_once(self):
        return FakeState(self.pose)

    def start_cartesian_pose_control(self, controller_mode):
        self.controller_mode = controller_mode
        return FakeActiveControl(self)

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


FAKE_FRANKA = SimpleNamespace(
    Robot=FakeRobot,
    CartesianPose=FakeCartesianPose,
    ControllerMode=SimpleNamespace(JointImpedance="JointImpedance"),
    RealtimeConfig=SimpleNamespace(kIgnore="kIgnore"),
)


def franka_test_context():
    FakeRobot.instances.clear()
    return (
        patched_attr(robot_interface, "_load_pylibfranka", lambda: FAKE_FRANKA),
        patched_attr(robot_interface.cfg, "FRANKA_REQUIRE_MOTION_CONFIRMATION", False),
        patched_attr(robot_interface.cfg, "FRANKA_COLLISION_TORQUE_NM", 11.0),
        patched_attr(robot_interface.cfg, "FRANKA_COLLISION_FORCE_N", 12.0),
        patched_attr(robot_interface.cfg, "FRANKA_LOAD_MASS_KG", 0.75),
        patched_attr(robot_interface.cfg, "FRANKA_LOAD_CENTER_OF_MASS_IN_FLANGE_M", (0.01, 0.02, 0.03)),
        patched_attr(robot_interface.cfg, "FRANKA_LOAD_INERTIA_KGM2", (0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.3)),
        patched_attr(robot_interface.cfg, "ROBOT_WORKSPACE_MIN_M", (-1.0, -1.0, -1.0)),
        patched_attr(robot_interface.cfg, "ROBOT_WORKSPACE_MAX_M", (1.0, 1.0, 1.0)),
        patched_attr(robot_interface.cfg, "ROBOT_MAX_CARTESIAN_SPEED_MPS", 0.5),
        patched_attr(robot_interface.cfg, "FRANKA_CARTESIAN_MOTION_TOLERANCE_M", 1e-6),
        patched_attr(robot_interface.cfg, "FRANKA_MIN_CARTESIAN_SEGMENT_DURATION_S", 0.01),
        patched_attr(robot_interface.cfg, "GRIPPER_TCP_IN_EE_TRANSLATION_M", (0.0, 0.0, 0.0)),
        patched_attr(robot_interface.cfg, "GRIPPER_TCP_IN_EE_RPY_DEG", (0.0, 0.0, 0.0)),
    )


class nested_context:
    def __init__(self, contexts):
        self.contexts = contexts

    def __enter__(self):
        for context in self.contexts:
            context.__enter__()

    def __exit__(self, exc_type, exc, traceback):
        for context in reversed(self.contexts):
            context.__exit__(exc_type, exc, traceback)


def test_connect_and_read_state():
    with nested_context(franka_test_context()):
        interface = robot_interface.FrankaRobotInterface(
            robot_ip="fake-robot",
            require_motion_confirmation=False,
        )
        connection = interface.connect()
        require(connection["status"] == "connected", "Franka interface did not connect")
        require(connection["hardware_motion_enabled"] is True, "hardware flag should be true")

        fake_robot = FakeRobot.instances[-1]
        require(fake_robot.realtime_config == "kIgnore", "RealtimeConfig.kIgnore should be used when available")
        require(fake_robot.collision_behavior[0] == [11.0] * 7, "collision torque threshold not configured")
        require(fake_robot.collision_behavior[2] == [12.0] * 6, "collision force threshold not configured")
        require(fake_robot.joint_impedance is not None, "joint impedance should be configured when available")
        require(fake_robot.cartesian_impedance is not None, "cartesian impedance should be configured when available")
        require(fake_robot.load[0] == 0.75, "payload mass was not configured")
        require(fake_robot.load[1] == [0.01, 0.02, 0.03], "payload center of mass was not configured")
        require(fake_robot.load[2] == [0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.3], "payload inertia was not configured")
        require(connection["franka_control_config"]["load"]["applied"] is True, "connection result should report applied payload")

        state = interface.read_state()
        require_close(np.array(state.ee_pose)[:3, 3], [0.0, 0.0, 0.0], "state pose translation wrong")
        require(state.measured_force_n == [1.0, 2.0, 3.0], "force conversion failed")
        require(state.measured_torque_nm == [0.1, 0.2, 0.3], "torque conversion failed")


def test_move_to_waypoints_commands_cartesian_pose():
    with nested_context(franka_test_context()):
        interface = robot_interface.FrankaRobotInterface(
            robot_ip="fake-robot",
            require_motion_confirmation=False,
        )
        result = interface.move_to_waypoints(
            [[0.1, 0.0, 0.2]],
            speed_mps=0.25,
            require_confirmation=False,
        )

        fake_robot = FakeRobot.instances[-1]
        require(result["status"] == "executed", "waypoint motion should execute")
        require(result["motion_summary"] == "cartesian_waypoints_complete", "motion summary changed")
        require(result["segments_executed"] == 1, "expected one motion segment")
        require(fake_robot.commands, "no CartesianPose commands were written")
        require(fake_robot.commands[-1].motion_finished is True, "final command should finish motion")
        require_close(fake_robot.pose[12:15], [0.1, 0.0, 0.2], "final robot pose command wrong")


def test_execute_delta_and_move_to_target_xyz():
    with nested_context(franka_test_context()):
        interface = robot_interface.FrankaRobotInterface(
            robot_ip="fake-robot",
            require_motion_confirmation=False,
        )
        action = PolicyAction(
            action_type=ACTION_CARTESIAN_DELTA,
            delta_xyz_m=[0.0, 0.0, 0.1],
        )
        result = interface.execute_action(action)
        fake_robot = FakeRobot.instances[-1]
        require(result["status"] == "executed", "cartesian delta action should execute")
        require_close(fake_robot.pose[12:15], [0.0, 0.0, 0.1], "delta action final pose wrong")

        result = interface.move_to_target_xyz(
            [0.1, 0.0, 0.2],
            approach_height=0.05,
            require_confirmation=False,
        )
        require(result["source"] == "move_to_target_xyz", "target movement source missing")
        require(result["trajectory_metadata"]["target_xyz_is"] == "desired_gripper_tcp_position_in_robot_base", "trajectory metadata missing")
        require_close(fake_robot.pose[12:15], [0.1, 0.0, 0.2], "target motion final pose wrong")


def test_workspace_rejection():
    with nested_context(franka_test_context()):
        interface = robot_interface.FrankaRobotInterface(
            robot_ip="fake-robot",
            require_motion_confirmation=False,
        )
        require_raises(
            ValueError,
            lambda: interface.move_to_waypoints([[2.0, 0.0, 0.0]], require_confirmation=False),
            "outside workspace waypoint should be rejected",
        )


if __name__ == "__main__":
    raise SystemExit(
        run_tests(
            [
                ("connect and read state", test_connect_and_read_state),
                ("move to waypoints commands CartesianPose", test_move_to_waypoints_commands_cartesian_pose),
                ("execute delta and move to target xyz", test_execute_delta_and_move_to_target_xyz),
                ("workspace rejection", test_workspace_rejection),
            ]
        )
    )
