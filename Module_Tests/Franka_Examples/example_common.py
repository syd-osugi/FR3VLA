"""
Shared utilities for Franka robot examples.

Provides common helpers used across all Franka_Examples scripts:
- setDefaultBehaviour(): configures collision thresholds and joint/cartesian impedance parameters.
- MotionGenerator: a synchronized minimum-jerk trajectory generator that computes time-parameterized
  joint position targets between a start and goal configuration, respecting per-axis velocity and
  acceleration limits.
"""

from typing import List, Tuple

import numpy as np

from pylibfranka import JointPositions, Robot, RobotState


def setDefaultBehaviour(robot: Robot):
    """Apply default collision behavior and impedance settings to the robot.

    These parameters define how the robot reacts to unexpected forces (collision thresholds)
    and the stiffness of its joints/cartesian axes during compliant control modes.
    Must be called before starting any control loop.
    """
    # Torque thresholds per joint (Nm): below lower = collision detected, above upper = severe collision
    lower_torque_thresholds = [20.0] * 7  # Nm
    upper_torque_thresholds = [40.0] * 7  # Nm
    # Force thresholds for Cartesian axes (N for linear, Nm for angular): [fx, fy, fz, tx, ty, tz]
    lower_force_thresholds = [10.0] * 6  # N (linear) and Nm (angular)
    upper_force_thresholds = [20.0] * 6  # N (linear) and Nm (angular)

    robot.set_collision_behavior(
        lower_torque_thresholds,
        upper_torque_thresholds,
        lower_force_thresholds,
        upper_force_thresholds,
    )

    # Joint impedance (stiffness per joint in Nm/rad): higher = stiffer joint response
    robot.set_joint_impedance([3000.0, 3000.0, 3000.0, 2500.0, 2500.0, 2000.0, 2000.0])  # Nm/rad
    # Cartesian impedance (stiffness per Cartesian axis): [Kx, Ky, Kz, Ktx, Kty, Ktz] in N/m and Nm/rad
    robot.set_cartesian_impedance([3000.0, 3000.0, 3000.0, 300.0, 300.0, 300.0])  # N/m and Nm/rad])


class MotionGenerator:
    """Synchronized minimum-jerk trajectory generator for 7-DOF joint motion.

    Computes time-parameterized trajectories between a start configuration and a goal
    configuration, respecting per-axis maximum velocities and accelerations. All 7 joints
    are synchronized to share the same trajectory duration (the maximum across all axes).

    Usage:
        mg = MotionGenerator(speed_factor=0.05, q_goal=[...])
        # In control loop:
        joint_positions = mg(robot_state, duration_sec)
        if joint_positions.motion_finished:
            break
    """

    def __init__(self, speed_factor: float, q_goal: List[float]):
        """Initialize the motion generator.

        Args:
            speed_factor: Scales max velocity and acceleration (0.0-1.0). Lower = slower motion.
            q_goal: Target joint positions (7 values in radians) for the trajectory.
        """
        self.q_goal = np.array(q_goal)
        self.time = 0.0

        # Maximum joint velocities (rad/s) per axis, scaled by speed_factor
        self.dq_max = np.array([2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5]) * speed_factor
        # Maximum accelerations at start (rad/s^2), scaled by speed_factor
        self.ddq_max_start = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]) * speed_factor
        # Maximum accelerations at goal (rad/s^2), scaled by speed_factor
        self.ddq_max_goal = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]) * speed_factor

        # State variables computed during synchronization phase
        self.q_start = np.zeros(7)        # Starting joint positions (set on first call)
        self.delta_q = np.zeros(7)        # Joint displacement vector (q_goal - q_start)
        self.dq_max_sync = np.zeros(7)    # Synchronized max velocity for each axis
        self.t_1_sync = np.zeros(7)       # Acceleration phase duration per axis
        self.t_2_sync = np.zeros(7)       # Deceleration phase start time per axis
        self.t_f_sync = np.zeros(7)       # Total trajectory duration per axis
        self.q_1 = np.zeros(7)            # Joint positions at end of acceleration phase

        self.delta_q_motion_finished = 1e-12  # Threshold to consider a joint at goal
        self.initialized = False              # Whether first call has been made

    def calculate_synchronized_values(self):
        """Compute synchronized trajectory timing parameters for all 7 joints.

        This method determines a common trajectory duration across all axes by:
        1. Computing per-axis maximum reachable velocity given the displacement and acceleration limits.
        2. Finding the maximum trajectory duration across all axes.
        3. Solving a quadratic equation to find the synchronized max velocity that allows all
           axes to complete their motion within the common duration.

        Results are stored in self.dq_max_sync, self.t_1_sync, self.t_2_sync, self.t_f_sync, self.q_1.
        """
        dq_max_reach = self.dq_max.copy()
        t_f = np.zeros(7)
        delta_t_2 = np.zeros(7)
        t_1 = np.zeros(7)
        delta_t_2_sync = np.zeros(7)
        sign_delta_q = np.sign(self.delta_q).astype(int)

        # Phase 1: Compute per-axis trajectory durations assuming individual max velocities
        for i in range(7):
            if abs(self.delta_q[i]) > self.delta_q_motion_finished:
                # Check if the axis can reach its max velocity within the given displacement
                # If displacement is too short, the axis follows a trapezoidal profile (no constant velocity phase)
                if abs(self.delta_q[i]) < (
                    3.0 / 4.0 * (self.dq_max[i] ** 2 / self.ddq_max_start[i])
                    + 3.0 / 4.0 * (self.dq_max[i] ** 2 / self.ddq_max_goal[i])
                ):
                    # Compute reduced max velocity for short-distance motion
                    dq_max_reach[i] = np.sqrt(
                        4.0
                        / 3.0
                        * self.delta_q[i]
                        * sign_delta_q[i]
                        * (self.ddq_max_start[i] * self.ddq_max_goal[i])
                        / (self.ddq_max_start[i] + self.ddq_max_goal[i])
                    )

                t_1[i] = 1.5 * dq_max_reach[i] / self.ddq_max_start[i]
                delta_t_2[i] = 1.5 * dq_max_reach[i] / self.ddq_max_goal[i]
                t_f[i] = t_1[i] / 2.0 + delta_t_2[i] / 2.0 + abs(self.delta_q[i]) / dq_max_reach[i]

        # The synchronized duration is the maximum across all axes
        max_t_f = np.max(t_f)

        # Phase 2: Solve for synchronized max velocity using quadratic formula
        # This ensures all axes complete simultaneously within the common duration
        for i in range(7):
            if abs(self.delta_q[i]) > self.delta_q_motion_finished:
                # Coefficients of the quadratic equation for synchronized velocity
                a = 1.5 / 2.0 * (self.ddq_max_goal[i] + self.ddq_max_start[i])
                b = -1.0 * max_t_f * self.ddq_max_goal[i] * self.ddq_max_start[i]
                c = abs(self.delta_q[i]) * self.ddq_max_goal[i] * self.ddq_max_start[i]

                delta = b * b - 4.0 * a * c
                if delta < 0.0:
                    delta = 0.0

                self.dq_max_sync[i] = (-1.0 * b - np.sqrt(delta)) / (2.0 * a)
                self.t_1_sync[i] = 1.5 * self.dq_max_sync[i] / self.ddq_max_start[i]
                delta_t_2_sync[i] = 1.5 * self.dq_max_sync[i] / self.ddq_max_goal[i]
                self.t_f_sync[i] = (
                    self.t_1_sync[i] / 2.0
                    + delta_t_2_sync[i] / 2.0
                    + abs(self.delta_q[i] / self.dq_max_sync[i])
                )
                self.t_2_sync[i] = self.t_f_sync[i] - delta_t_2_sync[i]
                self.q_1[i] = self.dq_max_sync[i] * sign_delta_q[i] * (0.5 * self.t_1_sync[i])

    def calculate_desired_values(self, t: float) -> Tuple[np.ndarray, bool]:
        """Calculate desired joint displacements at time t along the trajectory.

        Uses piecewise cubic polynomials (minimum jerk profile) to compute smooth joint positions
        at any point during the trajectory. The motion is divided into three phases per axis:
        - Acceleration phase [0, t1]: smooth acceleration from rest
        - Constant velocity phase [t1, t2]: constant velocity segment (if displacement is long enough)
        - Deceleration phase [t2, tf]: smooth deceleration to rest

        Args:
            t: Current time elapsed since trajectory start.

        Returns:
            delta_q_d: Array of 7 desired joint displacements from start position.
            motion_finished: True if all joints have reached their goal positions.
        """
        delta_q_d = np.zeros(7)
        sign_delta_q = np.sign(self.delta_q).astype(int)
        t_d = self.t_2_sync - self.t_1_sync
        delta_t_2_sync = self.t_f_sync - self.t_2_sync
        joint_motion_finished = [False] * 7

        for i in range(7):
            if abs(self.delta_q[i]) < self.delta_q_motion_finished:
                # Axis displacement is negligible; mark as finished
                delta_q_d[i] = 0
                joint_motion_finished[i] = True
            else:
                if t < self.t_1_sync[i]:
                    # Acceleration phase: cubic polynomial for smooth start
                    delta_q_d[i] = (
                        -1.0
                        / (self.t_1_sync[i] ** 3)
                        * self.dq_max_sync[i]
                        * sign_delta_q[i]
                        * (0.5 * t - self.t_1_sync[i])
                        * (t**3)
                    )
                elif t >= self.t_1_sync[i] and t < self.t_2_sync[i]:
                    # Constant velocity phase (or shortened phase for short distances)
                    delta_q_d[i] = (
                        self.q_1[i] + (t - self.t_1_sync[i]) * self.dq_max_sync[i] * sign_delta_q[i]
                    )
                elif t >= self.t_2_sync[i] and t < self.t_f_sync[i]:
                    # Deceleration phase: cubic polynomial for smooth stop
                    delta_q_d[i] = (
                        self.delta_q[i]
                        + 0.5
                        * (
                            1.0
                            / (delta_t_2_sync[i] ** 3)
                            * (t - self.t_1_sync[i] - 2.0 * delta_t_2_sync[i] - t_d[i])
                            * ((t - self.t_1_sync[i] - t_d[i]) ** 3)
                            + (2.0 * t - 2.0 * self.t_1_sync[i] - delta_t_2_sync[i] - 2.0 * t_d[i])
                        )
                        * self.dq_max_sync[i]
                        * sign_delta_q[i]
                    )
                else:
                    # Past trajectory end: hold at goal position
                    delta_q_d[i] = self.delta_q[i]
                    joint_motion_finished[i] = True

        return delta_q_d, all(joint_motion_finished)

    def __call__(self, robot_state: RobotState, duration_sec: float) -> JointPositions:
        """Generate joint position targets for the current control cycle.

        This is the primary interface for the MotionGenerator. On the first call, it initializes
        the trajectory from the current robot state. On subsequent calls, it computes the desired
        joint positions along the pre-computed trajectory.

        Args:
            robot_state: Current state of the Franka robot (provides starting position on init).
            duration_sec: Time elapsed since the last control cycle (seconds).

        Returns:
            JointPositions object with the target joint positions and motion_finished flag.
        """
        self.time += duration_sec

        if not self.initialized:
            # First call: capture current state as start position and compute trajectory parameters
            self.q_start = np.array(robot_state.q)
            self.delta_q = self.q_goal - self.q_start
            self.calculate_synchronized_values()
            self.initialized = True

        delta_q_d, motion_finished = self.calculate_desired_values(self.time)
        joint_positions = list(self.q_start + delta_q_d)

        output = JointPositions(joint_positions)
        output.motion_finished = motion_finished
        return output
