#!/usr/bin/env python3


# Copyright (c) 2025 Franka Robotics GmbH
# Use of this source code is governed by the Apache-2.0 license, see LICENSE

"""
Joint Impedance Target Example

This example shows a joint impedance controller that renders a spring damper system
to move the robot through a sequence of target joint configurations.
The controller will generate smooth trajectories between positions and run in a continuous loop.
"""

import argparse
import sys
import time

import numpy as np

from pylibfranka import Robot, Torques


class SimpleMotionGenerator:
    """Minimum jerk trajectory generator for smooth joint motion between configurations.

    Uses a 5th-order polynomial (minimum jerk profile) to generate smooth trajectories
    with continuous velocity and acceleration, ensuring jerk (rate of change of acceleration)
    is minimized for comfortable, vibration-free motion.

    The trajectory profile s(t) satisfies:
      s(0) = 0, s'(0) = 0, s''(0) = 0
      s(1) = 1, s'(1) = 0, s''(1) = 0
    where s is the normalized progress parameter in [0, 1].
    """

    def __init__(self, start_position, end_position, duration=3.0):
        """Initialize the trajectory generator.

        Args:
            start_position: Starting joint positions (iterable of 7 values in radians)
            end_position: Target joint positions (iterable of 7 values in radians)
            duration: Duration of the trajectory in seconds
        """
        self.start_position = np.array(start_position)
        self.end_position = np.array(end_position)
        self.duration = duration
        self.start_time = None

    def start(self):
        """Record the current time as the trajectory start time."""
        self.start_time = time.time()

    def get_position(self):
        """Compute the target joint position at the current time along the trajectory.

        Returns:
            np.ndarray: 7-element array of target joint positions (radians).
        """
        if self.start_time is None:
            return self.start_position

        elapsed_time = time.time() - self.start_time
        # Normalize elapsed time to [0, 1] and compute minimum jerk progress parameter
        s = self._minimum_jerk(min(elapsed_time / self.duration, 1.0))

        # Interpolate between start and end positions using the progress parameter
        return self.start_position + s * (self.end_position - self.start_position)

    def is_finished(self):
        """Check if the trajectory duration has elapsed.

        Returns:
            bool: True if the trajectory has completed, False otherwise.
        """
        if self.start_time is None:
            return False

        elapsed_time = time.time() - self.start_time
        return elapsed_time >= self.duration

    def _minimum_jerk(self, t):
        """Compute the minimum jerk trajectory profile at normalized time t.

        The 5th-order polynomial ensures continuous velocity and acceleration at
        the boundaries, minimizing jerk (derivative of acceleration) for smooth motion.

        Args:
            t: Normalized time in [0, 1].

        Returns:
            float: Progress parameter in [0, 1].
        """
        return 10 * (t**3) - 15 * (t**4) + 6 * (t**5)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="localhost", help="Robot IP address")
    args = parser.parse_args()

    # Sequence of target joint configurations for the robot to visit in order
    # Each configuration is a 7-element list of joint angles in radians
    target_joint_positions = [
        # Home position (slightly bent arm)
        [0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0],
        # Extended arm pointing forward
        [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0],
        # Arm pointing to the right
        [0.5, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0],
        # Arm pointing to the left
        [-0.5, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0],
        # Home position again
        [0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0],
    ]

    # Joint impedance parameters for spring-damper behavior
    # Stiffness of 0 means the controller renders a pure damper (no spring force)
    # This allows the robot to be fully compliant at the target configuration
    joint_stiffness = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # Critical damping coefficient: 2*sqrt(k) for each joint
    joint_damping = [2.0 * np.sqrt(k) for k in joint_stiffness]

    try:
        # Connect to robot
        robot = Robot(args.ip)

        # Disable collision detection by setting thresholds to very large values
        # This is necessary for torque control where we compute the full torque command
        lower_torque_thresholds = [1e6]*7
        upper_torque_thresholds = [1e6]*7
        lower_force_thresholds = [1e6]*6
        upper_force_thresholds = [1e6]*6

        robot.set_collision_behavior(
            lower_torque_thresholds,
            upper_torque_thresholds,
            lower_force_thresholds,
            upper_force_thresholds,
        )

        # Capture the initial joint configuration as the starting point
        initial_state = robot.read_once()
        current_position = np.array(initial_state.q)

        # Start torque control mode (direct torque control, not position/velocity)
        active_control = robot.start_torque_control()

        # Load the robot kinematic/dynamic model for computing coriolis forces
        model = robot.load_model()

        # Time to hold at each target position before transitioning to the next
        wait_time = 0.5  # seconds

        # Iterate through each target configuration, generating smooth trajectories between them
        for target_position in target_joint_positions:
            # Create a minimum-jerk trajectory from current position to the target
            trajectory = SimpleMotionGenerator(
                current_position,
                target_position,
                duration=3.0,
            )
            trajectory.start()

            # State variables for tracking trajectory progress and wait period
            target_reached = False
            wait_started = False
            wait_start_time = 0

            # Control loop for the current trajectory segment
            while True:
                # Read current robot state from the torque control interface
                robot_state, _ = active_control.readOnce()

                # Compute dynamic model terms and read state variables
                coriolis = np.array(model.coriolis(robot_state))
                q = np.array(robot_state.q)          # Current joint positions (rad)
                dq = np.array(robot_state.dq)        # Current joint velocities (rad/s)

                # Get the current target position along the minimum-jerk trajectory
                q_goal = trajectory.get_position()

                # Compute position error: current position minus target position
                position_error = q - q_goal

                # Compute joint-space impedance control torque:
                # tau = -K * (q - q_goal) - D * dq
                # With K=0, this reduces to pure damping: tau = -D * dq
                tau_task = np.zeros(7)
                for i in range(7):
                    tau_task[i] = -joint_stiffness[i] * position_error[i] - joint_damping[i] * dq[i]

                # Add coriolis compensation to decouple the dynamics
                # This makes the robot behave as if each joint were independent
                tau_d = tau_task + coriolis

                # Wrap torque vector into Torques command object and send to robot
                torque_command = Torques(tau_d.tolist())
                torque_command.motion_finished = False
                active_control.writeOnce(torque_command)

                # Detect when the trajectory has completed (first time only)
                if trajectory.is_finished() and not target_reached:
                    target_reached = True
                    wait_started = True
                    wait_start_time = time.time()

                # Check if the hold period at the target has elapsed
                if wait_started and (time.time() - wait_start_time >= wait_time):
                    # Update current position for the next trajectory segment
                    current_position = q_goal
                    break

    except Exception as e:
        print(f"\nError occurred: {e}")
        if robot is not None:
            robot.stop()
        return -1

    return 0


if __name__ == "__main__":
    sys.exit(main())
