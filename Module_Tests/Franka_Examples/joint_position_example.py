#!/usr/bin/env python3

# Copyright (c) 2025 Franka Robotics GmbH
# Use of this source code is governed by the Apache-2.0 license, see LICENSE

"""
Joint Position Example

Moves joints 4, 5, and 7 simultaneously using joint position control with a sinusoidal profile.
Joints 1, 2, 3, and 6 remain stationary. The motion follows a smooth cosine trajectory that
starts at rest, reaches a peak displacement, and returns to the start position.

Control mode: CartesianImpedance
Duration: 5 seconds
Moving joints: joint_4, joint_5, joint_7 (simultaneous)
Peak displacement: pi/4 radians (~45 degrees) per moving joint
"""

import argparse
import time

import numpy as np

from pylibfranka import ControllerMode, JointPositions, Robot


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="localhost", help="Robot IP address")
    args = parser.parse_args()

    # Connect to robot
    robot = Robot(args.ip)

    try:
        # Set collision behavior thresholds for safety
        # Torque thresholds per joint (Nm): lower/upper bounds for collision detection
        lower_torque_thresholds = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
        upper_torque_thresholds = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
        # Force thresholds for Cartesian axes [fx, fy, fz, tx, ty, tz]
        lower_force_thresholds = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]
        upper_force_thresholds = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]

        robot.set_collision_behavior(
            lower_torque_thresholds,
            upper_torque_thresholds,
            lower_force_thresholds,
            upper_force_thresholds,
        )

        # Safety warning: prompts user to keep emergency stop accessible
        print("WARNING: This example will move the robot!")
        print("Please make sure to have the user stop button at hand!")
        input("Press Enter to continue...")

        # Start joint position control in CartesianImpedance mode
        # CartesianImpedance allows the robot to be compliant while tracking position targets
        active_control = robot.start_joint_position_control(ControllerMode.CartesianImpedance)

        initial_position = [0.0] * 7
        time_elapsed = 0.0
        motion_finished = False

        # External control loop: run at the robot's control frequency (~1kHz)
        while not motion_finished:
            # Read current robot state and time since last control cycle
            robot_state, duration = active_control.readOnce()

            # Accumulate elapsed time for trajectory progression
            time_elapsed += duration.to_sec()

            # On first iteration, capture the current joint position as the reference
            # q_d is the desired (commanded) position; fall back to q (measured) if unavailable
            if time_elapsed <= duration.to_sec():
                initial_position = robot_state.q_d if hasattr(robot_state, "q_d") else robot_state.q

            # Compute sinusoidal displacement profile for moving joints
            # The profile starts at 0, reaches pi/4 peak, and returns to 0 over 5 seconds
            delta_angle = np.pi / 8.0 * (1 - np.cos(np.pi / 2.5 * time_elapsed))

            # Compute new joint positions: only joints 4, 5, 7 move; others stay fixed
            new_positions = [
                initial_position[0],    # joint_1: stationary
                initial_position[1],    # joint_2: stationary
                initial_position[2],    # joint_3: stationary
                initial_position[3] + delta_angle,  # joint_4: moves with sinusoidal profile
                initial_position[4] + delta_angle,  # joint_5: moves with sinusoidal profile
                initial_position[5],    # joint_6: stationary
                initial_position[6] + delta_angle,  # joint_7: moves with sinusoidal profile
            ]

            # Wrap the position array into a JointPositions command object
            joint_positions = JointPositions(new_positions)

            # Signal completion after 5 seconds of motion
            if time_elapsed >= 5.0:
                joint_positions.motion_finished = True
                motion_finished = True
                print("Finished motion, shutting down example")

            # Send joint position command to the robot controller
            active_control.writeOnce(joint_positions)

    except Exception as e:
        print(f"Error occurred: {e}")
        if robot is not None:
            robot.stop()
        return -1


if __name__ == "__main__":
    main()
