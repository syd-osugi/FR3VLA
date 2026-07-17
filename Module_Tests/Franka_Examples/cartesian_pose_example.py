#!/usr/bin/env python3

# Copyright (c) 2025 Franka Robotics GmbH
# Use of this source code is governed by the Apache-2.0 license, see LICENSE

"""
Cartesian Pose Example

Moves the Franka robot's end effector along a circular arc in the X-Z plane using
cartesian pose control. The end effector traces a semicircular path starting and ending
at the initial position, with the arc centered below the starting point.

Control mode: JointImpedance (compliant Cartesian pose tracking)
Duration: 5 seconds
Path: Semicircle with radius 0.15m in X-Z plane
"""

import argparse
import time

import numpy as np

from pylibfranka import CartesianPose, ControllerMode, RealtimeConfig, Robot


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="localhost", help="Robot IP address")
    args = parser.parse_args()

    # Connect to robot with realtime scheduling (ignored if not available)
    robot = Robot(args.ip, RealtimeConfig.kIgnore)

    try:
        # Set collision behavior thresholds for safety during motion
        # Torque thresholds per joint (Nm): collision detection bands
        lower_torque_thresholds = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
        upper_torque_thresholds = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
        # Force thresholds for Cartesian axes [fx, fy, fz, tx, ty, tz]: collision detection bands
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

        # Start cartesian pose control in JointImpedance mode
        # This allows the robot to track Cartesian pose commands while maintaining compliance
        active_control = robot.start_cartesian_pose_control(ControllerMode.JointImpedance)

        time_elapsed = 0.0
        motion_finished = False

        # Capture the initial end-effector pose as the reference for all subsequent poses
        robot_state, duration = active_control.readOnce()
        initial_cartesian_pose = robot_state.O_T_EE

        # External control loop: run at the robot's control frequency
        while not motion_finished:

            # Read current robot state and time since last cycle
            robot_state, duration = active_control.readOnce()

            # Compute circular arc parameters in X-Z plane
            # angle sweeps from 0 to pi/2 over 5 seconds, creating a quarter-circle profile
            kRadius = 0.15
            angle = np.pi / 4 * (1 - np.cos(np.pi / 5.0 * time_elapsed))
            delta_x = kRadius * np.sin(angle)
            delta_z = kRadius * (np.cos(angle) - 1)

            # Update elapsed time for trajectory progression
            time_elapsed += duration.to_sec()

            # Apply the computed displacement to the initial pose
            # O_T_EE is a 4x4 homogeneous transform stored as 16-element flat array
            # Indices 12, 13, 14 correspond to x, y, z position components
            new_cartesian_pose = initial_cartesian_pose.copy()
            new_cartesian_pose[12] += delta_x  # x position displacement
            new_cartesian_pose[14] += delta_z  # z position displacement

            # Wrap the pose array into a CartesianPose command object
            cartesian_pose = CartesianPose(new_cartesian_pose)

            # Signal completion after 5 seconds of motion
            if time_elapsed >= 5.0:
                cartesian_pose.motion_finished = True
                motion_finished = True
                print("Finished motion, shutting down example")

            # Send the pose command to the robot controller
            active_control.writeOnce(cartesian_pose)

    except Exception as e:
        print(f"Error occurred: {e}")
        if robot is not None:
            robot.stop()
        return -1


if __name__ == "__main__":
    main()
