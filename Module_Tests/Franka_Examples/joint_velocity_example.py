#!/usr/bin/env python3

# Copyright (c) 2025 Franka Robotics GmbH
# Use of this source code is governed by the Apache-2.0 license, see LICENSE

"""
Joint Velocity Example

Commands simultaneous angular velocity to joints 4-7 using joint velocity control.
The velocity follows a sinusoidal profile that oscillates direction every 1 second,
with zero velocity at the start and end of the motion.

Control mode: CartesianImpedance
Duration: 2 seconds (2 full cycles of the 1-second period)
Peak angular velocity: 1.0 rad/s
Moving joints: joint_4, joint_5, joint_6, joint_7 (simultaneous, same velocity)
Stationary joints: joint_1, joint_2, joint_3 (zero velocity)
"""

import argparse

import numpy as np

from pylibfranka import ControllerMode, JointVelocities, Robot


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

        # Start joint velocity control in CartesianImpedance mode
        active_control = robot.start_joint_velocity_control(ControllerMode.CartesianImpedance)

        time_max = 1.0        # Period of the sinusoidal velocity oscillation (seconds)
        omega_max = 1.0       # Peak angular velocity (rad/s)
        time_elapsed = 0.0
        motion_finished = False

        # External control loop: run at the robot's control frequency
        while not motion_finished:
            # Read current robot state and time since last control cycle
            robot_state, duration = active_control.readOnce()

            # Accumulate elapsed time for trajectory progression
            time_elapsed += duration.to_sec()

            # Compute oscillating angular velocity profile:
            # The cycle term alternates sign every time_max seconds, creating bidirectional motion.
            # The cosine term creates a smooth sinusoidal envelope that starts and ends at zero.
            cycle = np.floor(
                np.power(-1.0, (time_elapsed - np.fmod(time_elapsed, time_max)) / time_max)
            )
            omega = cycle * omega_max / 2.0 * (1.0 - np.cos(2.0 * np.pi / time_max * time_elapsed))

            # Joint velocity vector: [dq1, dq2, dq3, dq4, dq5, dq6, dq7]
            # Joints 4-7 share the same angular velocity; joints 1-3 are stationary
            velocities = [
                0.0,     # joint_1: stationary (zero velocity)
                0.0,     # joint_2: stationary (zero velocity)
                0.0,     # joint_3: stationary (zero velocity)
                omega,   # joint_4: oscillating angular velocity
                omega,   # joint_5: oscillating angular velocity
                omega,   # joint_6: oscillating angular velocity
                omega,   # joint_7: oscillating angular velocity
            ]

            # Wrap into JointVelocities command object
            joint_velocities = JointVelocities(velocities)

            # Signal completion after 2 full cycles (2 * time_max seconds)
            if time_elapsed >= 2.0 * time_max:
                joint_velocities.motion_finished = True
                motion_finished = True
                print("Finished motion, shutting down example")

            # Send joint velocity command to the robot controller
            active_control.writeOnce(joint_velocities)

    except Exception as e:
        print(f"Error occurred: {e}")
        if robot is not None:
            robot.stop()
        return -1


if __name__ == "__main__":
    main()
