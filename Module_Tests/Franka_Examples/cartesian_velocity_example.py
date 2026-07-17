#!/usr/bin/env python3

# Copyright (c) 2025 Franka Robotics GmbH
# Use of this source code is governed by the Apache-2.0 license, see LICENSE

"""
Cartesian Velocity Example

Two-phase motion:
1. Moves the robot to a home configuration using joint position control with a MotionGenerator.
2. Commands Cartesian end-effector velocity along a sinusoidal profile at 45 degrees in the X-Z plane.

Phase 1: Joint position motion to home pose (duration depends on MotionGenerator)
Phase 2: Cartesian velocity oscillation at 0.01 m/s peak, alternating direction every 1 second
Total velocity phase: 2 seconds
Control mode: CartesianImpedance
"""

import argparse

import numpy as np
from example_common import MotionGenerator, setDefaultBehaviour

from pylibfranka import CartesianVelocities, ControllerMode, Robot


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="localhost", help="Robot IP address")
    args = parser.parse_args()

    # Connect to robot and enable automatic error recovery
    robot = Robot(args.ip)
    robot.automatic_error_recovery()

    # Safety warning: prompts user to keep emergency stop accessible
    print("WARNING: This example will move the robot!")
    print("Please make sure to have the user stop button at hand!")
    input("Press Enter to continue...")

    try:
        # Apply default collision thresholds and impedance parameters
        setDefaultBehaviour(robot)

        # Home configuration: neutral pose with wrist bent
        q_goal = [0.0, 0.0, 0.0, -np.pi/2, 0.0, np.pi/2, np.pi/4]

        # Create motion generator for phase 1 (slow speed for smooth approach)
        motion_generator = MotionGenerator(speed_factor=0.05, q_goal=q_goal)

        # Phase 1: Move to home configuration using joint position control
        control = robot.start_joint_position_control(ControllerMode.CartesianImpedance)
        while True:
            state, duration = control.readOnce()  # Unpack (RobotState, Duration) tuple
            joint_positions = motion_generator(state, duration.to_sec())

            control.writeOnce(joint_positions)
            if joint_positions.motion_finished:
                robot.stop()
                break

    except KeyboardInterrupt:
        print("\nMotion interrupted by user")

    print("Finished moving to initial joint configuration.")

    # Reconfigure impedance for the velocity control phase
    # NOTE: These must be set OUTSIDE the control loop, before starting it
    robot.set_joint_impedance([3000.0, 3000.0, 3000.0, 2500.0, 2500.0, 2000.0, 2000.0])  # Nm/rad

    try:
        # Set collision behavior for phase 2
        lower_torque_thresholds = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
        upper_torque_thresholds = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
        lower_force_thresholds = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]
        upper_force_thresholds = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]

        robot.set_collision_behavior(
            lower_torque_thresholds,
            upper_torque_thresholds,
            lower_force_thresholds,
            upper_force_thresholds,
        )

        # Start Cartesian velocity control in CartesianImpedance mode
        active_control = robot.start_cartesian_velocity_control(ControllerMode.CartesianImpedance)

        time_max = 1.0        # Period of the sinusoidal velocity oscillation (seconds)
        v_max = 0.01          # Peak linear velocity (m/s)
        angle = np.pi / 4.0   # Direction angle in X-Z plane (45 degrees)
        time_elapsed = 0.0
        motion_finished = False

        # Phase 2: External control loop for Cartesian velocity commands
        while not motion_finished:
            # Read current robot state and cycle duration
            robot_state, duration = active_control.readOnce()

            # Accumulate elapsed time for trajectory progression
            time_elapsed += duration.to_sec()

            # Compute oscillating velocity profile:
            # The cycle term alternates sign every time_max seconds, creating bidirectional motion.
            # The cosine term creates a smooth sinusoidal envelope that starts and ends at zero velocity.
            cycle = np.floor(
                np.power(-1.0, (time_elapsed - np.fmod(time_elapsed, time_max)) / time_max)
            )
            v = cycle * v_max / 2.0 * (1.0 - np.cos(2.0 * np.pi / time_max * time_elapsed))

            # Project velocity onto the X-Z plane at the specified angle
            v_x = np.cos(angle) * v
            v_z = np.sin(angle) * v

            # Cartesian velocity vector: [vx, vy, vz, wx, wy, wz]
            velocities = [
                v_x,     # X-axis linear velocity
                0.0,     # Y-axis linear velocity (zero)
                v_z,     # Z-axis linear velocity
                0.0,     # X-axis angular velocity
                0.0,     # Y-axis angular velocity
                0.0,     # Z-axis angular velocity
            ]

            # Wrap into CartesianVelocities command object
            cartesian_velocities = CartesianVelocities(velocities)

            # Signal completion after 2 full cycles (2 * time_max seconds)
            if time_elapsed >= 2.0 * time_max:
                cartesian_velocities.motion_finished = True
                motion_finished = True
                print("Finished motion, shutting down example")

            # Send velocity command to the robot controller
            active_control.writeOnce(cartesian_velocities)

    except KeyboardInterrupt:
        print("\nMotion interrupted by user")

    except Exception as e:
        print(f"Error occurred: {e}")
        if robot is not None:
            robot.stop()
        return -1


if __name__ == "__main__":
    main()
