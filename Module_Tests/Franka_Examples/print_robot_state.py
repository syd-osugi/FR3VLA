#!/usr/bin/env python3

# Copyright (c) 2025 Franka Robotics GmbH
# Use of this source code is governed by the Apache-2.0 license, see LICENSE

"""
Robot State Printer

Reads and displays key state variables from a Franka robot at a configurable rate.
Useful for debugging, monitoring, and understanding the data available from the
RobotState object returned by pylibfranka.

State variables displayed:
    - Robot mode (enum: e.g., IDLE, RUNNING, COMMAND_FAILURE)
    - Joint positions q (7 values, radians)
    - Joint velocities dq (7 values, rad/s)
    - End effector position in base frame [x, y, z] (meters)
    - External joint torques tau_ext_hat_filtered (7 values, Nm)
    - External wrench in base frame O_F_ext_hat_K (6 values, N/Nm)

Command-line arguments:
    --ip:    Robot IP address (default: localhost)
    --rate:  State reading rate in Hz (default: 0.5 Hz = one reading every 2 seconds)
    --count: Number of readings to print (0 = continuous until Ctrl+C)
"""

import argparse
import time

import numpy as np

from pylibfranka import Robot


def print_robot_state(state):
    """Extract and print the most important fields from a RobotState object.

    Each field is wrapped in try/except to handle cases where the attribute
    may not be available (e.g., robot not in a state that reports that data).

    Args:
        state: RobotState object from pylibfranka containing current robot readings.
    """
    print("Robot State (Critical Attributes):")

    # Robot mode indicates the current operational state of the robot controller
    try:
        mode = state.robot_mode
        mode_str = str(mode).split(".")[-1]  # Extract enum value name (e.g., "IDLE")
        print(f"  Robot Mode: {mode_str}")
    except (AttributeError, ValueError):
        print("  Robot Mode: <not available>")

    # Joint positions: current measured angles of all 7 joints in radians
    try:
        print(f"  Joint Positions (q): {np.round(state.q, 4).tolist()}")
    except (AttributeError, TypeError):
        print("  Joint Positions (q): <not available>")

    # Joint velocities: current measured angular velocities of all 7 joints in rad/s
    try:
        print(f"  Joint Velocities (dq): {np.round(state.dq, 4).tolist()}")
    except (AttributeError, TypeError):
        print("  Joint Velocities (dq): <not available>")

    # End effector pose: O_T_EE is a 4x4 homogeneous transformation matrix
    # stored as a 16-element flat array in column-major order.
    # Indices 12, 13, 14 contain the x, y, z position of the end effector in the base frame.
    print("  End Effector:")
    try:
        position = [state.O_T_EE[12], state.O_T_EE[13], state.O_T_EE[14]]
        print(f"    Position: {np.round(position, 4).tolist()}")
    except (AttributeError, TypeError, IndexError):
        print("    Position: <not available>")

    # External joint torques: filtered measurement of external forces at each joint
    # tau_ext_hat_filtered represents the torque exerted by the environment on each joint,
    # estimated by subtracting the robot's dynamic model from the measured motor torques.
    try:
        print(f"  External Joint Torques: {np.round(state.tau_ext_hat_filtered, 4).tolist()}")
    except (AttributeError, TypeError):
        print("  External Joint Torques: <not available>")

    # External wrench: force and torque at the end effector, expressed in the base frame
    # O_F_ext_hat_K = [fx, fy, fz, tx, ty, tz] in Newtons and Newton-meters
    try:
        print(f"  External Wrench (base frame): {np.round(state.O_F_ext_hat_K, 4).tolist()}")
    except (AttributeError, TypeError):
        print("  External Wrench (base frame): <not available>")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Print Franka robot state")
    parser.add_argument("--ip", type=str, default="localhost", help="Robot IP address")
    parser.add_argument(
        "--rate", type=float, default=0.5, help="Rate at which to print state (in Hz)"
    )
    parser.add_argument(
        "--count", type=int, default=1, help="Number of state readings to print (0 for continuous)"
    )
    args = parser.parse_args()

    # Connect to robot
    print(f"Connecting to robot at {args.ip}...")
    robot = Robot(args.ip)

    try:
        # Set collision behavior thresholds (same defaults as other examples)
        # These must be set before reading state in some control modes
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

        print("Connected to robot. Reading state...")

        count = 0
        # Loop: read and print robot state at the specified rate
        # count=0 means run continuously until interrupted
        while args.count == 0 or count < args.count:
            # Read the current robot state (blocking call)
            state = robot.read_once()

            # Display the state with a separator banner
            print("\n" + "=" * 80)
            print(f"Robot State Reading #{count+1}")
            print("=" * 80)
            print_robot_state(state)

            count += 1

            # Sleep between readings (except after the last reading in finite mode)
            if args.count == 0 or count < args.count:
                time.sleep(1.0 / args.rate)

    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()
