#!/usr/bin/env python3

# Copyright (c) 2025 Franka Robotics GmbH
# Use of this source code is governed by the Apache-2.0 license, see LICENSE

"""
Gripper Control Example

Demonstrates Franka gripper operations: homing, grasping, and verification.
Connects to the gripper attached to the Franka robot's end effector, performs
a homing operation (if requested), grasps an object, verifies the grasp,
and then releases.

Command-line arguments:
    --ip:       Gripper IP address (required)
    --width:    Width of the object to grasp (default: 0.005 m = 5 mm)
    --homing:   Perform homing before grasping (default: 1 = yes)
    --speed:    Grasping speed (default: 0.1 m/s)
    --force:    Grasping force (default: 60 N)
"""

import argparse
import time

from pylibfranka import Gripper


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, required=True, help="Gripper IP address")
    parser.add_argument("--width", type=float, default=0.005, help="Object width to grasp (meters)")
    parser.add_argument(
        "--homing", type=int, default=1, choices=[0, 1], help="Perform homing (0 or 1)"
    )
    parser.add_argument("--speed", type=float, default=0.1, help="Gripper grasping speed (m/s)")
    parser.add_argument("--force", type=float, default=60, help="Gripper grasping force (N)")
    args = parser.parse_args()

    try:
        # Connect to the Franka gripper
        gripper = Gripper(args.ip)
        grasping_width = args.width

        if args.homing:
            # Homing moves the gripper jaws from fully open to fully closed (and back)
            # to calibrate the maximum grasping width for the current finger configuration.
            # This is necessary because the max width cannot be determined statically.
            print("Homing gripper")
            gripper.homing()

        # Wait for homing to complete and gripper to settle
        time.sleep(2.0)

        # Read and display current gripper state for diagnostics
        gripper_state = gripper.read_once()
        print(f"Gripper width: {gripper_state.width}")
        print(f"Gripper is grasped: {gripper_state.is_grasped}")
        print(f"Gripper temperature: {gripper_state.temperature}")
        print(f"Gripper time: {gripper_state.time.to_sec()}")

        # Attempt to grasp an object of the specified width
        # Returns True if the grasp was successful (object detected within width tolerance)
        if not gripper.grasp(grasping_width, args.speed, args.force):
            print("Failed to grasp object.")
            return -1

        # Hold the grasp briefly then verify the object is still held
        time.sleep(3.0)

        gripper_state = gripper.read_once()
        if not gripper_state.is_grasped:
            print("Object lost.")
            return -1

        # Release the object by stopping the gripper
        print("Grasped object, will release it now.")
        gripper.stop()

    except Exception as e:
        print(f"Error occurred: {e}")
        return -1

    return 0


if __name__ == "__main__":
    main()
