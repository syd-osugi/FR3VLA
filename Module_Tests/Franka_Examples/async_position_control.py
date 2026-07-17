# Copyright (c) 2026 Franka Robotics GmbH
# Apache-2.0

"""
Asynchronous Position Control Example

Demonstrates low-rate asynchronous position control of a Franka robot using the
AsyncPositionControlHandler API. The robot oscillates all 7 joints between two
positions (±0.25 rad from the home configuration) at 50 Hz for 10 seconds.

Key features:
- Uses AsyncPositionControlHandler (newer low-rate API) instead of the traditional
  start_joint_position_control() interface
- External timing loop with fixed 20ms (50 Hz) control period
- SIGINT handler for clean shutdown via Ctrl+C
"""

import signal
import sys
import time
import math
import argparse
import threading
from datetime import timedelta

import pylibfranka as franka
from example_common import setDefaultBehaviour

# Maximum joint velocities (rad/s) per axis for the trajectory generator
kDefaultMaximumVelocities = [0.655, 0.655, 0.655, 0.655, 1.315, 1.315, 1.315]
# Goal tolerance in degrees: accepted error band for declaring motion complete
kDefaultGoalTolerance = 10.0

# Flag for clean shutdown via signal handler
motion_finished = False


def signal_handler(sig, frame):
    """Handle SIGINT (Ctrl+C) by setting the motion_finished flag for clean shutdown."""
    global motion_finished
    if sig == signal.SIGINT:
        motion_finished = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="localhost", help="Robot IP address")
    args = parser.parse_args()

    # Register signal handler for clean shutdown on Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Connect to robot with realtime scheduling (ignored if not available)
        robot = franka.Robot(args.ip, franka.RealtimeConfig.kIgnore)
    except Exception as e:
        print(f"Could not connect to robot: {e}")
        sys.exit(-1)

    # Apply default collision thresholds and impedance parameters
    setDefaultBehaviour(robot)

    # Home position: a standard Franka configuration
    initial_position = [0,
                        -math.pi / 4,
                        0,
                        -3 * math.pi / 4,
                        0,
                        math.pi / 2,
                        math.pi / 4]

    time_elapsed = 0.0
    direction = 1.0
    time_since_last_log = 0.0

    def calculate_joint_position_target(period_sec):
        """Compute the oscillating joint position target for the current control cycle.

        All joints oscillate symmetrically between two positions (±0.25 rad from home).
        The direction flips every 1 second, creating a back-and-forth motion.

        Args:
            period_sec: Time elapsed since the last control cycle (seconds).

        Returns:
            JointPositionTarget: The target joint positions wrapped in the async handler's
                                 target object type.
        """
        nonlocal time_elapsed, direction, time_since_last_log

        time_elapsed += period_sec

        # Compute target positions: oscillate ±0.25 rad from initial position
        target_positions = [
            initial_position[i] + direction * 0.25
            for i in range(7)
        ]

        # Flip direction every 1 second
        time_since_last_log += period_sec
        if time_since_last_log >= 1.0:
            direction *= -1.0
            time_since_last_log = 0.0

        return franka.AsyncPositionControlHandler.JointPositionTarget(
            joint_positions=target_positions
        )

    # Configure the asynchronous position control handler with velocity limits and tolerance
    joint_position_control_configuration = \
        franka.AsyncPositionControlHandler.Configuration(
            maximum_joint_velocities=kDefaultMaximumVelocities,
            goal_tolerance=kDefaultGoalTolerance
        )

    # Apply the configuration to the robot (returns handler and any error)
    result = franka.AsyncPositionControlHandler.configure(robot,
                                                   joint_position_control_configuration)

    if result.error_message is not None:
        print(result.error_message)
        sys.exit(-1)

    position_control_handler = result.handler
    target_feedback = position_control_handler.get_target_feedback()

    # Control loop period: 20ms = 50 Hz
    time_step = 0.020  # 20 ms, 50 Hz

    global motion_finished
    while not motion_finished:
        loop_start = time.monotonic()

        # Get feedback from the handler to check for errors
        target_feedback = position_control_handler.get_target_feedback()
        if target_feedback.error_message is not None:
            print(target_feedback.error_message)
            sys.exit(-1)

        # Compute the next joint position target and send it to the handler
        next_target = calculate_joint_position_target(time_step)
        command_result = position_control_handler.set_joint_position_target(next_target)

        if command_result.error_message is not None:
            print(command_result.error_message)
            sys.exit(-1)

        # Run for 10 seconds then stop
        if time_elapsed > 10.0:
            position_control_handler.stop_control()
            motion_finished = True
            print("Control finished")
            break

        # Maintain fixed control frequency by sleeping for remaining cycle time
        sleep_time = time_step - (time.monotonic() - loop_start)
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
