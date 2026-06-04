#!/usr/bin/env python3
"""
Test 26: Read Franka EE pose, move 5 cm, then read the EE pose again.

This is a hardware diagnostic for checking whether Franka O_T_EE changes in the
expected direction and magnitude after a small Cartesian move. It uses the
project's FrankaRobotInterface so workspace bounds, speed limits, collision
behavior, and operator confirmation stay consistent with the rest of Working.

WHY THIS TEST IS USEFUL FOR D405 EXTRINSICS:
============================================
D405 hand-eye calibration depends on the robot pose reported by Franka O_T_EE.
If O_T_EE does not change by the expected amount after a known commanded move,
then the calibration script may be pairing correct camera images with incorrect
or misunderstood robot poses.

This script performs a simple sanity check:

  1. Connect to the Franka.
  2. Read pose A from O_T_EE.
  3. Command a small Cartesian move, default +0.05 m along robot-base X.
  4. Wait for settling.
  5. Read pose B from O_T_EE.
  6. Print the measured delta between pose A and pose B.
  7. Save the result JSON to Module_Tests/Test_Outputs.

HOW TO RUN:
===========
Default +5 cm X move:

    python Module_Tests/Working_Tests/franka_pose_delta_test26.py

Move along another robot-base axis:

    python Module_Tests/Working_Tests/franka_pose_delta_test26.py --axis y
    python Module_Tests/Working_Tests/franka_pose_delta_test26.py --axis z

Move in the negative direction:

    python Module_Tests/Working_Tests/franka_pose_delta_test26.py \
        --axis x --distance-m -0.05

Slow the move down:

    python Module_Tests/Working_Tests/franka_pose_delta_test26.py --speed-mps 0.01

SAFETY NOTES:
=============
This script moves the physical robot. Keep the user stop button within reach.
The script asks for confirmation through FrankaRobotInterface unless you pass
--skip-confirmation. Use that flag only when the workspace is clear and you are
intentionally automating the motion.

WHAT TO EXPECT:
===============
For the default command, the printed measured delta should be close to:

    dx = +0.050 m, dy ~= 0.000 m, dz ~= 0.000 m

Small deviations are normal. Large deviations, sign flips, or movement along an
unexpected axis are clues that the robot-frame interpretation needs attention
before trusting hand-eye calibration results.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from _working_test_utils import TEST_OUTPUTS_DIR

import config as cfg
from robot.robot_interface import FrankaRobotInterface


AXIS_TO_INDEX = {
    "x": 0,
    "y": 1,
    "z": 2,
}


def pose_from_state(state):
    """
    Extract a 4x4 EE pose matrix from RobotState.

    FrankaRobotInterface already converts raw Franka O_T_EE from column-major
    flat form into a nested 4x4 list. This helper makes sure the pose exists and
    has the shape the diagnostic expects.
    """
    if state.ee_pose is None:
        raise RuntimeError("Franka state did not include ee_pose")
    pose = np.array(state.ee_pose, dtype=float)
    if pose.shape != (4, 4):
        raise RuntimeError(f"Expected 4x4 ee_pose, got shape {pose.shape}")
    return pose


def print_pose(label, pose):
    """
    Print both the translation and full matrix for a pose.

    Translation is the quick sanity check. The full matrix is useful when
    comparing orientation before and after a supposedly translation-only move.
    """
    xyz = pose[:3, 3]
    print(f"\n{label}")
    print(f"  translation xyz [m]: [{xyz[0]: .6f}, {xyz[1]: .6f}, {xyz[2]: .6f}]")
    print("  matrix:")
    for row in pose:
        print(f"    [{row[0]: .6f} {row[1]: .6f} {row[2]: .6f} {row[3]: .6f}]")


def write_result(args, start_pose, end_pose, target_xyz, motion_result):
    """
    Save the diagnostic result to JSON for later comparison.

    Keeping the exact start/end matrices is helpful when debugging calibration
    sessions because you can compare the command, the measured O_T_EE delta, and
    any suspicious extrinsic results after the robot is powered down.
    """
    TEST_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = TEST_OUTPUTS_DIR / "franka_pose_delta_test26.json"
    data = {
        "robot_ip": args.ip,
        "axis": args.axis,
        "distance_m": args.distance_m,
        "speed_mps": args.speed_mps,
        "settle_s": args.settle_s,
        "start_pose": start_pose.tolist(),
        "target_xyz": target_xyz.tolist(),
        "end_pose": end_pose.tolist(),
        "measured_delta_xyz_m": (end_pose[:3, 3] - start_pose[:3, 3]).tolist(),
        "motion_result": motion_result,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def parse_args():
    """
    Parse command-line options for a small, bounded Cartesian diagnostic move.

    The default is intentionally small and slow. The script also rejects moves
    larger than 10 cm in main() so accidental units mistakes are caught before
    the robot is commanded.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Read Franka O_T_EE, move a small Cartesian delta, then read O_T_EE again."
        )
    )
    parser.add_argument(
        "--ip",
        default=cfg.FRANKA_IP,
        help="Franka robot IP address. Defaults to config.FRANKA_IP.",
    )
    parser.add_argument(
        "--axis",
        choices=sorted(AXIS_TO_INDEX),
        default="x",
        help="Robot-base axis to move along. Default: x.",
    )
    parser.add_argument(
        "--distance-m",
        type=float,
        default=0.05,
        help="Signed Cartesian distance in meters. Default: +0.05.",
    )
    parser.add_argument(
        "--speed-mps",
        type=float,
        default=0.02,
        help="Cartesian speed in m/s. Default: 0.02.",
    )
    parser.add_argument(
        "--settle-s",
        type=float,
        default=2.0,
        help="Seconds to wait after the move before reading pose B. Default: 2.0.",
    )
    parser.add_argument(
        "--skip-confirmation",
        action="store_true",
        help="Skip the typed motion confirmation prompt. Use only when the area is clear.",
    )
    return parser.parse_args()


def main():
    """
    Run the pose-delta diagnostic.

    This function is intentionally linear and verbose: print pose A, move, wait,
    print pose B, print delta. The output should be easy to compare with the
    commanded axis and distance while standing at the robot.
    """
    args = parse_args()
    if not np.isfinite(args.distance_m) or abs(args.distance_m) > 0.10:
        print("ERROR: distance must be finite and no more than 0.10 m for this diagnostic.")
        return 1
    if not np.isfinite(args.speed_mps) or args.speed_mps <= 0.0:
        print("ERROR: speed must be finite and positive.")
        return 1
    if not np.isfinite(args.settle_s) or args.settle_s < 0.0:
        print("ERROR: settle time must be finite and non-negative.")
        return 1

    print("=" * 72)
    print("  FRANKA 5 CM POSE DELTA DIAGNOSTIC")
    print("=" * 72)
    print("This script will move the robot. Keep the user stop button within reach.")
    print(f"Robot IP: {args.ip}")
    print(f"Commanded move: {args.distance_m:+.4f} m along robot-base {args.axis.upper()}")
    print(f"Speed: {args.speed_mps:.4f} m/s")

    interface = FrankaRobotInterface(
        robot_ip=args.ip,
        require_motion_confirmation=not args.skip_confirmation,
    )

    try:
        connection = interface.connect()
        print(f"\nConnected: {connection}")

        start_state = interface.read_state()
        start_pose = pose_from_state(start_state)
        print_pose("Pose A before motion", start_pose)

        target_xyz = start_pose[:3, 3].copy()
        target_xyz[AXIS_TO_INDEX[args.axis]] += args.distance_m
        print(
            "\nTarget translation xyz [m]: "
            f"[{target_xyz[0]: .6f}, {target_xyz[1]: .6f}, {target_xyz[2]: .6f}]"
        )

        motion_result = interface.move_to_waypoints(
            [target_xyz.tolist()],
            speed_mps=args.speed_mps,
            source="franka_pose_delta_test26",
        )
        print(f"\nMotion result: {motion_result}")

        if args.settle_s > 0.0:
            print(f"\nWaiting {args.settle_s:.2f} s before reading pose B...")
            time.sleep(args.settle_s)

        end_state = interface.read_state()
        end_pose = pose_from_state(end_state)
        print_pose("Pose B after motion", end_pose)

        delta_xyz = end_pose[:3, 3] - start_pose[:3, 3]
        target_delta = target_xyz - start_pose[:3, 3]
        print("\nMeasured translation delta [m]:")
        print(f"  dx={delta_xyz[0]: .6f}, dy={delta_xyz[1]: .6f}, dz={delta_xyz[2]: .6f}")
        print("Expected translation delta [m]:")
        print(
            f"  dx={target_delta[0]: .6f}, "
            f"dy={target_delta[1]: .6f}, dz={target_delta[2]: .6f}"
        )
        print(f"Delta magnitude: {np.linalg.norm(delta_xyz):.6f} m")

        output_path = write_result(args, start_pose, end_pose, target_xyz, motion_result)
        print(f"\nSaved diagnostic JSON: {output_path}")
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted; requesting robot stop.")
        interface.stop()
        return 130
    except Exception as exc:
        print(f"\nERROR: {exc}")
        interface.stop()
        return 1
    finally:
        interface.close()


if __name__ == "__main__":
    sys.exit(main())
