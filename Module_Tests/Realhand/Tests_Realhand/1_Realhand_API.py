#!/usr/bin/env python3

"""
RealHand API Command Control Test (Test 1)

Comprehensive smoke test of the RealHand gripper SDK API. Exercises the full set of
control and sensing interfaces in sequence:

  1. Configuration loading (YAML-based hand detection for left/right setup)
  2. Speed and torque parameter configuration and verification
  3. Finger movement commands (pointing pose, then open pose)
  4. State queries (joint positions, motor temperatures, fault codes)
  5. Continuous pressure sensor polling for all five fingers (thumb, index, middle, ring, pinky)

This script is intended for initial hardware validation and API surface exploration.
It runs in an infinite loop reading pressure data once initial commands complete.

Pressure sensor data format:
  Each get_*_matrix_touch() call returns a 2D array representing the pressure sensor
  grid on that finger's pad. Values indicate the magnitude of contact pressure detected
  at each sensor element.

Joint position range: 0 (fully closed) to 250 (fully open) per joint.
Speed/torque range: 0-255 per joint.
"""

import time, os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(target_dir)
from RealHand.real_hand_api import RealHandApi
from RealHand.utils.load_write_yaml import LoadWriteYaml


configyaml = LoadWriteYaml()  # Initialize YAML configuration loader
# Load hand configuration from RealHand/config/setting.yaml
# This file specifies which hands are connected and their communication parameters
setting = configyaml.load_setting_yaml()

# Detect which hand(s) are configured in the YAML settings
left_hand = False
right_hand = False
if setting['REAL_HAND']['LEFT_HAND']['EXISTS'] == True:
    left_hand = True
elif setting['REAL_HAND']['RIGHT_HAND']['EXISTS'] == True:
    right_hand = True

# The SDK only supports one hand at a time. If both are detected, prefer left.
if left_hand == True and right_hand == True:
    left_hand = True
    right_hand = False

# Extract configuration parameters for the selected hand
if left_hand == True:
    hand_exists = True
    hand_joint = setting['REAL_HAND']['LEFT_HAND']['JOINT']     # Hand model (e.g., "L6", "L7")
    hand_type = "left"
    is_touch = setting['REAL_HAND']['LEFT_HAND']['TOUCH']       # Touch sensor capability flag
    can = setting['REAL_HAND']['LEFT_HAND']['CAN']              # CAN bus identifier
    modbus = setting['REAL_HAND']['LEFT_HAND']['MODBUS']        # Modbus slave ID
if right_hand == True:
    hand_exists = True
    hand_joint = setting['REAL_HAND']['RIGHT_HAND']['JOINT']
    hand_type = "right"
    is_touch = setting['REAL_HAND']['RIGHT_HAND']['TOUCH']
    can = setting['REAL_HAND']['RIGHT_HAND']['CAN']
    modbus = setting['REAL_HAND']['RIGHT_HAND']['MODBUS']

# Initialize the RealHand API client with the detected configuration
# Full API reference: https://github.com/realhand-dev/realhand-python-sdk/blob/main/doc/API-Reference.md
real_hand = RealHandApi(hand_joint=hand_joint, hand_type=hand_type, modbus=modbus, can=can)

# =========================================================
# Phase 1: Configure speed and torque, then verify
# =========================================================

# Set per-joint movement speed (range: 0-255, higher = faster)
# Each element controls the speed of one joint motor
speed = [10, 10, 10, 10, 10, 10]
real_hand.set_speed(speed)
# Verify the speed was set correctly by reading it back
set_speed = real_hand.get_speed()
print(f"Set speed is {set_speed}.")

# Set per-joint torque limit (range: 0-255, higher = stronger grip force)
# This controls the maximum gripping force each finger can apply
torque = [10, 10, 10, 10, 10, 10]
real_hand.set_torque(torque)
# Verify the torque was set correctly by reading it back
set_torque = real_hand.get_torque()
print(f"Set torque is {set_torque}.")

# =========================================================
# Phase 2: Execute finger movement commands
# =========================================================

# Move to a "pointing" pose: index finger partially extended, others closed
# Position values map to joint angles: 0 = fully closed, 250 = fully open
position_pointing = [0, 18, 225, 0, 0, 0]
real_hand.finger_move(position_pointing)

time.sleep(2)  # Wait for mechanical motion to complete before next command

# Open all fingers fully (position 250 = fully open)
# Ending in open position is a safety convention for convenience
position_openhand = [250, 250, 250, 250, 250, 250]
real_hand.finger_move(position_openhand)

# =========================================================
# Phase 3: Query current state (positions, temperature, faults)
# =========================================================

# Read current joint positions (6 values, one per joint)
hand_state = real_hand.get_state()
print(f"Hand state is {hand_state}.")

# Read motor temperatures for all joints (in degrees Celsius)
temp_state = real_hand.get_temperature()
print(f"Motor temp state is {temp_state}.")

# Read fault codes for all joints
# Fault code meanings:
#   0 = Normal (no fault)
#   1 = Current overload (motor drawing too much current)
#   2 = Over temperature (motor thermal protection triggered)
#   3 = Encoding error (hall sensor / encoder communication failure)
#   4 = Over/under voltage (power supply out of range)
fault_state = real_hand.get_fault()
print(f"Motor fault state is {fault_state}.")

# =========================================================
# Phase 4: Continuous pressure sensor polling (infinite loop)
# =========================================================

# Poll all finger pad pressure sensors at ~2 Hz
# Each get_*_matrix_touch() returns a 2D array of pressure values
# from the sensor grid embedded in that finger's pad
while True:
    thumb_pressure_state = real_hand.get_thumb_matrix_touch()
    print(f"Thumb pressure state is {thumb_pressure_state}.")

    index_pressure_state = real_hand.get_index_matrix_touch()
    print(f"Index finger pressure state is {index_pressure_state}.")

    middle_pressure_state = real_hand.get_middle_matrix_touch()
    print(f"Middle finger pressure state is {middle_pressure_state}.")

    ring_pressure_state = real_hand.get_ring_matrix_touch()
    print(f"Ring finger pressure state is {ring_pressure_state}.")

    little_pressure_state = real_hand.get_little_matrix_touch()
    print(f"Pinky finger pressure state is {little_pressure_state}.")

    time.sleep(0.5)

print('We made it')