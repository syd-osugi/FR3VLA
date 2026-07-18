#!/usr/bin/env python3

"""
RealHand Pressure State Test (Test 2) - Index Finger Focus

Demonstrates closed-loop grasping using index finger pressure sensor feedback.
The script closes the hand (except the thumb) and continuously polls the index
finger's pressure sensor matrix until contact is detected (max pressure >= 10).

Purpose: Understand how get_index_matrix_touch() responds during object contact,
        mapping pressure sensor grid values to physical grip events.

Procedure:
  1. Start with hand fully open (all joints at position 250)
  2. Close fingers 2-6 (index, middle, ring, pinky) while keeping thumb at 225
  3. Poll index finger pressure matrix in a loop until max pressure >= 10
  4. Print final hand state, motor temperatures, and fault codes

Pressure sensor data:
  get_index_matrix_touch() returns a 2D array where each element represents
  a sensor element on the index finger pad. Values increase as the finger
  makes firmer contact with an object.
"""

import time, os, sys
import numpy

current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(target_dir)
from RealHand.real_hand_api import RealHandApi
from RealHand.utils.load_write_yaml import LoadWriteYaml

configyaml = LoadWriteYaml()  # Initialize YAML configuration loader
# Load hand configuration from RealHand/config/setting.yaml
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
    hand_joint = setting['REAL_HAND']['LEFT_HAND']['JOINT']
    hand_type = "left"
    is_touch = setting['REAL_HAND']['LEFT_HAND']['TOUCH']
    can = setting['REAL_HAND']['LEFT_HAND']['CAN']
    modbus = setting['REAL_HAND']['LEFT_HAND']['MODBUS']
if right_hand == True:
    hand_exists = True
    hand_joint = setting['REAL_HAND']['RIGHT_HAND']['JOINT']
    hand_type = "right"
    is_touch = setting['REAL_HAND']['RIGHT_HAND']['TOUCH']
    can = setting['REAL_HAND']['RIGHT_HAND']['CAN']
    modbus = setting['REAL_HAND']['RIGHT_HAND']['MODBUS']

# Initialize the RealHand API client
real_hand = RealHandApi(hand_joint=hand_joint, hand_type=hand_type, modbus=modbus, can=can)

# =========================================================
# Phase 1: Initialize with open hand
# =========================================================

# Set per-joint movement speed (range: 0-255)
speed = [10, 10, 10, 10, 10, 10]
real_hand.set_speed(speed)

# Set per-joint torque limit for gripping (range: 0-255)
torque = [10, 10, 10, 10, 10, 10]
real_hand.set_torque(torque)

# Open all fingers fully before grasping
position_openhand = [250, 250, 250, 250, 250, 250]
real_hand.finger_move(position_openhand)
time.sleep(2)  # Wait for mechanical motion to complete

# =========================================================
# Phase 2: Close hand while monitoring index pressure feedback
# =========================================================

# Initialize pressure reading from the index finger sensor grid
index_pressure_state = numpy.array(real_hand.get_index_matrix_touch())

# Close fingers until index finger detects contact (max pressure >= 10)
# The thumb stays open (position 225) while fingers 2-6 close to position 0
while numpy.max(index_pressure_state) < 10:  # Max pressure < 10: no contact yet
    # Close index, middle, ring, and pinky; keep thumb slightly open
    position_closed = [225, 0, 0, 0, 0, 0]
    real_hand.finger_move(position_closed)
    # Read the index finger pressure matrix and convert to numpy array for analysis
    index_pressure_state = numpy.array(real_hand.get_index_matrix_touch())
    print(f"Index finger pressure state is {index_pressure_state}.")

time.sleep(0.5)  # Allow mechanical settling after contact

# =========================================================
# Phase 3: Print diagnostic state
# =========================================================

# Read current joint positions (6 values, one per joint)
hand_state = real_hand.get_state()
print(f"Hand state is {hand_state}.")

# Read motor temperatures (degrees Celsius) for all joints
temp_state = real_hand.get_temperature()
print(f"Motor temp state is {temp_state}.")

# Read fault codes for all joints
# Fault codes: 0=Normal, 1=Current Overload, 2=Over Temperature,
#              3=Encoding Error, 4=Over/Under Voltage
fault_state = real_hand.get_fault()
print(f"Motor fault state is {fault_state}.")

print('We made it')