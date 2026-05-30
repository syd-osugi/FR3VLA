###########################
# API Command get_pressure_state Control Test
# Test closing the hand over various objects to understand how get_pressure_state responds
# Currently tracks only index finger
###########################

import time, os, sys
import numpy

current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(target_dir)
from RealHand.real_hand_api import RealHandApi
from RealHand.utils.load_write_yaml import LoadWriteYaml

configyaml = LoadWriteYaml() # Initialize configuration file
# Read configuration file
# This line gets all info from RealHand/config/setting.yaml
setting = configyaml.load_setting_yaml()

left_hand = False
right_hand = False
if setting['REAL_HAND']['LEFT_HAND']['EXISTS'] == True:
    left_hand = True
elif setting['REAL_HAND']['RIGHT_HAND']['EXISTS'] == True:
    right_hand = True
# GUI control only supports single hand, mutual exclusion for left/right hand here
if left_hand == True and right_hand == True:
    left_hand = True
    right_hand = False
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

# Allows us to use https://github.com/realhand-dev/realhand-python-sdk/blob/main/doc/API-Reference.md
real_hand = RealHandApi(hand_joint=hand_joint, hand_type=hand_type, modbus=modbus, can=can)

###########################
# Start with open hand
###########################

# Change set movement speed
speed = [10, 10, 10, 10, 10, 10]
real_hand.set_speed(speed)

# Change set torque limit, used to control gripping force.
torque = [10, 10, 10, 10, 10, 10]
real_hand.set_torque(torque)

position_openhand = [250, 250, 250, 250, 250, 250]
real_hand.finger_move(position_openhand) # Always start in open position for convenience
time.sleep(2) # Allow for hand to move before begining next move

###########################
# Close hand around object
###########################

# Initialize index_pressure_state
index_pressure_state = numpy.array(real_hand.get_index_matrix_touch())

while numpy.max(index_pressure_state) < 10: # Max pressure < 10
    # Pressure data from sensors on the finger pads
    position_closed = [225, 0, 0, 0, 0, 0] # Close hand aside from the thumb
    real_hand.finger_move(position_closed)
    index_pressure_state = numpy.array(real_hand.get_index_matrix_touch()) # Retrieve pressure data in form of an array
    print(f"Index finger pressure state is {index_pressure_state}.")

time.sleep(.5) # Allow for hand to move before begining next move

# position_openhand = [250, 250, 250, 250, 250, 250]
# real_hand.finger_move(position_openhand) # Always end in open position for convenience

# Current state of hand postions
hand_state = real_hand.get_state()
print(f"Hand state is {hand_state}.")

# Retrieves the motor temperature of the current joints.
temp_state = real_hand.get_temperature()
print(f"Motor temp state is {temp_state}.")
# Retrieves current joint motor faults.
# 0 means normal; 1: Current Overload; 2: Over Temperature; 3: Encoding Error; 4: Over/Under Voltage.
fault_state = real_hand.get_fault()
print(f"Motor fault state is {fault_state}.")



print('We made it')