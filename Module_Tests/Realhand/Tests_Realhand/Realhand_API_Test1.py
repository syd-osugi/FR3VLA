###########################
# API Command Control Test
###########################

import time, os, sys

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
# API Command Control Test
###########################

# Change set movement speed
# speed: A list containing speed data. Each element corresponds to the speed of each joint.
# If it is L7, it is 7 elements, corresponding to each motor speed.
# Range of each element value: 0~255.
speed = [10, 10, 10, 10, 10, 10]
real_hand.set_speed(speed)
# Retrieves the currently set speed values.
set_speed = real_hand.get_speed()
print(f"Set speed is {set_speed}.")

# Change set torque limit, used to control gripping force.
# torque: A list containing force data. Each element corresponds to the force value of each finger.
# If it is L7, it is 7 elements, corresponding to each motor force value.
# Range of each element value: 0~255.
torque = [10, 10, 10, 10, 10, 10]
real_hand.set_torque(torque)
# Retrieves current finger torque list information.
set_torque = real_hand.get_torque()
print(f"Set torque is {set_torque}.")

# Set joint position
# Sets the target positions of the joints, used to control finger movement.
position_pointing = [0, 18, 225, 0, 0, 0]
real_hand.finger_move(position_pointing)

time.sleep(2) # Allow for hand to move before begining next move

position_openhand = [250, 250, 250, 250, 250, 250]
real_hand.finger_move(position_openhand) # Always end in open position for convenience

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

while True:
    # Pressure data from sensors on the finger pads
    thumb_pressure_state = real_hand.get_thumb_matrix_touch()
    print(f"Thumb pressure state is {thumb_pressure_state}.")

    # Pressure data from sensors on the finger pads
    index_pressure_state = real_hand.get_index_matrix_touch()
    print(f"Index finger pressure state is {index_pressure_state}.")

    # Pressure data from sensors on the finger pads
    middle_pressure_state = real_hand.get_middle_matrix_touch()
    print(f"Middle finger pressure state is {middle_pressure_state}.")

    # Pressure data from sensors on the finger pads
    ring_pressure_state = real_hand.get_ring_matrix_touch()
    print(f"Ring finger pressure state is {ring_pressure_state}.")

    # Pressure data from sensors on the finger pads
    little_pressure_state = real_hand.get_little_matrix_touch()
    print(f"Pinky finger pressure state is {little_pressure_state}.")

    time.sleep(0.5)

print('We made it')