#!/usr/bin/env python3
import can
import time,sys,os
import threading
import numpy as np
from enum import Enum
current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(target_dir)
from utils.color_msg import ColorMsg

class FrameProperty(Enum):
    INVALID_FRAME_PROPERTY = 0x00  # Invalid CAN frame property | no return
    # Parallel command area
    ROLL_POS = 0x01  # Roll joint position | Coordinate frame is built at the root of each finger, rotation angle is defined with the finger in a straight state
    YAW_POS = 0x02  # Yaw joint position | Coordinate frame is built at the root of each finger, rotation angle is defined with the finger in a straight state
    ROOT1_POS = 0x03  # Root joint 1 position | Finger root joint closest to the palm
    ROOT2_POS = 0x04  # Root joint 2 position | Finger root joint closest to the palm
    ROOT3_POS = 0x05  # Root joint 3 position | Finger root joint closest to the palm
    TIP_POS = 0x06  # Fingertip joint position | Joint closest to the fingertip

    ROLL_SPEED = 0x09  # Roll joint speed | Coordinate frame is built at the root of each finger, rotation angle is defined with the finger in a straight state
    YAW_SPEED = 0x0A  # Yaw joint speed | Coordinate frame is built at the root of each finger, rotation angle is defined with the finger in a straight state
    ROOT1_SPEED = 0x0B  # Root joint 1 speed | Finger root joint closest to the palm
    ROOT2_SPEED = 0x0C  # Root joint 2 speed | Finger root joint closest to the palm
    ROOT3_SPEED = 0x0D  # Root joint 3 speed | Finger root joint closest to the palm
    TIP_SPEED = 0x0E  # Fingertip joint speed | Joint closest to the fingertip

    ROLL_TORQUE = 0x11  # Roll joint torque | Coordinate frame is built at the root of each finger, rotation angle is defined with the finger in a straight state
    YAW_TORQUE = 0x12  # Yaw joint torque | Coordinate frame is built at the root of each finger, rotation angle is defined with the finger in a straight state
    ROOT1_TORQUE = 0x13  # Root joint 1 torque | Finger root joint closest to the palm
    ROOT2_TORQUE = 0x14  # Root joint 2 torque | Finger root joint closest to the palm
    ROOT3_TORQUE = 0x15  # Root joint 3 torque | Finger root joint closest to the palm
    TIP_TORQUE = 0x16  # Fingertip joint torque | Joint closest to the fingertip

    ROLL_FAULT = 0x19  # Roll joint fault code | Coordinate frame is built at the root of each finger, rotation angle is defined with the finger in a straight state
    YAW_FAULT = 0x1A  # Yaw joint fault code | Coordinate frame is built at the root of each finger, rotation angle is defined with the finger in a straight state
    ROOT1_FAULT = 0x1B  # Root joint 1 fault code | Finger root joint closest to the palm
    ROOT2_FAULT = 0x1C  # Root joint 2 fault code | Finger root joint closest to the palm
    ROOT3_FAULT = 0x1D  # Root joint 3 fault code | Finger root joint closest to the palm
    TIP_FAULT = 0x1E  # Fingertip joint fault code | Joint closest to the fingertip

    ROLL_TEMPERATURE = 0x21  # Roll joint temperature | Coordinate frame is built at the root of each finger, rotation angle is defined with the finger in a straight state
    YAW_TEMPERATURE = 0x22  # Yaw joint temperature | Coordinate frame is built at the root of each finger, rotation angle is defined with the finger in a straight state
    ROOT1_TEMPERATURE = 0x23  # Root joint 1 temperature | Finger root joint closest to the palm
    ROOT2_TEMPERATURE = 0x24  # Root joint 2 temperature | Finger root joint closest to the palm
    ROOT3_TEMPERATURE = 0x25  # Root joint 3 temperature | Finger root joint closest to the palm
    TIP_TEMPERATURE = 0x26  # Fingertip joint temperature | Joint closest to the fingertip
    # Parallel command area

    # Serial command area
    THUMB_POS = 0x41  # Thumb joint position | Returns this type of data
    INDEX_POS = 0x42  # Index finger joint position | Returns this type of data
    MIDDLE_POS = 0x43  # Middle finger joint position | Returns this type of data
    RING_POS = 0x44  # Ring finger joint position | Returns this type of data
    LITTLE_POS = 0x45  # Little finger joint position | Returns this type of data

    THUMB_SPEED = 0x49  # Thumb speed | Returns this type of data
    INDEX_SPEED = 0x4A  # Index finger speed | Returns this type of data
    MIDDLE_SPEED = 0x4B  # Middle finger speed | Returns this type of data
    RING_SPEED = 0x4C  # Ring finger speed | Returns this type of data
    LITTLE_SPEED = 0x4D  # Little finger speed | Returns this type of data

    THUMB_TORQUE = 0x51  # Thumb torque | Returns this type of data
    INDEX_TORQUE = 0x52  # Index finger torque | Returns this type of data
    MIDDLE_TORQUE = 0x53  # Middle finger torque | Returns this type of data
    RING_TORQUE = 0x54  # Ring finger torque | Returns this type of data
    LITTLE_TORQUE = 0x55  # Little finger torque | Returns this type of data

    THUMB_FAULT = 0x59  # Thumb fault code | Returns this type of data
    INDEX_FAULT = 0x5A  # Index finger fault code | Returns this type of data
    MIDDLE_FAULT = 0x5B  # Middle finger fault code | Returns this type of data
    RING_FAULT = 0x5C  # Ring finger fault code | Returns this type of data
    LITTLE_FAULT = 0x5D  # Little finger fault code | Returns this type of data

    THUMB_TEMPERATURE = 0x61  # Thumb temperature | Returns this type of data
    INDEX_TEMPERATURE = 0x62  # Index finger temperature | Returns this type of data
    MIDDLE_TEMPERATURE = 0x63  # Middle finger temperature | Returns this type of data
    RING_TEMPERATURE = 0x64  # Ring finger temperature | Returns this type of data
    LITTLE_TEMPERATURE = 0x65  # Little finger temperature | Returns this type of data
    # Serial command area

    # Combined command area, merging non-essential single-control data for the same finger
    FINGER_SPEED = 0x81  # Finger speed | Returns this type of data
    FINGER_TORQUE = 0x82  # Finger torque | Returns this type of data
    FINGER_FAULT = 0x83  # Finger fault code | Returns this type of data

    # Fingertip sensor data group
    HAND_NORMAL_FORCE = 0x90  # Normal pressure of all five fingers
    HAND_TANGENTIAL_FORCE = 0x91  # Tangential pressure of all five fingers
    HAND_TANGENTIAL_FORCE_DIR = 0x92  # Tangential direction of all five fingers
    HAND_APPROACH_INC = 0x93  # Approach sensing of all five fingers

    THUMB_ALL_DATA = 0x98  # All thumb data
    INDEX_ALL_DATA = 0x99  # All index finger data
    MIDDLE_ALL_DATA = 0x9A  # All middle finger data
    RING_ALL_DATA = 0x9B  # All ring finger data
    LITTLE_ALL_DATA = 0x9C  # All little finger data
    # Action command ·ACTION
    ACTION_PLAY = 0xA0  # Action

    # Configuration command ·CONFIG
    HAND_UID = 0xC0  # Device unique identifier
    HAND_HARDWARE_VERSION = 0xC1  # Hardware version
    HAND_SOFTWARE_VERSION = 0xC2  # Software version
    HAND_COMM_ID = 0xC3  # Device ID
    HAND_FACTORY_RESET = 0xCE  # Restore factory settings
    HAND_SAVE_PARAMETER = 0xCF  # Save parameters

    WHOLE_FRAME = 0xF0  # Whole-frame transmission | Returns one-byte frame property + entire structure for 485 and network transmission


class RealHandL24Can:
    def __init__(self, config, can_channel='can0', baudrate=1000000, can_id=0x28):
        self.config = config
        self.can_id = can_id
        self.running = True
        self.x01, self.x02, self.x03, self.x04,self.x05,self.x06,self.x07, self.x08,self.x09,self.x0A,self.x0B,self.x0C,self.x0D,self.x0E,self.speed = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
        # Speed
        self.x49, self.x4a, self.x4b, self.x4c, self.x4d = [],[],[],[],[]
        self.x41,self.x42,self.x43,self.x44,self.x45 = [],[],[],[],[]
        # Initialize CAN bus according to operating system
        if sys.platform == "linux":
            self.bus = can.interface.Bus(
                channel=can_channel, interface="socketcan", bitrate=baudrate, 
                can_filters=[{"can_id": can_id, "can_mask": 0x7FF}]
            )
        elif sys.platform == "win32":
            self.bus = can.interface.Bus(
                channel=can_channel, interface='pcan', bitrate=baudrate, 
                can_filters=[{"can_id": can_id, "can_mask": 0x7FF}]
            )
        else:
            raise EnvironmentError("Unsupported platform for CAN interface")

        # Initialize publisher and related parameters according to can_id
        if can_id == 0x28:  # Left hand
            self.hand_exists = config['REAL_HAND']['LEFT_HAND']['EXISTS']
            self.hand_joint = config['REAL_HAND']['LEFT_HAND']['JOINT']
            self.hand_names = config['REAL_HAND']['LEFT_HAND']['NAME']
        elif can_id == 0x27:  # Right hand

            self.hand_exists = config['REAL_HAND']['RIGHT_HAND']['EXISTS']
            self.hand_joint = config['REAL_HAND']['RIGHT_HAND']['JOINT']
            self.hand_names = config['REAL_HAND']['RIGHT_HAND']['NAME']


        # Start receiving thread
        self.receive_thread = threading.Thread(target=self.receive_response)
        self.receive_thread.daemon = True
        self.receive_thread.start()

    def send_command(self, frame_property, data_list):
        """
        Send command to CAN bus
        :param frame_property: Data frame property
        :param data_list: Data payload
        """
        frame_property_value = int(frame_property.value) if hasattr(frame_property, 'value') else frame_property
        data = [frame_property_value] + [int(val) for val in data_list]
        msg = can.Message(arbitration_id=self.can_id, data=data, is_extended_id=False)
        try:
            self.bus.send(msg)
            #print(f"Message sent: ID={hex(self.can_id)}, Data={data}")
        except can.CanError as e:
            print(f"Failed to send message: {e}")
        time.sleep(0.002)

    def receive_response(self):
        """
        Receive and process CAN bus response messages
        """
        while self.running:
            try:
                msg = self.bus.recv(timeout=1.0)  # Blocking receive, 1 second timeout
                if msg:
                    self.process_response(msg)
            except can.CanError as e:
                print(f"Error receiving message: {e}")
    

    def set_joint_positions(self, joint_ranges):
        if len(joint_ranges) == 25:
            l24_pose = self.joint_map(joint_ranges)
            # Use list comprehension to split the list into subarrays of 6 elements each
            chunks = [l24_pose[i:i+6] for i in range(0, 30, 6)]
            self.send_command(FrameProperty.THUMB_POS, chunks[0])
            self.send_command(FrameProperty.INDEX_POS, chunks[1])
            self.send_command(FrameProperty.MIDDLE_POS, chunks[2])
            self.send_command(FrameProperty.RING_POS, chunks[3])
            self.send_command(FrameProperty.LITTLE_POS, chunks[4])
        #self.set_tip_positions(joint_ranges[:5])
        #print(l24_pose)
    
    # Set all finger roll joint positions
    def set_roll_positions(self, joint_ranges):
        self.send_command(FrameProperty.ROLL_POS, joint_ranges)
    # Set all finger yaw joint positions
    def set_yaw_positions(self, joint_ranges):
        self.send_command(FrameProperty.YAW_POS, joint_ranges)
    # Set all finger root1 joint positions
    def set_root1_positions(self, joint_ranges):
        self.send_command(FrameProperty.ROOT1_POS, joint_ranges)
    # Set all finger root2 joint positions
    def set_root2_positions(self, joint_ranges):
        self.send_command(FrameProperty.ROOT2_POS, joint_ranges)
    # Set all finger root3 joint positions
    def set_root3_positions(self, joint_ranges):
        self.send_command(FrameProperty.ROOT3_POS, joint_ranges)
    # Set all fingertip joint positions
    def set_tip_positions(self, joint_ranges=[80]*5):
        self.send_command(FrameProperty.TIP_POS, joint_ranges)
    # Get thumb joint positions
    def get_thumb_positions(self,j=[0]):
        self.send_command(FrameProperty.THUMB_POS, j)
    # Get index finger joint positions
    def get_index_positions(self, j=[0]):
        self.send_command(FrameProperty.INDEX_POS,j)
    # Get middle finger joint positions
    def get_middle_positions(self, j=[0]):
        self.send_command(FrameProperty.MIDDLE_POS,j)
    # Get ring finger joint positions
    def get_ring_positions(self, j=[0]):
        self.send_command(FrameProperty.RING_POS,j)
    # Get little finger joint positions
    def get_little_positions(self, j=[0]):
        self.send_command(FrameProperty.LITTLE_POS, j)
    # Disable 01 mode
    def set_disability_mode(self, j=[1,1,1,1,1]):
        self.send_command(0x85,j)
    # Enable 00 mode
    def set_enable_mode(self, j=[00,00,00,00,00]):
        self.send_command(0x85,j)

    
    def set_speed(self, speed):
        self.speed = [speed]*6
        ColorMsg(msg=f"L24 speed set to: {self.speed}", color="yellow")
        self.send_command(FrameProperty.THUMB_SPEED, self.speed)
        self.send_command(FrameProperty.INDEX_SPEED, self.speed)
        self.send_command(FrameProperty.MIDDLE_SPEED, self.speed)
        self.send_command(FrameProperty.RING_SPEED, self.speed)
        self.send_command(FrameProperty.LITTLE_SPEED, self.speed)
        
    def set_finger_torque(self, torque):
        self.send_command(0x42, torque)

    def request_device_info(self):
        self.send_command(0xC0, [0])
        self.send_command(0xC1, [0])
        self.send_command(0xC2, [0])

    def save_parameters(self):
        self.send_command(0xCF, [])
    def process_response(self, msg):
        if msg.arbitration_id == self.can_id:
            frame_type = msg.data[0]
            response_data = msg.data[1:]
            if frame_type == 0x01:
                self.x01 = list(response_data)
            elif frame_type == 0x02:
                self.x02 = list(response_data)
            elif frame_type == 0x03:
                self.x03 = list(response_data)
            elif frame_type == 0x04:
                self.x04 = list(response_data)
            elif frame_type == 0x05:
                self.x05 = list(response_data)
            elif frame_type == 0x06:
                self.x06 = list(response_data)
                print("_-"*20)
                print(self.x06)
            elif frame_type == 0xC0:
                print(f"Device ID info: {response_data}")
                if self.can_id == 0x28:
                    self.right_hand_info = response_data
                elif self.can_id == 0x27:
                    self.left_hand_info = response_data
            elif frame_type == 0x08:
                self.x08 = list(response_data)
            elif frame_type == 0x09:
                self.x09 = list(response_data)
            elif frame_type == 0x0A:
                self.x0A = list(response_data)
            elif frame_type == 0x0B:
                self.x0B = list(response_data)
            elif frame_type == 0x0C:
                self.x0C = list(response_data)
            elif frame_type == 0x0D:
                self.x0D = list(response_data)
            elif frame_type == 0x22:
                #ColorMsg(msg=f"Five-finger tangential force direction: {list(response_data)}")
                d = list(response_data)
                self.tangential_force_dir = [float(i) for i in d]
            elif frame_type == 0x23:
                #ColorMsg(msg=f"Five-finger approach: {list(response_data)}")
                d = list(response_data)
                self.approach_inc = [float(i) for i in d]
            elif frame_type == 0x41: # Thumb joint position return value
                self.x41 = list(response_data)
            elif frame_type == 0x42: # Index finger joint position return value
                self.x42 = list(response_data)
            elif frame_type == 0x43: # Middle finger joint position return value
                self.x43 = list(response_data)
            elif frame_type == 0x44: # Ring finger joint position return value
                self.x44 = list(response_data)
            elif frame_type == 0x45: # Little finger joint position return value
                self.x45 = list(response_data)
            elif frame_type == 0x49: # Thumb speed return value
                self.x49 = list(response_data)
            elif frame_type == 0x4a: # Index finger speed return value
                self.x4a = list(response_data)
            elif frame_type == 0x4b: # Middle finger speed return value
                self.x4b = list(response_data)
            elif frame_type == 0x4c: # Ring finger speed return value
                self.x4c = list(response_data)
            elif frame_type == 0x4d: # Little finger speed return value
                self.x4d = list(response_data)

    # Topic mapping for L24
    def joint_map(self, pose):
        # L24 CAN data by default receives 30 data points
        l24_pose = [0.0] * 30  # Initialize l24_pose as 30 zeros

        # Mapping table, using a dictionary to simplify the mapping relationship
        mapping = {
            0: 10,  1: 5,   2: 0,   3: 15,  4: None,  5: 20,
            6: None, 7: 6,   8: 1,   9: 16,  10: None, 11: 21,
            12: None, 13: None, 14: 2,  15: 17, 16: None, 17: 22,
            18: None, 19: 8,  20: 3,   21: 18, 22: None, 23: 23,
            24: None, 25: 9,  26: 4,   27: 19, 28: None, 29: 24
        }

        # Iterate over the mapping dictionary to map values
        for l24_idx, pose_idx in mapping.items():
            if pose_idx is not None:
                l24_pose[l24_idx] = pose[pose_idx]

        return l24_pose

    # Convert L24 state values to CMD format state values
    def state_to_cmd(self, l24_state):
        # L24 CAN by default receives 30 data points, initialize pose as 25 zeros
        pose = [0.0] * 25  # Original command data controlling L24 has 25 values

        # Mapping relationship: dictionary storing the mapping between l24_state indices and pose indices
        mapping = {
            0: 10,  1: 5,   2: 0,   3: 15,  5: 20,  7: 6,
            8: 1,   9: 16,  11: 21, 14: 2,  15: 17, 17: 22,
            19: 8,  20: 3,  21: 18, 23: 23, 25: 9,   26: 4,
            27: 19, 29: 24
        }
        # Iterate over mapping dictionary and update pose values
        for l24_idx, pose_idx in mapping.items():
            pose[pose_idx] = l24_state[l24_idx]
        return pose

    # Get all joint data
    def get_current_status(self, j=''):
        time.sleep(0.01)
        self.send_command(FrameProperty.THUMB_POS, j)
        self.send_command(FrameProperty.INDEX_POS,j)
        self.send_command(FrameProperty.MIDDLE_POS,j)
        self.send_command(FrameProperty.RING_POS,j)
        self.send_command(FrameProperty.LITTLE_POS, j)
        #return self.x41, self.x42, self.x43, self.x44, self.x45
        time.sleep(0.1)
        state= self.x41+ self.x42+ self.x43+ self.x44+ self.x45
        if len(state) == 30:
            l24_state = self.state_to_cmd(l24_state=state)
            return l24_state
    
    def get_speed(self,j=''):
        time.sleep(0.1)
        self.send_command(FrameProperty.THUMB_SPEED, j) # Thumb speed
        self.send_command(FrameProperty.INDEX_SPEED, j) # Index finger speed
        self.send_command(FrameProperty.MIDDLE_SPEED, j) # Middle finger speed
        self.send_command(FrameProperty.RING_SPEED, j) # Ring finger speed
        self.send_command(FrameProperty.LITTLE_SPEED, j) # Little finger speed
        speed = self.x49+ self.x4a+ self.x4b+ self.x4c+ self.x4d
        if len(speed) == 30:
            l24_speed = self.state_to_cmd(l24_state=speed)
            return l24_speed
    
    def get_finger_torque(self):
        return self.finger_torque
    # def get_current(self):
    #     return self.x06
    # def get_fault(self):
    #     return self.x07
    def close_can_interface(self):
        if self.bus:
            self.bus.shutdown()  # Close CAN bus

    '''
    This method is only used to demonstrate the data mapping relationship.
    For actual use, it is better to use the methods above.
    '''
    def joint_map_2(self, pose):
        l24_pose = [0.0]*30 # L24 CAN by default receives 30 data; pose is the command data with 25 elements used to control L24, here we map it
        '''
        Mapping needed
        # L24 CAN data format
        #["Thumb roll 0-10", "Thumb abduction 1-5", "Thumb root 2-0", "Thumb middle 3-15", "Reserved 4-", "Thumb tip 5-20", "Reserved 6-", "Index abduction 7-6", "Index root 8-1", "Index middle 9-16", "Reserved 10-", "Index tip 11-21", "Reserved 12-", "Reserved 13-", "Middle root 14-2", "Middle middle 15-17", "Reserved 16-", "Middle tip 17-22", "Reserved 18-", "Ring abduction 19-8", "Ring root 20-3", "Ring middle 21-18", "Reserved 22-", "Ring tip 23-23", "Reserved 24-", "Little abduction 25-9", "Little root 26-4", "Little middle 27-19", "Reserved 28-", "Little tip 29-24"]
        # CMD received data format
        #["Thumb root 0", "Index root 1", "Middle root 2", "Ring root 3","Little root 4","Thumb abduction 5","Index abduction 6","Middle abduction 7","Ring abduction 8","Little abduction 9","Thumb roll 10","Reserved","Reserved","Reserved","Reserved","Thumb middle 15","Index middle 16","Middle middle 17","Ring middle 18","Little middle 19","Thumb tip 20","Index tip 21","Middle tip 22","Ring tip 23","Little tip 24"]
        '''
        l24_pose[0] = pose[10]
        l24_pose[1] = pose[5]
        l24_pose[2] = pose[0]
        l24_pose[3] = pose[15]
        l24_pose[4] = 0.0
        l24_pose[5] = pose[20]
        l24_pose[6] = 0.0
        l24_pose[7] = pose[6]
        l24_pose[8] = pose[1]
        l24_pose[9] = pose[16]
        l24_pose[10] = 0.0
        l24_pose[11] = pose[21]
        l24_pose[12] = 0.0
        l24_pose[13] = 0.0
        l24_pose[14] = pose[2]
        l24_pose[15] = pose[17]
        l24_pose[16] = 0.0
        l24_pose[17] = pose[22]
        l24_pose[18] = 0.0
        l24_pose[19] = pose[8]
        l24_pose[20] = pose[3]
        l24_pose[21] = pose[18]
        l24_pose[22] = 0.0
        l24_pose[23] = pose[23]
        l24_pose[24] = 0.0
        l24_pose[25] = pose[9]
        l24_pose[26] = pose[4]
        l24_pose[27] = pose[19]
        l24_pose[28] = 0.0
        l24_pose[29] = pose[24]
        return l24_pose
    

    def show_fun_table(self):
        pass
