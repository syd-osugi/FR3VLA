#!/usr/bin/env python3
import sys,os,time
current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.append(target_dir)

from RealHand.real_hand_api import RealHandApi
from RealHand.utils.load_write_yaml import LoadWriteYaml
from RealHand.utils.init_real_hand import InitRealHand
from RealHand.utils.color_msg import ColorMsg
'''
Currently L10 does not listen to the CAN command for current speed, so real-time speed acquisition is not supported for now.
'''
class GetSpeed:
    def __init__(self):
        # Verify current RealHand configuration
        init_hand = InitRealHand()
        # Get current RealHand information
        left_hand ,left_hand_joint ,left_hand_type ,left_hand_force,left_hand_pose, left_hand_torque, left_hand_speed ,right_hand ,right_hand_joint ,right_hand_type ,right_hand_force,right_hand_pose, right_hand_torque, right_hand_speed,setting = init_hand.current_hand()
        if left_hand_joint != False and left_hand_type != False:
            # Initialize API
            self.hand = RealHandApi(hand_joint=left_hand_joint,hand_type=left_hand_type)
        if right_hand_joint != False and right_hand_type != False:
            # Initialize API
            self.hand = RealHandApi(hand_joint=right_hand_joint,hand_type=right_hand_type)
        self.get_speed()
    
    def get_speed(self):
        while True:
            speed = self.hand.get_speed()
            print(f"Current speed: {speed}")
            time.sleep(0.01)

if __name__ == "__main__":
    GetSpeed()
