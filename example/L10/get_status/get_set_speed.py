#!/usr/bin/env python3
import sys,os,time
import argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.append(target_dir)

from RealHand.real_hand_api import RealHandApi
from RealHand.utils.load_write_yaml import LoadWriteYaml
from RealHand.utils.init_real_hand import InitRealHand
from RealHand.utils.color_msg import ColorMsg
'''
Currently, L10 does not listen for CAN commands for current speed, so real-time speed retrieval is temporarily not supported
'''
class GetSpeed:
    def __init__(self,hand_joint="L10",hand_type="left",speed=[180,180,180,180,180]):
        self.speed = speed
        self.hand_joint = hand_joint
        self.hand_type = hand_type
        self.hand = RealHandApi(hand_joint=hand_joint,hand_type=hand_type)
        self.set_speed()
        self.get_speed()
    
    def set_speed(self):
        if self.hand_joint == "L7":
            if len(self.speed) == 5:
                s = self.speed+[100,100]
            else:
                s = self.speed
            self.hand.set_speed(speed=s)
        else:
            self.hand.set_speed(speed=self.speed)
        time.sleep(1)
        ColorMsg(msg=f"Set speed: {self.speed}", color='green')
    def get_speed(self):
        #while True:
        speed = self.hand.get_speed()
        ColorMsg(msg=f"Current speed: {speed}", color='yellow')
        time.sleep(0.01)

if __name__ == "__main__":
    # python3 get_set_speed.py --hand_joint L10 --hand_type right --speed 100 123 211 121 222
    parser = argparse.ArgumentParser(description='GetSpeed Example')
    parser.add_argument('--hand_joint', type=str, default='L10',required=True, help='Finger joint type, default is L10')
    parser.add_argument('--hand_type', type=str, default='left',required=True, help='Hand type, default is left')
    parser.add_argument('--speed', 
                   nargs=5,  # Receives 5 parameters
                   type=int, 
                   default=[180]*5,
                   required=True,
                   help='Finger speed (5 integers), default is 180 180 180 180 180')

    args = parser.parse_args()
    GetSpeed(hand_joint=args.hand_joint, hand_type=args.hand_type,speed=args.speed)