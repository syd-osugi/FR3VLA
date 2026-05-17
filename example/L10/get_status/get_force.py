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

class GetForce:
    def __init__(self,hand_joint="L10",hand_type="left"):
        self.hand_joint = hand_joint
        self.hand_type = hand_type
        self.touch_type = -1
        self.hand = RealHandApi(hand_joint=self.hand_joint,hand_type=self.hand_type)
        self.get_touch_type()
        self.get_force()
    
    def get_touch_type(self):
        t = self.hand.get_touch_type()
        if t == 2:
            ColorMsg(msg="Pressure sensor type is matrix pressure sensor", color='green')
        elif t == -1:
            ColorMsg(msg="No pressure sensor", color='red')
        self.touch_type = t
    def get_force(self):
        for i in range(3):
            touch = self.hand.get_matrix_touch()
        print(touch)

if __name__ == "__main__":
    # python3 get_force.py --hand_joint L10 --hand_type right
    parser = argparse.ArgumentParser(description='GetSpeed Example')
    parser.add_argument('--hand_joint', type=str, default='L10',required=True, help='Finger joint type, default is L10')
    parser.add_argument('--hand_type', type=str, default='left',required=True, help='Hand type, default is left')

    args = parser.parse_args()
    GetForce(hand_joint=args.hand_joint,hand_type=args.hand_type)