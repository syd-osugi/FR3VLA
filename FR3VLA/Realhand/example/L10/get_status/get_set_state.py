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

class GetState:
    def __init__(self,hand_joint="L10",hand_type="left",position=[180,180,180,180,180]):
        self.position = position
        self.hand_joint = hand_joint
        self.hand_type = hand_type
        self.hand = RealHandApi(hand_joint=self.hand_joint,hand_type=self.hand_type)
        self.set_position()
        self.get_state()

    def set_position(self):
        #for i in range(15):
        if self.hand_joint == "L7":
            if len(self.position) == 5:
                p = self.position + [100, 100]
            else:
                p = self.position
            self.hand.finger_move(pose=p)
        else:
            self.hand.finger_move(pose=self.position)
        time.sleep(0.01)
        ColorMsg(msg=f"Set position: {self.position}", color='green')
            
    # Get current state
    def get_state(self):
        state = self.hand.get_state()
        print(f"Current state: {state}")
        time.sleep(0.01)

if __name__ == "__main__":
    # python3 get_set_state.py --hand_joint L10 --hand_type right --position 100 123 211 121 222 255 255 255 255 255
    parser = argparse.ArgumentParser(description='GetSpeed Example')
    parser.add_argument('--hand_joint', type=str, default='L10',required=True, help='Finger joint type, default is L10')
    parser.add_argument('--hand_type', type=str, default='left',required=True, help='Hand type, default is left')
    parser.add_argument('--position', 
                   nargs='+',  # Receives parameters
                   type=int, 
                   default=[180]*10,
                   required=True,
                   help='The number of finger position parameters varies by hand model: L7 has 5, L10 has 10, L20 has 20, L25 has 25')

    args = parser.parse_args()
    GetState(hand_joint=args.hand_joint,hand_type=args.hand_type,position=args.position)