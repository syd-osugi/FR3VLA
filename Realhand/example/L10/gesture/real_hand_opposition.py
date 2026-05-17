#!/usr/bin/env python3
import sys,os,time,argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.append(target_dir)
from RealHand.real_hand_api import RealHandApi
from RealHand.utils.load_write_yaml import LoadWriteYaml
from RealHand.utils.init_real_hand import InitRealHand
from RealHand.utils.color_msg import ColorMsg
'''
Cyclic thumb opposition with other fingers
'''
def main():
    parser = argparse.ArgumentParser(description='Process gesture parameters')
    parser.add_argument('--hand_type', choices=['left', 'right'], required=True, help='Specify left or right hand')
    parser.add_argument('--hand_joint', required=True, help='Specify RealHand model')
    parser.add_argument('--can', default="can0", help='Specify CAN ID')
    args = parser.parse_args()
    print(f"Hand Type: {args.hand_type}, Joint: {args.hand_joint}")

    hand_joint = args.hand_joint
    hand_type = args.hand_type
    can = args.can
    hand = RealHandApi(hand_joint=hand_joint,hand_type=hand_type, can=can)
    # Set speed
    hand.set_speed(speed=[100,80,80,80,80])
    # Finger pose data
    poses = [
        [255.0,70.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0], # Palm open
        [135.0,128.0,146.0,255.0,255.0,255.0,255.0,255.0,255.0,80.0], # Thumb touches index finger
        [255.0,70.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0], # Palm open
        [135.0,88.0,255.0,138.0,255.0,255.0,255.0,255.0,255.0,65.0], # Thumb touches middle finger
        [255.0,70.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0], # Palm open
        [135.0,63.0,255.0,255.0,140.0,255.0,255.0,255.0,255.0,40.0], # Thumb touches ring finger
        [255.0,70.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0], # Palm open
        [137.0,70.0,255.0,255.0,255.0,131.0,255.0,255.0,120.0,15.0], # Thumb touches pinky finger
        [255.0,70.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0,255.0], # Palm open
    ]
    while True:
        for pose in poses:
            hand.finger_move(pose=pose)
            time.sleep(1.3)


if __name__ == "__main__":
    # python3 real_hand_opposition.py --hand_type left --hand_joint L10 --can=can0
    main()