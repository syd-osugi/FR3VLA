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
Make a fist
'''
def main():
    parser = argparse.ArgumentParser(description='Process gesture parameters')
    parser.add_argument('--hand_type', choices=['left', 'right'], required=True, help='Specify left or right hand')
    parser.add_argument('--hand_joint', required=True, help='Specify RealHand model')
    parser.add_argument('--can', default="can0", help='Specify CAN interface')
    args = parser.parse_args()
    print(f"Hand Type: {args.hand_type}, Joint: {args.hand_joint}")

    hand_joint = args.hand_joint
    hand_type = args.hand_type
    can = args.can
    # Initialize API
    hand = RealHandApi(hand_joint=hand_joint,hand_type=hand_type,can=can)
    # Set speed
    speed = [120,250,250,250,250,250,250]
    hand.set_speed(speed=speed)
    ColorMsg(msg=f"Current: {hand_joint} {hand_type}, Speed set to: {speed}", color="green")
    # Finger pose data
    pose = [55, 90, 0, 0, 0, 0, 75]
    ColorMsg(msg=f"Current: {hand_joint} {hand_type}, Finger coordinates: {pose}", color="green")
    hand.finger_move(pose=pose)


if __name__ == "__main__":
    # python3 real_hand_fist.py --hand_joint L7 --hand_type left --can can0
    main()