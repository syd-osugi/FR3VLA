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
    parser.add_argument('--can', default="can0", help='Specify CAN ID')
    args = parser.parse_args()
    print(f"Hand Type: {args.hand_type}, Joint: {args.hand_joint}")

    hand_joint = args.hand_joint
    hand_type = args.hand_type
    can = args.can
        # Initialize API
    hand = RealHandApi(hand_joint=hand_joint,hand_type=hand_type, can=can)
    # Set speed
    speed = [120,250,250,250,250,250,250]
    hand.set_speed(speed=speed)
    ColorMsg(msg=f"Current speed set to:{speed}", color="green")
    pose = [[255,255,255,255,255,255,255],[255,255,0,255,255,255,255],[255,255,0,0,255,255,255],[255,255,0,0,0,255,255],[255,255,0,0,0,0,255],[72,90,0,0,0,0,55]]
    while True:
        for i in range(6):
            print("_-"*10)
            print(i)
            ColorMsg(msg=f"Current finger coordinates:{pose[i]}", color="green")
            hand.finger_move(pose=pose[i])
            time.sleep(3)


if __name__ == "__main__":
    # python3 real_hand_loop.py --hand_joint L7 --hand_type left --can can0
    main()