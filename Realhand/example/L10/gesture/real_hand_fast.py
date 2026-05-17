#!/usr/bin/env python3
import sys,os,time,argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.append(target_dir)
from RealHand.real_hand_api import RealHandApi
'''
Fast finger movement
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
    while True:
        for i in range(10):
            if i % 2 == 0:
                pose = [255, 128, 255, 255, 255, 255, 128, 128, 128, 128]
            else:
                pose = [80, 80, 80, 80, 80, 80, 80, 80, 80, 80]
            print(f"Pose {i}: {pose}")
            # Add code to update pose here, e.g., send to hardware device
            time.sleep(0.1)  # Wait 0.1 seconds
            hand.finger_move(pose=pose)

if __name__ == "__main__":
    # python3 real_hand_fast.py --hand_type left --hand_joint L10 --can=can0
    main()