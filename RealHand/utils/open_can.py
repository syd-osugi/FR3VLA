#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
'''
Author: HJX
Date: 2025-04-01 14:09:21
LastEditors: Please set LastEditors
LastEditTime: 2025-04-11 09:15:31
FilePath: /Real_Hand_SDK_ROS/src/real_hand_sdk_ros/scripts/RealHand/utils/open_can.py
Description: 
symbol_custom_string_obkorol_copyright: 
'''
import sys,os,time,subprocess
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from color_msg import ColorMsg
from load_write_yaml import LoadWriteYaml
# from ament_index_python.packages import get_package_share_directory
import os


class OpenCan:
    def __init__(self,load_yaml=None):
        self.yaml = LoadWriteYaml()
        self.password = self.yaml.load_setting_yaml()["PASSWORD"]

    def open_can0(self):
        try:
            # Check whether can0 interface already exists and is up
            result = subprocess.run(
                ["ip", "link", "show", "can0"],
                check=True,
                text=True,
                capture_output=True
            )
            if "state UP" in result.stdout:
                return 
            # If not in UP state, configure the interface
            subprocess.run(
                ["sudo", "-S", "ip", "link", "set", "can0", "up", "type", "can", "bitrate", "1000000"],
                input=f"{self.password}\n",
                check=True,
                text=True,
                capture_output=True
            )
            
        except subprocess.CalledProcessError as e:
            pass
        except Exception as e:
            pass
    def open_can(self,can="can0"):
        try:
            # Check whether can interface already exists and is up
            result = subprocess.run(
                ["ip", "link", "show", can],
                check=True,
                text=True,
                capture_output=True
            )
            if "state UP" in result.stdout:
                return 
            # If not in UP state, configure the interface
            subprocess.run(
                ["sudo", "-S", "ip", "link", "set", can, "up", "type", "can", "bitrate", "1000000"],
                input=f"{self.password}\n",
                check=True,
                text=True,
                capture_output=True
            )
            
        except subprocess.CalledProcessError as e:
            pass
        except Exception as e:
            pass
            

    def is_can_up_sysfs(self, interface="can0"):
        # Check whether interface directory exists
        if not os.path.exists(f"/sys/class/net/{interface}"):
            return False
        # Read interface state
        try:
            with open(f"/sys/class/net/{interface}/flags", "r", encoding="utf-8") as f:
                flags = int(f.read().strip(), 16)
                IFF_UP = 0x1
                return (flags & IFF_UP) != 0
        except Exception as e:
            print(f"Error reading CAN interface state: {e}")
            return False
        
    def close_can0(self):
        try:
            # Check whether can0 interface exists
            result = subprocess.run(
                ["ip", "link", "show", "can0"],
                check=True,
                text=True,
                capture_output=True
            )
            
            # If interface exists and is in UP state, bring it down
            if "state UP" in result.stdout:
                subprocess.run(
                    ["sudo", "-S", "ip", "link", "set", "can0", "down"],
                    input=f"{self.password}\n",
                    check=True,
                    text=True,
                    capture_output=True
                )
                return True
            return False
            
        except subprocess.CalledProcessError as e:
            print(f"Error closing CAN interface: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False
    
    def close_can(self,can="can0"):
        try:
            # Check whether can interface exists
            result = subprocess.run(
                ["ip", "link", "show", can],
                check=True,
                text=True,
                capture_output=True
            )
            
            # If interface exists and is in UP state, bring it down
            if "state UP" in result.stdout:
                subprocess.run(
                    ["sudo", "-S", "ip", "link", "set", can, "down"],
                    input=f"{self.password}\n",
                    check=True,
                    text=True,
                    capture_output=True
                )
                return True
            return False
            
        except subprocess.CalledProcessError as e:
            print(f"Error closing CAN interface: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False
