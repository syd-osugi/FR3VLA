#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L30灵巧手上位机控制程序
基于CANFD协议的17自由度灵巧手控制系统

作者: AI Assistant
版本: 1.0
日期: 2025-07-28
"""

import sys
import os
import time
import threading,struct


from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum

import numpy as np


# 添加python3.11目录到路径
#sys.path.append(os.path.join(os.path.dirname(__file__), 'python3.11'))

# 导入CANFD库
from ctypes import *

# CANFD库常量和结构体定义
STATUS_OK = 0

class CanFD_Config(Structure):
    _fields_ = [("NomBaud", c_uint),
                ("DatBaud", c_uint),
                ("NomPres", c_ushort),
                ("NomTseg1", c_char),
                ("NomTseg2", c_char),
                ("NomSJW", c_char),
                ("DatPres", c_char),
                ("DatTseg1", c_char),
                ("DatTseg2", c_char),
                ("DatSJW", c_char),
                ("Config", c_char),
                ("Model", c_char),
                ("Cantype", c_char)]

class CanFD_Msg(Structure):
    _fields_ = [("ID", c_uint),
                ("TimeStamp", c_uint),
                ("FrameType", c_ubyte),
                ("DLC", c_ubyte),
                ("ExternFlag", c_ubyte),
                ("RemoteFlag", c_ubyte),
                ("BusSatus", c_ubyte),
                ("ErrSatus", c_ubyte),
                ("TECounter", c_ubyte),
                ("RECounter", c_ubyte),
                ("Data", c_ubyte*64)]

class Dev_Info(Structure):
    _fields_ = [("HW_Type", c_char*32),
                ("HW_Ser", c_char*32),
                ("HW_Ver", c_char*32),
                ("FW_Ver", c_char*32),
                ("MF_Date", c_char*32)]

# 协议常量定义
class DeviceID(Enum):
    RIGHT_HAND = 0x01
    LEFT_HAND = 0x02

# 数据单位转换常量 - 根据协议规范v2.0
POSITION_UNIT = 0.087  # 位置单位：0.087度/LSB
VELOCITY_UNIT = 0.732  # 速度单位：0.732RPM/LSB

class RegisterAddress(Enum):
    SYS_DEVICE_INFO = 0x00
    SYS_CALI_MODE = 0x01
    SYS_ERROR_STATUS = 0x02
    SYS_CURRENT_POS = 0x03
    SYS_CURRENT_VEL = 0x04
    SYS_CONFIG_STATUS = 0x05
    SYS_TARGET_POS = 0x06
    SYS_TARGET_VEL = 0x07
    SYS_TARGET_TORQUE = 0x08  # 目标力矩寄存器
    TACTILE_THUMB_DATA1 = 0x09
    TACTILE_THUMB_DATA2 = 0x0A
    TACTILE_INDEX_DATA1 = 0x0B
    TACTILE_INDEX_DATA2 = 0x0C
    TACTILE_MIDDLE_DATA1 = 0x0D
    TACTILE_MIDDLE_DATA2 = 0x0E
    TACTILE_RING_DATA1 = 0x0F
    TACTILE_RING_DATA2 = 0x10
    TACTILE_PINKY_DATA1 = 0x11
    TACTILE_PINKY_DATA2 = 0x12

# 关节信息定义
@dataclass
class JointInfo:
    id: int
    name: str
    finger: str
    min_pos: int = -32768
    max_pos: int = 32767
    current_pos: int = 0
    target_pos: int = 0
    current_vel: int = 0
    target_vel: int = 0
    target_acc: int = 0
    error_status: int = 0
    config_status: int = 0

# 手指关节定义 - 按照协议规范v2.0的电机ID分配
JOINT_DEFINITIONS = [
    # 拇指 (4 DOF) - 电机ID 1-4，根据C代码设置限位
    JointInfo(1, "指根弯曲", "拇指", -100, 500),      # 电机ID:1 THUMB_MCP
    JointInfo(2, "指尖弯曲", "拇指", -100, 600),       # 电机ID:2 THUMB_IP
    JointInfo(3, "侧摆", "拇指", -100, 400),          # 电机ID:3 THUMB_ABD
    JointInfo(4, "旋转", "拇指", -100, 1000),          # 电机ID:4 THUMB_CMC

    # 无名指 (3 DOF) - 电机ID 5-7
    JointInfo(5, "侧摆运动", "无名指", -300, 300),        # 电机ID:5 RING_ABD
    JointInfo(6, "指尖弯曲", "无名指", -100, 600),   # 电机ID:6 RING_PIP
    JointInfo(7, "指根弯曲", "无名指", -100, 600),   # 电机ID:7 RING_MCP

    # 中指 (3 DOF) - 电机ID 8,9,13
    JointInfo(8, "指根弯曲", "中指", -100, 600),     # 电机ID:8 MIDDLE_MCP
    JointInfo(9, "指尖弯曲", "中指", -100, 600),     # 电机ID:9 MIDDLE_PIP
    JointInfo(13, "侧摆", "中指", -300, 300),        # 电机ID:13 MIDDLE_ABD

    # 小指 (3 DOF) - 电机ID 10-12
    JointInfo(10, "指根弯曲", "小指", -100, 600),    # 电机ID:10 PINKY_MCP
    JointInfo(11, "指尖弯曲", "小指", -100, 600),    # 电机ID:11 PINKY_DIP
    JointInfo(12, "侧摆", "小指", -300, 300),        # 电机ID:12 PINKY_ABD

    # 食指 (3 DOF) - 电机ID 14-16
    JointInfo(14, "侧摆运动", "食指", -300, 300),        # 电机ID:14 INDEX_ABD
    JointInfo(15, "指根弯曲", "食指", -100, 600),    # 电机ID:15 INDEX_MCP
    JointInfo(16, "指尖弯曲", "食指", -100, 600),    # 电机ID:16 INDEX_PIP

    # 手腕 (1 DOF) - 电机ID 17，暂时使用默认范围
    JointInfo(17, "俯仰", "手腕", -1000, 1000),        # 电机ID:17 HAND_WRITE
]

class CANFDCommunication:
    """CANFD通信类"""

    def __init__(self):
        # # 根据Python架构选择对应的DLL
        # import platform
        # arch = platform.architecture()[0]
        # if arch == '64bit':
        #     # 如果有64位版本的DLL，放在python3.11/x64/目录下
        #     self.dll_path = os.path.join(os.path.dirname(__file__), 'python3.11', 'x64', 'hcanbus.dll')
        #     if not os.path.exists(self.dll_path):
        #         # 回退到32位版本（会失败，但给出明确提示）
        #         self.dll_path = os.path.join(os.path.dirname(__file__), 'python3.11', 'hcanbus.dll')
        # else:
        #     self.dll_path = os.path.join(os.path.dirname(__file__), 'python3.11', 'hcanbus.dll')
        self.canDLL = None
        self.channel = 0
        self.device_id = DeviceID.RIGHT_HAND.value
        self.is_connected = False
        
        self.dlc2len = [0,1,2,3,4,5,6,7,8,12,16,20,24,32,48,64]

    def close(self):
        self.canDLL.CAN_CloseDevice(self.channel)
    def initialize(self) -> bool:
        """初始化CANFD通信"""
        try:
            print("正在初始化CANFD通信...")

            # # 检查DLL文件是否存在
            # if not os.path.exists(self.dll_path):
            #     print(f"错误: 找不到CANFD库文件: {self.dll_path}")
            #     return False

            # # 加载DLL
            # print(f"加载CANFD库: {self.dll_path}")
            # 载入动态库（优先使用SDK内置路径，回退系统路径）
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "canfd"))
            local_libusb = os.path.join(base_dir, "libusb-1.0.so")
            local_libcanbus = os.path.join(base_dir, "libcanbus.so")
            sys_libusb = "/usr/local/lib/libusb-1.0.so"
            sys_libcanbus = "/usr/local/lib/libcanbus.so"

            libusb_path = local_libusb if os.path.exists(local_libusb) else sys_libusb
            libcanbus_path = local_libcanbus if os.path.exists(local_libcanbus) else sys_libcanbus

            CDLL(libusb_path, RTLD_GLOBAL)
            time.sleep(0.1)  # 确保库加载完成
            self.canDLL = cdll.LoadLibrary(libcanbus_path)

            # 扫描设备（添加超时控制）
            print("=" * 50)
            print("开始扫描CANFD设备...")
            start_time = time.time()

            try:
                ret = self.canDLL.CAN_ScanDevice()
                scan_time = time.time() - start_time

                print(f"扫描完成: 找到 {ret} 个设备 (耗时: {scan_time:.3f}s)")

                if ret <= 0:
                    print("❌ 错误: 未找到CANFD设备")
                    print("   请检查:")
                    print("   1. CANFD适配器是否连接")
                    print("   2. 设备驱动是否安装")
                    print("   3. 设备是否被其他程序占用")
                    return False
                else:
                    print(f"✅ 成功找到 {ret} 个CANFD设备")

            except Exception as e:
                scan_time = time.time() - start_time
                print(f"❌ 扫描设备异常 (耗时: {scan_time:.3f}s): {e}")
                return False

            # 打开设备（添加超时控制）
            print(f"正在打开CANFD设备通道 {self.channel}...")
            start_time = time.time()

            try:
                ret = self.canDLL.CAN_OpenDevice(0,self.channel)
                open_time = time.time() - start_time

                if ret != STATUS_OK:
                    print(f"❌ 打开设备失败，错误码: {ret} (耗时: {open_time:.3f}s)")
                    print(f"   可能原因:")
                    print(f"   1. 设备通道 {self.channel} 不存在")
                    print(f"   2. 设备已被其他程序占用")
                    print(f"   3. 设备权限不足")
                    return False
                else:
                    print(f"✅ 设备通道 {self.channel} 打开成功 (耗时: {open_time:.3f}s)")

            except Exception as e:
                open_time = time.time() - start_time
                print(f"❌ 打开设备异常 (耗时: {open_time:.3f}s): {e}")
                return False

            # 读取设备信息
            self._read_device_info()

            # 配置CANFD参数
            print("正在配置CANFD参数...")
            print("  仲裁段波特率: 1Mbps")
            print("  数据段波特率: 5Mbps")

            start_time = time.time()

            try:
                # 1Mbps仲裁段，5Mbps数据段
                can_config = CanFD_Config(
                    1000000,  # NomBaud: 仲裁段波特率 1Mbps
                    5000000,  # DatBaud: 数据段波特率 5Mbps
                    0x0,      # NomPres: 仲裁段预分频
                    0x0,      # NomTseg1: 仲裁段时间段1
                    0x0,      # NomTseg2: 仲裁段时间段2
                    0x0,      # NomSJW: 仲裁段同步跳转宽度
                    0x0,      # DatPres: 数据段预分频
                    0x0,      # DatTseg1: 数据段时间段1
                    0x0,      # DatTseg2: 数据段时间段2
                    0x0,      # DatSJW: 数据段同步跳转宽度
                    0x0,      # Config: 配置标志
                    0x0,      # Model: 模式
                    0x1       # Cantype: CANFD类型
                )

                ret = self.canDLL.CANFD_Init(0,self.channel, byref(can_config))
                config_time = time.time() - start_time

                if ret != STATUS_OK:
                    print(f"❌ CANFD初始化失败，错误码: {ret} (耗时: {config_time:.3f}s)")
                    print("   正在关闭设备...")
                    self.canDLL.CAN_CloseDevice(0,self.channel)
                    return False
                else:
                    print(f"✅ CANFD配置成功 (耗时: {config_time:.3f}s)")

            except Exception as e:
                config_time = time.time() - start_time
                print(f"❌ CANFD配置异常 (耗时: {config_time:.3f}s): {e}")
                try:
                    self.canDLL.CAN_CloseDevice(0,self.channel)
                except:
                    pass
                return False

            # 设置接收过滤器（接收所有消息）
            print("正在设置接收过滤器...")
            start_time = time.time()

            try:
                ret = self.canDLL.CAN_SetFilter(self.channel, 0, 0, 0, 0, 1)
                filter_time = time.time() - start_time

                if ret != STATUS_OK:
                    print(f"❌ 设置过滤器失败，错误码: {ret} (耗时: {filter_time:.3f}s)")
                    print("   正在关闭设备...")
                    self.canDLL.CAN_CloseDevice(self.channel)
                    return False
                else:
                    print(f"✅ 过滤器设置成功 (耗时: {filter_time:.3f}s)")

            except Exception as e:
                filter_time = time.time() - start_time
                print(f"❌ 设置过滤器异常 (耗时: {filter_time:.3f}s): {e}")
                try:
                    self.canDLL.CAN_CloseDevice(self.channel)
                except:
                    pass
                return False

            self.is_connected = True
            print("✅ CANFD通信初始化完成")
            print("=" * 50)
            return True

        except OSError as e:
            if "193" in str(e):
                print("错误: DLL架构不匹配")
                print("请确认CANFD库文件与Python架构匹配")
            else:
                print(f"错误: 加载CANFD库失败: {e}")
            return False
        except Exception as e:
            print(f"CANFD初始化失败: {e}")
            return False

    def query_device_type(self,device_id=0x01) -> Optional[str]:
        """查询设备类型（左手/右手）"""
        if not self.is_connected:
            print("❌ 设备未连接，无法查询设备类型")
            return None

        print("🔍 正在查询设备类型...",flush=True)

        # 尝试查询右手设备
        
        if device_id == 0x01:
            print("   查询右手设备 (ID: 0x01)...", flush=True)
            right_hand_response = self._query_single_device(DeviceID.RIGHT_HAND.value)
        if device_id == 0x02:
        # 尝试查询左手设备
            print("   查询左手设备 (ID: 0x02)...", flush=True)
            left_hand_response = self._query_single_device(DeviceID.LEFT_HAND.value)

        # 分析响应结果
        # if right_hand_response and left_hand_response:
        #     print("⚠️ 检测到左右手设备都有响应，默认选择右手")
        #     self.device_id = DeviceID.RIGHT_HAND.value
        #     return "右手"
        if right_hand_response:
            print("✅ 检测到右手设备", flush=True)
            self.device_id = DeviceID.RIGHT_HAND.value
            return "右手"
        elif left_hand_response:
            print("✅ 检测到左手设备", flush=True)
            self.device_id = DeviceID.LEFT_HAND.value
            return "左手"
        else:
            print("❌ 未检测到任何灵巧手设备响应", flush=True)
            return None

    def _query_single_device(self, device_id: int) -> bool:
        """查询单个设备是否存在"""
        try:
            # 多次尝试查询
            for attempt in range(1):
                #print(f"     尝试 {attempt + 1}/3...")

                # 临时设置设备ID用于发送
                original_device_id = self.device_id
                self.device_id = device_id

                # 发送设备信息查询命令
                success = self.send_message(RegisterAddress.SYS_DEVICE_INFO.value, b'', False)

                # 恢复原始设备ID
                self.device_id = original_device_id

                if not success:
                    print(f"     发送查询命令失败", flush=True)
                    continue

                # 等待响应
                import time
                time.sleep(0.1)  # 100ms等待

                # 接收响应 - 不过滤设备ID，接收所有消息
                messages = self.receive_messages(200, filter_device_id=False)

                # 检查是否有来自目标设备的响应
                for frame_id, data in messages:
                    response_device_id = (frame_id >> 21) & 0xFF
                    register_addr = (frame_id >> 13) & 0xFF

                    if (response_device_id == device_id and
                        register_addr == RegisterAddress.SYS_DEVICE_INFO.value and
                        len(data) > 0):
                        print(f"     ✅ 设备 0x{device_id:02X} 响应正常 (数据长度: {len(data)})", flush=True)

                        # 检查数据是否全为0
                        if all(b == 0 for b in data):
                            print(f"     ⚠️ 设备信息数据全为0，可能设备信息未初始化", flush=True)
                            # 即使数据为0，也认为设备存在并响应
                            # 根据查询的设备ID来判断类型
                            device_type = "右手" if device_id == 0x01 else "左手"
                            print(f"     根据查询ID判断设备类型: {device_type}", flush=True)
                            return True
                        else:
                            # 解析设备信息
                            try:
                                if len(data) >= 50:
                                    product_model = data[0:10].decode('utf-8', errors='ignore').strip('\x00')
                                    # 手型标志位在第51字节（索引50）：1=右手，0=左手
                                    hand_type = "右手" if len(data) > 50 and data[50] == 1 else "左手"
                                    print(f"     设备信息: {product_model}, 类型: {hand_type}", flush=True)
                                else:
                                    print(f"     数据长度不足，无法解析设备信息", flush=True)
                            except Exception as e:
                                print(f"     解析设备信息失败: {e}", flush=True)
                            return True

                print(f"     第 {attempt + 1} 次查询无响应", flush=True)
                time.sleep(0.1)  # 重试间隔

            print(f"     ❌ 设备 0x{device_id:02X} 无响应,请检查配置文件是否正确", flush=True)
            return False

        except Exception as e:
            print(f"     ❌ 查询设备 0x{device_id:02X} 异常: {e}", flush=True)
            return False

        except OSError as e:
            if "193" in str(e):
                print("错误: DLL架构不匹配", flush=True)
                print("请确认CANFD库文件与Python架构匹配", flush=True)
            else:
                print(f"错误: 加载CANFD库失败: {e}", flush=True)
            return False
        except Exception as e:
            print(f"CANFD初始化失败: {e}", flush=True)
            return False

    def _read_device_info(self):
        """读取并显示设备信息"""
        try:
            devinfo = Dev_Info()
            ret = self.canDLL.CAN_ReadDevInfo(self.channel, byref(devinfo))
            if ret == STATUS_OK:
                print("\n设备信息:")
                print(f"  设备型号: {devinfo.HW_Type.decode('utf-8', errors='ignore').strip()}")
                print(f"  序列号  : {devinfo.HW_Ser.decode('utf-8', errors='ignore').strip()}")
                print(f"  硬件版本: {devinfo.HW_Ver.decode('utf-8', errors='ignore').strip()}")
                print(f"  固件版本: {devinfo.FW_Ver.decode('utf-8', errors='ignore').strip()}")
                print(f"  生产日期: {devinfo.MF_Date.decode('utf-8', errors='ignore').strip()}")
                print()
            else:
                print("警告: 无法读取设备信息")
        except Exception as e:
            print(f"读取设备信息失败: {e}")

    def create_frame_id(self, device_id: int, register_addr: int, is_write: bool) -> int:
        """创建CANFD扩展帧ID"""
        frame_id = (device_id << 21) | (register_addr << 13) | (1 if is_write else 0) << 12
        return frame_id

    def send_message(self, register_addr: int, data: bytes, is_write: bool = True) -> bool:
        """发送CANFD消息"""
        if not self.is_connected:
            print("错误: CANFD未连接")
            return False

        try:
            frame_id = self.create_frame_id(self.device_id, register_addr, is_write)

            # 限制数据长度
            data_len = min(len(data), 64)

            # 创建数据数组并初始化为0
            data_array = (c_ubyte * 64)()
            for i in range(64):
                data_array[i] = 0

            # 填充实际数据
            for i, byte_val in enumerate(data[:data_len]):
                data_array[i] = byte_val

            # 计算DLC (Data Length Code)
            dlc = self._get_dlc_from_length(data_len)

            # 创建消息
            msg = CanFD_Msg(
                frame_id,     # ID
                0,            # TimeStamp
                4,            # FrameType (CANFD)
                dlc,          # DLC
                1,            # ExternFlag (扩展帧)
                0,            # RemoteFlag
                0,            # BusSatus
                0,            # ErrSatus
                0,            # TECounter
                0,            # RECounter
                data_array    # Data
            )
            #print(f"{msg.ID}-{msg.FrameType}-{msg.DLC}-{msg.ExternFlag}-{msg.Data}")
            # 发送消息
            ret = self.canDLL.CANFD_Transmit(0,self.channel, byref(msg), 1, 100)
            if ret == 1:
                # 根据寄存器类型显示不同的详细信息
                '''
                if register_addr == RegisterAddress.SYS_TARGET_POS.value:
                    print(f"     ✅ 位置命令发送成功:")
                    print(f"        CAN ID: 0x{frame_id:08X}")
                    print(f"        设备ID: 0x{self.device_id:02X}")
                    print(f"        寄存器: 0x{register_addr:02X} (目标位置)")
                    print(f"        数据长度: {data_len}字节")
                    print(f"        数据内容: {data.hex().upper()}")
                elif register_addr == RegisterAddress.SYS_TARGET_VEL.value:
                    print(f"     ✅ 速度命令发送成功:")
                    print(f"        CAN ID: 0x{frame_id:08X}")
                    print(f"        设备ID: 0x{self.device_id:02X}")
                    print(f"        寄存器: 0x{register_addr:02X} (目标速度)")
                    print(f"        数据长度: {data_len}字节")
                    print(f"        数据内容: {data.hex().upper()}")
                elif register_addr == RegisterAddress.SYS_TARGET_TORQUE.value:
                    print(f"     ✅ 力矩命令发送成功:")
                    print(f"        CAN ID: 0x{frame_id:08X}")
                    print(f"        设备ID: 0x{self.device_id:02X}")
                    print(f"        寄存器: 0x{register_addr:02X} (目标力矩)")
                    print(f"        数据长度: {data_len}字节")
                    print(f"        数据内容: {data.hex().upper()}")
                else:
                    print(f"     ✅ 消息发送成功:")
                    print(f"        CAN ID: 0x{frame_id:08X}")
                    print(f"        设备ID: 0x{self.device_id:02X}")
                    print(f"        寄存器: 0x{register_addr:02X}")
                    print(f"        数据长度: {data_len}字节")
                    if data_len > 0:
                        print(f"        数据内容: {data.hex().upper()}")
                '''
                return True
            else:
                print(f"     ❌ 发送失败:")
                print(f"        FRAME_ID: 0x{frame_id:02X}")
                print(f"        寄存器: 0x{register_addr:02X}")
                print(f"        返回值: {ret}")
                print(f"        数据长度: {data_len}字节")
                return False

        except Exception as e:
            print(f"发送消息异常: {e}")
            return False

    def _get_dlc_from_length(self, length: int) -> int:
        """根据数据长度获取DLC值"""
        if length <= 8:
            return length
        elif length <= 12:
            return 9
        elif length <= 16:
            return 10
        elif length <= 20:
            return 11
        elif length <= 24:
            return 12
        elif length <= 32:
            return 13
        elif length <= 48:
            return 14
        else:
            return 15  # 64字节

    def receive_messages(self, timeout_ms: int = 100, filter_device_id: bool = True) -> List[Tuple[int, bytes]]:
        """接收CANFD消息

        Args:
            timeout_ms: 超时时间(毫秒)
            filter_device_id: 是否过滤设备ID，False时接收所有消息
        """
        if not self.is_connected:
            return []

        try:
            # 创建接收缓冲区
            from ctypes import POINTER

            class CanFD_Msg_ARRAY(Structure):
                _fields_ = [('SIZE', c_uint16), ('STRUCT_ARRAY', POINTER(CanFD_Msg))]

                def __init__(self, num_of_structs):
                    self.STRUCT_ARRAY = cast((CanFD_Msg * num_of_structs)(), POINTER(CanFD_Msg))
                    self.SIZE = num_of_structs
                    self.ADDR = self.STRUCT_ARRAY[0]

            receive_buffer = CanFD_Msg_ARRAY(100)

            # 接收消息
            ret = self.canDLL.CANFD_Receive(0,self.channel, byref(receive_buffer.ADDR), 100, timeout_ms)

            messages = []
            if ret > 0:
                #print(f"     接收到 {ret} 条消息")
                for i in range(ret):
                    msg = receive_buffer.STRUCT_ARRAY[i]

                    # 检查消息有效性
                    if msg.DLC >= len(self.dlc2len):
                        print(f"     警告: 无效的DLC值: {msg.DLC}")
                        continue

                    data_len = self.dlc2len[msg.DLC]
                    data = bytes(msg.Data[:data_len])

                    # 解析帧ID获取寄存器地址
                    response_device_id = (msg.ID >> 21) & 0xFF
                    register_addr = (msg.ID >> 13) & 0xFF
                    is_write = (msg.ID >> 12) & 0x1

                    #print(f"     消息 {i+1}: ID=0x{msg.ID:08X}, 设备=0x{response_device_id:02X}, 寄存器=0x{register_addr:02X}, 长度={data_len}")
                    #print(f"     数据: {data.hex().upper()}")

                    # 根据filter_device_id参数决定是否过滤
                    if not filter_device_id or response_device_id == self.device_id:
                        messages.append((msg.ID, data))
                    else:
                        print(f"     过滤掉设备0x{response_device_id:02X}的消息 (当前目标设备: 0x{self.device_id:02X})")
            else:
                print(f"     未接收到任何消息 (超时: {timeout_ms}ms)")

            return messages

        except Exception as e:
            print(f"接收消息异常: {e}")
            return []

    def close(self):
        """关闭CANFD连接"""
        if self.canDLL and self.is_connected:
            try:
                print("关闭CANFD连接...")
                self.canDLL.CAN_CloseDevice(self.channel)
                print("CANFD连接已关闭")
            except Exception as e:
                print(f"关闭CANFD连接失败: {e}")
            finally:
                self.is_connected = False

    def check_connection(self) -> bool:
        """检查连接状态"""
        if not self.is_connected or not self.canDLL:
            return False

        try:
            # 尝试发送一个简单的查询命令来检测连接
            test_result = self.send_message(RegisterAddress.SYS_ERROR_STATUS.value, b'', False)
            return test_result
        except Exception:
            return False

    def reconnect(self) -> bool:
        """重新连接"""
        print("尝试重新连接CANFD设备...")
        self.close()
        time.sleep(1)  # 等待1秒
        return self.initialize()
    

    

class DexterousHandModel:
    """灵巧手数据模型"""

    def __init__(self):
        self.joints = {joint.id: joint for joint in JOINT_DEFINITIONS}
        self.device_info = None
        self.calibration_mode = 0
        self.tactile_data = {
            'thumb': np.zeros((6, 12)),
            'index': np.zeros((6, 12)),
            'middle': np.zeros((6, 12)),
            'ring': np.zeros((6, 12)),
            'pinky': np.zeros((6, 12))
        }
        self.last_update_time = time.time()
        self.target_torques = [500] * 17 # Add target torques

    def update_joint_positions(self, positions: List[int]):
        """更新关节位置"""
        for i, pos in enumerate(positions[:17]):
            motor_id = i + 1  # 数组索引转换为电机ID (0->1, 1->2, ...)
            if motor_id in self.joints:
                self.joints[motor_id].current_pos = pos
        self.last_update_time = time.time()

    def update_joint_velocities(self, velocities: List[int]):
        """更新关节速度"""
        for i, vel in enumerate(velocities[:17]):
            motor_id = i + 1  # 数组索引转换为电机ID
            if motor_id in self.joints:
                self.joints[motor_id].current_vel = vel

    def update_error_status(self, errors: List[int]):
        """更新错误状态"""
        for i, error in enumerate(errors[:17]):
            motor_id = i + 1  # 数组索引转换为电机ID
            if motor_id in self.joints:
                self.joints[motor_id].error_status = error

    def update_tactile_data(self, finger: str, data: np.ndarray):
        """更新触觉传感器数据"""
        if finger in self.tactile_data:
            self.tactile_data[finger] = data.reshape((6, 12))

    
    def set_target_positions(self, values) -> bool:
        """设置 set_target_positions，只接受长度为17的list"""
        if not isinstance(values, (list, tuple)) or len(values) != 17:
            print("输入必须是长度17的list或tuple")
            return False
        data = bytearray(2*17)
        for motor_id, val in enumerate(values, start=1):
            index = (motor_id - 1) * 2
            if 2 == 1:
                # 单字节
                val_int = int(val)
                val_int = max(0, min(255, val_int))
                data[index] = val_int
            else:
                # 双字节
                min_a, max_a = self.joint_limits.get(motor_id, (-1000, 1000))
                val = max(min_a, min(max_a, float(val)))
                val_int = int(val)
                data[index:index+2] = val_int.to_bytes(2, byteorder='little', signed=True)
        return self.write_register(self.REG_SYS_TARGET_POS, data)
    def set_target_positions(self, positions: List[int]):
        """设置目标位置"""
        for i, pos in enumerate(positions[:17]):
            motor_id = i + 1  # 数组索引转换为电机ID
            if motor_id in self.joints:
                self.joints[motor_id].target_pos = pos

    def get_joint_by_finger(self, finger: str) -> List[JointInfo]:
        """根据手指名称获取关节"""
        return [joint for joint in self.joints.values() if joint.finger == finger]

    def get_all_current_positions(self) -> List[int]:
        """获取所有关节当前位置"""
        # 按照电机ID 1-17的顺序返回位置
        positions = [0] * 17
        for joint in self.joints.values():
            if 1 <= joint.id <= 17:
                positions[joint.id - 1] = joint.current_pos
        return positions

    def get_all_target_positions(self) -> List[int]:
        """获取所有关节目标位置"""
        # 按照电机ID 1-17的顺序返回位置
        positions = [0] * 17
        for joint in self.joints.values():
            if 1 <= joint.id <= 17:
                positions[joint.id - 1] = joint.target_pos
        return positions

class DexterousHandController:
    """灵巧手控制器"""
    # 设备ID
    DEVICE_ID_RIGHT = 0x01
    DEVICE_ID_LEFT = 0x02

    # 寄存器地址
    REG_SYS_DEVICE_INFO = 0x00
    REG_SYS_CALI_MODE = 0x01
    REG_SYS_ERROR_STATUS = 0x02
    REG_SYS_CURRENT_POS = 0x03
    REG_SYS_TARGET_POS = 0x06

    # 读写标志位
    RW_READ = 0
    RW_WRITE = 1

    # CANFD ID位偏移
    DEVICE_ID_SHIFT = 21
    REG_ADDR_SHIFT = 13
    RW_FLAG_SHIFT = 12
    def __init__(self,device_id=0x01):
        self.comm = CANFDCommunication()
        self.model = DexterousHandModel()
        self.is_running = False
        self.update_thread = None
        self.device_id = device_id
        self.update_interval = 0.01  # 10ms更新间隔
        self.dlc2len = [0,1,2,3,4,5,6,7,8,12,16,20,24,32,48,64]
        self.joint_limits = {
            1: (-100, 500), 2: (-100, 600), 3: (-100, 400), 4: (-100, 1000),
            5: (-300, 300), 6: (-100, 600), 7: (-100, 600),
            8: (-100, 600), 9: (-100, 600), 13: (-300, 300),
            10: (-100, 600), 11: (-100, 600), 12: (-300, 300),
            14: (-300, 300), 15: (-100, 600), 16: (-100, 600),
            17: (-1000, 1000)
        }

    def connect(self) -> Tuple[bool, Optional[str]]:
        """连接灵巧手，返回(连接成功, 设备类型)"""
        print(f"🔗 控制器开始连接")

        try:
            # 初始化CANFD通信
            result = self.comm.initialize()
            if not result:
                print("❌ CANFD通信初始化失败", flush=True)
                return False, None

            # 查询设备类型
            device_type = self.comm.query_device_type(device_id=self.device_id)
            if device_type:
                print(f"✅ 控制器连接成功，检测到设备类型: {device_type}", flush=True)
                return True, device_type
            else:
                print("❌ 未检测到灵巧手设备", flush=True)
                #self.comm.close()
                return False, None

        except Exception as e:
            print(f"❌ 控制器连接异常: {e}")
            return False, None

    def disconnect(self):
        """断开连接"""
        self.stop_monitoring()
        self.comm.close()

    def start_monitoring(self):
        """开始监控线程"""
        if not self.is_running:
            self.is_running = True
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()

    def stop_monitoring(self):
        """停止监控线程"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=1.0)

    def _update_loop(self):
        """更新循环"""
        connection_check_counter = 0
        connection_check_interval = 100  # 每100次循环检查一次连接

        while self.is_running:
            try:
                # 定期检查连接状态
                connection_check_counter += 1
                if connection_check_counter >= connection_check_interval:
                    connection_check_counter = 0
                    if not self.comm.check_connection():
                        print("检测到连接断开，尝试重连...")
                        if not self.comm.reconnect():
                            print("重连失败，暂停数据更新")
                            time.sleep(1)
                            continue

                # 读取当前位置
                self._read_current_positions()

                # 读取当前速度
                self._read_current_velocities()

                # 读取错误状态
                self._read_error_status()

                # 读取触觉传感器数据（降低频率）
                if connection_check_counter % 10 == 0:  # 每10次循环读取一次触觉数据
                    self._read_tactile_data()

                time.sleep(self.update_interval)

            except Exception as e:
                print(f"更新循环错误: {e}")
                time.sleep(0.1)

    def _read_current_positions(self):
        """读取当前位置"""
        try:
            # 发送读取位置命令
            if self.comm.send_message(RegisterAddress.SYS_CURRENT_POS.value, b'', False):
                # 等待响应
                time.sleep(0.005)  # 5ms等待

                # 接收响应
                messages = self.comm.receive_messages(50)
                for frame_id, data in messages:
                    if self._is_position_response(frame_id):
                        if len(data) >= 34:  # 17个关节 * 2字节
                            positions = self._parse_position_data(data)
                            if len(positions) == 17:
                                self.model.update_joint_positions(positions)
                            else:
                                print(f"警告: 位置数据长度不正确: {len(positions)}")
                        else:
                            print(f"警告: 位置数据长度不足: {len(data)}")
        except Exception as e:
            print(f"读取位置失败: {e}")

    def _read_current_velocities(self):
        """读取当前速度"""
        try:
            if self.comm.send_message(RegisterAddress.SYS_CURRENT_VEL.value, b'', False):
                time.sleep(0.005)  # 5ms等待

                messages = self.comm.receive_messages(50)
                for frame_id, data in messages:
                    if self._is_velocity_response(frame_id):
                        if len(data) >= 34:  # 17个关节 * 2字节
                            velocities = self._parse_velocity_data(data)
                            if len(velocities) == 17:
                                self.model.update_joint_velocities(velocities)
        except Exception as e:
            print(f"读取速度失败: {e}")

    def _read_error_status(self):
        """读取错误状态"""
        try:
            if self.comm.send_message(RegisterAddress.SYS_ERROR_STATUS.value, b'', False):
                time.sleep(0.005)  # 5ms等待

                messages = self.comm.receive_messages(50)
                for frame_id, data in messages:
                    if self._is_error_response(frame_id):
                        if len(data) >= 17:  # 17个关节状态
                            errors = list(data[:17])
                            self.model.update_error_status(errors)
        except Exception as e:
            print(f"读取错误状态失败: {e}")

    def _read_tactile_data(self):
        """读取触觉传感器数据"""
        # 读取各个手指的触觉数据
        tactile_registers = [
            (RegisterAddress.TACTILE_THUMB_DATA1.value, RegisterAddress.TACTILE_THUMB_DATA2.value, 'thumb'),
            (RegisterAddress.TACTILE_INDEX_DATA1.value, RegisterAddress.TACTILE_INDEX_DATA2.value, 'index'),
            (RegisterAddress.TACTILE_MIDDLE_DATA1.value, RegisterAddress.TACTILE_MIDDLE_DATA2.value, 'middle'),
            (RegisterAddress.TACTILE_RING_DATA1.value, RegisterAddress.TACTILE_RING_DATA2.value, 'ring'),
            (RegisterAddress.TACTILE_PINKY_DATA1.value, RegisterAddress.TACTILE_PINKY_DATA2.value, 'pinky'),
        ]

        for reg1, reg2, finger in tactile_registers:
            # 读取第一部分数据
            self.comm.send_message(reg1, b'', False)
            # 读取第二部分数据
            self.comm.send_message(reg2, b'', False)

    def _is_position_response(self, frame_id: int) -> bool:
        """判断是否为位置响应"""
        register_addr = (frame_id >> 13) & 0xFF
        return register_addr == RegisterAddress.SYS_CURRENT_POS.value

    def _is_velocity_response(self, frame_id: int) -> bool:
        """判断是否为速度响应"""
        register_addr = (frame_id >> 13) & 0xFF
        return register_addr == RegisterAddress.SYS_CURRENT_VEL.value

    def _is_error_response(self, frame_id: int) -> bool:
        """判断是否为错误状态响应"""
        register_addr = (frame_id >> 13) & 0xFF
        return register_addr == RegisterAddress.SYS_ERROR_STATUS.value

    def _parse_position_data(self, data: bytes) -> List[int]:
        """解析位置数据

        数据格式：34字节，每2字节对应一个电机的当前位置
        单位：0.087度/LSB (根据协议规范v2.0)
        数据类型：int16_t，小端序，有符号
        """
        positions = []
        for i in range(0, min(34, len(data)), 2):
            if i + 1 < len(data):
                # 解析16位有符号整数，小端序
                pos = int.from_bytes(data[i:i+2], byteorder='little', signed=True)
                # 注意：这里返回原始值，如需要角度值可乘以POSITION_UNIT(0.087)
                positions.append(pos)
        return positions


    def _parse_velocity_data(self, data: bytes) -> List[int]:
        """解析速度数据

        数据格式：34字节，每2字节对应一个电机的当前速度
        单位：0.732RPM/LSB (根据协议规范v2.0)
        数据类型：int16_t，小端序，有符号
        """
        velocities = []
        for i in range(0, min(34, len(data)), 2):
            if i + 1 < len(data):
                # 解析16位有符号整数，小端序
                vel = int.from_bytes(data[i:i+2], byteorder='little', signed=True)
                # 注意：这里返回原始值，如需要RPM值可乘以VELOCITY_UNIT(0.732)
                velocities.append(vel)
        return velocities

    def set_joint_positions(self, positions: List[int]) -> bool:
        """设置关节位置"""
        if len(positions) != 17:
            print(f"❌ 位置数据长度错误: 期望17个，实际{len(positions)}个")
            return False

        #print(f"📤 设置关节位置命令:")
        #print(f"   输入位置数组: {positions}")

        # 更新模型
        #self.model.set_target_positions(positions)

        # 构造位置数据
        data = bytearray()
        actual_positions = []

        for i, pos in enumerate(positions):
            # 限制位置范围 - 使用关节实际限位
            original_pos = pos
            joint = next((j for j in JOINT_DEFINITIONS if j.id == i+1), None)
            if joint:
                clamped_pos = max(joint.min_pos, min(joint.max_pos, pos))
            else:
                clamped_pos = max(-32768, min(32767, pos))  # 默认范围
            actual_positions.append(clamped_pos)

            if original_pos != clamped_pos:
                print(f"   ⚠️ 电机{i+1}: 位置被限制 {original_pos} → {clamped_pos}")
            # 确保clamped_pos是整数
            clamped_pos = int(round(clamped_pos))
            actual_positions.append(clamped_pos)
            # 转换为小端序字节
            pos_bytes = clamped_pos.to_bytes(2, byteorder='little', signed=True)
            data.extend(pos_bytes)

            # 详细打印每个关节的信息
            if i < len(JOINT_DEFINITIONS):
                joint_def = JOINT_DEFINITIONS[i]
                angle_deg = clamped_pos * POSITION_UNIT  # 转换为角度
                # print(f"   电机{joint_def.id:2d} ({joint_def.finger}-{joint_def.name}): "
                #       f"原始值={clamped_pos:6d}, 角度={angle_deg:7.2f}°, "
                #       f"字节=[{pos_bytes[0]:02X} {pos_bytes[1]:02X}]")

        #print(f"   实际发送位置: {actual_positions}")
        #print(f"   数据包大小: {len(data)}字节")
        #print(f"   原始数据: {data.hex().upper()}")

        # 发送位置命令
        # print("*" * 30, flush=True)
        # print(data.hex(' ').upper())
        success = self.comm.send_message(RegisterAddress.SYS_TARGET_POS.value, bytes(data), True)

        if success:
            #print(f"   ✅ 位置命令发送成功")
            pass
        else:
            #print(f"   ❌ 位置命令发送失败")
            pass

        return success

    def set_joint_velocities(self, velocities: List[int]) -> bool:
        """设置关节速度"""
        if len(velocities) != 17:
            print(f"❌ 速度数据长度错误: 期望17个，实际{len(velocities)}个")
            return False

        print(f"📤 设置关节速度命令:")
        print(f"   输入速度数组: {velocities}")

        # 构造速度数据
        data = bytearray()
        actual_velocities = []

        for i, vel in enumerate(velocities):
            original_vel = vel
            # 限制速度范围，且先四舍五入取整
            clamped_vel = max(0, min(65535, int(round(vel))))
            actual_velocities.append(clamped_vel)

            if original_vel != clamped_vel:
                print(f"   ⚠️ 电机{i+1}: 速度被限制 {original_vel} → {clamped_vel}")

            # 转换为小端序字节 (无符号)
            vel_bytes = clamped_vel.to_bytes(2, byteorder='little', signed=False)
            data.extend(vel_bytes)

            # 详细打印每个关节的信息
            if i < len(JOINT_DEFINITIONS):
                joint_def = JOINT_DEFINITIONS[i]
                rpm_value = clamped_vel * VELOCITY_UNIT  # 转换为RPM
                # print(f"   电机{joint_def.id:2d} ({joint_def.finger}-{joint_def.name}): "
                #       f"原始值={clamped_vel:6d}, RPM={rpm_value:7.2f}, "
                #       f"字节=[{vel_bytes[0]:02X} {vel_bytes[1]:02X}]")

        # print(f"   实际发送速度: {actual_velocities}")
        # print(f"   数据包大小: {len(data)}字节")
        # print(f"   原始数据: {data.hex().upper()}")

        # 发送速度命令
        success = self.comm.send_message(RegisterAddress.SYS_TARGET_VEL.value, bytes(data), True)

        if success:
            print(f"   ✅ 速度命令发送成功")
        else:
            print(f"   ❌ 速度命令发送失败")

        return success

    def set_default_velocity(self, default_vel: int = 100) -> bool:
        """设置默认速度（位置模式下需要设置一次）

        Args:
            default_vel: 默认速度值，单位为原始值（乘以0.732得到RPM）
        """
        print(f"🚀 设置默认速度: {default_vel} (约{default_vel * VELOCITY_UNIT:.1f} RPM)")

        # 为所有17个电机设置相同的默认速度
        default_velocities = [default_vel] * 17
        return self.set_joint_velocities(default_velocities)

    def set_joint_torques(self, torques: List[int]) -> bool:
        """设置关节力矩限制"""
        if len(torques) != 17:
            print(f"❌ 力矩数据长度错误: 期望17个，实际{len(torques)}个")
            return False

        print(f"📤 设置关节力矩命令:")
        print(f"   输入力矩数组: {torques}")

        # 构造力矩数据 - 使用uint16_t[17]，每2字节对应一个电机
        data = bytearray()
        actual_torques = []

        for i, torque in enumerate(torques):
            # 限制力矩范围 (无符号16位: 0-1000)
            original_torque = torque
            clamped_torque = max(0, min(1000, torque))
            actual_torques.append(clamped_torque)

            if original_torque != clamped_torque:
                print(f"   ⚠️ 电机{i+1}: 力矩被限制 {original_torque} → {clamped_torque}")

            # 转换为2字节小端序 (uint16_t)
            torque_bytes = clamped_torque.to_bytes(2, byteorder='little', signed=False)
            data.extend(torque_bytes)

            # 详细打印每个关节的信息
            if i < len(JOINT_DEFINITIONS):
                joint_def = JOINT_DEFINITIONS[i]
                # 力矩单位：6.5mA (根据协议规范v2.0) - 注意：单位可能需要重新确认
                current_ma = clamped_torque * 6.5
                print(f"   电机{joint_def.id:2d} ({joint_def.finger}-{joint_def.name}): "
                      f"原始值={clamped_torque:4d}, 电流={current_ma:6.1f}mA, "
                      f"字节=[{torque_bytes[0]:02X} {torque_bytes[1]:02X}]")

        print(f"   实际发送力矩: {actual_torques}")
        print(f"   数据包大小: {len(data)}字节 (协议要求34字节)")
        print(f"   原始数据: {data.hex().upper()}")

        # 发送力矩命令
        success = self.comm.send_message(RegisterAddress.SYS_TARGET_TORQUE.value, bytes(data), True)

        if success:
            print(f"   ✅ 力矩命令发送成功")
        else:
            print(f"   ❌ 力矩命令发送失败")

        return success

    def set_default_torque(self, default_torque: int = 500) -> bool:
        """设置默认力矩限制

        Args:
            default_torque: 默认力矩值，单位为原始值
        """
        print(f"💪 设置默认力矩: {default_torque}")

        # 为所有17个电机设置相同的默认力矩
        default_torques = [default_torque] * 17
        self.model.target_torques = default_torques
        return self.set_joint_torques(default_torques)




    def set_calibration_mode(self, mode: int) -> bool:
        """设置校准模式"""
        data = bytes([mode])
        return self.comm.send_message(RegisterAddress.SYS_CALI_MODE.value, data, True)

    def read_device_info(self) -> Optional[str]:
        """读取设备信息"""
        try:
            print("📋 开始读取设备信息...")

            # 多次尝试读取设备信息
            for attempt in range(5):  # 增加尝试次数
                print(f"   尝试 {attempt + 1}/5...")

                if self.comm.send_message(RegisterAddress.SYS_DEVICE_INFO.value, b'', False):
                    time.sleep(0.05)  # 增加等待时间到50ms

                    # 接收响应，不过滤设备ID
                    messages = self.comm.receive_messages(200, filter_device_id=False)

                    # 收集所有有效的设备信息响应
                    valid_responses = []
                    for frame_id, data in messages:
                        device_id = (frame_id >> 21) & 0xFF
                        register_addr = (frame_id >> 13) & 0xFF

                        # 检查是否是来自目标设备的设备信息响应
                        if (device_id == self.comm.device_id and
                            register_addr == RegisterAddress.SYS_DEVICE_INFO.value):

                            print(f"   收到设备信息响应: 长度={len(data)}")
                            valid_responses.append(data)

                    # 优先处理长度最长的数据（通常是完整的设备信息）
                    if valid_responses:
                        # 按数据长度降序排序，优先处理最长的数据
                        valid_responses.sort(key=len, reverse=True)
                        data = valid_responses[0]

                        print(f"   选择最完整的数据进行解析: 长度={len(data)}")

                        try:
                            if len(data) >= 50:
                                # 解析完整设备信息 - 按照协议规范v2.0
                                product_model = data[0:10].decode('utf-8', errors='ignore').strip('\x00')
                                serial_number = data[10:30].decode('utf-8', errors='ignore').strip('\x00')
                                software_version = data[30:40].decode('utf-8', errors='ignore').strip('\x00')
                                hardware_version = data[40:50].decode('utf-8', errors='ignore').strip('\x00')
                                # 手型标志位在第51字节（索引50）：1=右手，0=左手
                                hand_type = "右手" if len(data) > 50 and data[50] == 1 else "左手"

                                info = f"产品型号: {product_model}\n"
                                info += f"序列号: {serial_number}\n"
                                info += f"软件版本: {software_version}\n"
                                info += f"硬件版本: {hardware_version}\n"
                                info += f"手型: {hand_type}"

                                print(f"   ✅ 完整设备信息解析成功")
                                return info

                            elif len(data) >= 10:
                                # 部分解析
                                print(f"   ⚠️ 数据长度不足({len(data)}字节)，尝试部分解析")
                                product_model = data[0:10].decode('utf-8', errors='ignore').strip('\x00')
                                info = f"产品型号: {product_model}\n"
                                info += f"数据长度: {len(data)}字节\n"
                                info += f"原始数据: {data.hex().upper()}"
                                return info
                            else:
                                # 数据太短，可能是测试数据，显示原始数据
                                print(f"   ⚠️ 收到测试数据或不完整数据({len(data)}字节)")
                                info = f"设备响应正常\n"
                                info += f"数据长度: {len(data)}字节\n"
                                info += f"原始数据: {data.hex().upper()}\n"
                                info += f"注意: 这可能是测试数据，请等待完整设备信息"
                                return info

                        except Exception as e:
                            print(f"   ❌ 解析设备信息失败: {e}")
                            # 即使解析失败，也返回原始数据
                            info = f"设备响应正常(解析失败)\n"
                            info += f"数据长度: {len(data)}字节\n"
                            info += f"原始数据: {data.hex().upper()}\n"
                            info += f"错误: {e}"
                            return info
                else:
                    print(f"   ❌ 发送查询命令失败")

                time.sleep(0.05)  # 重试间隔

            print("   ❌ 多次尝试后仍无法读取设备信息")
            return None

        except Exception as e:
            print(f"❌ 读取设备信息异常: {e}")
            return None

    def emergency_stop(self):
        """紧急停止"""
        # 发送所有关节位置为当前位置
        current_positions = self.model.get_all_current_positions()
        self.set_joint_positions(current_positions)

    def reset_to_zero(self):
        """复位到零位"""
        zero_positions = [0] * 17
        self.set_joint_positions(zero_positions)


    def denormalize_motor_values(self, norm_values: list[int]) -> list[float]:
        """
        将0~255的归一化值列表按 joint_limits 映射回实际角度值（单位：度），并保留两位小数
        :param norm_values: 长度为 N 的列表，每个元素是 0~255 的整数
        :return: 角度值（float）列表，顺序对应电机ID 1~N
        """
        angles = []
        for idx, val in enumerate(norm_values):
            motor_id = idx + 1  # 列表索引从0开始，电机ID从1开始
            if motor_id in self.joint_limits:
                min_val, max_val = self.joint_limits[motor_id]
                val = max(0, min(255, val))  # 防止越界
                angle = min_val + (val / 255.0) * (max_val - min_val)
                angles.append(round(angle, 2))
            else:
                print(f"[警告] motor_id {motor_id} 没有定义 joint_limits，跳过")
                angles.append(0.0)  # 或者可以添加None或其他默认值
        return angles


    
    def get_device_info(self):
        """读取设备信息"""
        product_model, serial_number, software_version, hardware_version, hand_type = None,None,None,None,None
        try:
            if self.comm.send_message(RegisterAddress.SYS_DEVICE_INFO.value, b'', False):
                time.sleep(0.05)  # 增加等待时间到50ms

                # 接收响应，不过滤设备ID
                messages = self.comm.receive_messages(200, filter_device_id=False)

                # 收集所有有效的设备信息响应
                valid_responses = []
                for frame_id, data in messages:
                    device_id = (frame_id >> 21) & 0xFF
                    register_addr = (frame_id >> 13) & 0xFF

                    # 检查是否是来自目标设备的设备信息响应
                    if (device_id == self.comm.device_id and
                        register_addr == RegisterAddress.SYS_DEVICE_INFO.value):

                        print(f"   收到设备信息响应: 长度={len(data)}")
                        valid_responses.append(data)

                # 优先处理长度最长的数据（通常是完整的设备信息）
                if valid_responses:
                    # 按数据长度降序排序，优先处理最长的数据
                    valid_responses.sort(key=len, reverse=True)
                    data = valid_responses[0]
                    try:
                        if len(data) >= 50:
                            # 解析完整设备信息 - 按照协议规范v2.0
                            product_model = data[0:10].decode('utf-8', errors='ignore').strip('\x00')
                            serial_number = data[10:30].decode('utf-8', errors='ignore').strip('\x00')
                            software_version = data[30:40].decode('utf-8', errors='ignore').strip('\x00')
                            hardware_version = data[40:50].decode('utf-8', errors='ignore').strip('\x00')
                            # 手型标志位在第51字节（索引50）：1=右手，0=左手
                            hand_type = "右手" if len(data) > 50 and data[50] == 1 else "左手"

                            info = f"产品型号: {product_model}\n"
                            info += f"序列号: {serial_number}\n"
                            info += f"软件版本: {software_version}\n"
                            info += f"硬件版本: {hardware_version}\n"
                            info += f"手型: {hand_type}"
                    except Exception as e:
                        print(f"   ❌ 解析设备信息失败: {e}")
                        # 即使解析失败，也返回原始数据
                        info = f"设备响应正常(解析失败)\n"
                        info += f"数据长度: {len(data)}字节\n"
                        info += f"原始数据: {data.hex().upper()}\n"
                        info += f"错误: {e}"
                
            else:
                print(f"   ❌ 发送查询命令失败")


        except Exception as e:
            print(f"❌ 读取设备信息异常: {e}")
            return None
        return product_model, serial_number, software_version, hardware_version, hand_type
    

    def get_current_speed(self):
        """读取当前速度"""
        try:
            if self.comm.send_message(0x04, b'', False):
                time.sleep(0.005)  # 5ms等待
                messages = self.comm.receive_messages(50)
                for frame_id, data in messages:
                    if self._is_velocity_response(frame_id):
                        if len(data) >= 34:  # 17个关节 * 2字节
                            velocities = self._parse_velocity_data(data)
                            
        except Exception as e:
            print(f"读取速度失败: {e}")
        return velocities

    def get_error_status(self):
        """读取错误状态"""
        try:
            if self.comm.send_message(RegisterAddress.SYS_ERROR_STATUS.value, b'', False):
                time.sleep(0.005)  # 5ms等待

                messages = self.comm.receive_messages(50)
                for frame_id, data in messages:
                    if self._is_error_response(frame_id):
                        if len(data) >= 17:  # 17个关节状态
                            errors = list(data[:17])
                            #self.model.update_error_status(errors)
                            return errors
        except Exception as e:
            print(f"读取错误状态失败: {e}")


    def get_current_state(self):
        """读取当前位置"""
        try:
            # 发送读取位置命令
            if self.comm.send_message(RegisterAddress.SYS_CURRENT_POS.value, b'', False):
                # 等待响应
                time.sleep(0.005)  # 5ms等待

                # 接收响应
                messages = self.comm.receive_messages(50)
                for frame_id, data in messages:
                    if self._is_position_response(frame_id):
                        if len(data) >= 34:  # 17个关节 * 2字节
                            
                            state_arc = self._parse_position_data(data)
                            if len(state_arc) == 17:
                                #self.model.update_joint_positions(positions)
                                state_range = self.normalize_raw_motor_values(state_arc)
                                return state_arc, state_range
                            else:
                                print(f"警告: 位置数据长度不正确: {len(state_arc)}")
                        else:
                            print(f"警告: 位置数据长度不足: {len(data)}")
        except Exception as e:
            print(f"读取位置失败: {e}")


    def normalize_raw_motor_values(self, raw_values: list[float]) -> list[int]:
        """
        将获取到的原始电机角度值（长度17）映射为0~255范围的列表
        :param raw_values: 长度为17的电机角度值列表
        :return: 长度为17的归一化值列表（每个在 0~255 之间）
        """
        assert len(raw_values) == 17, "必须提供17个电机的原始值"
        result = []
        for i, val in enumerate(raw_values):
            motor_id = i + 1
            if motor_id in self.joint_limits:
                min_val, max_val = self.joint_limits[motor_id]
                if max_val == min_val:
                    norm = 0
                else:
                    norm = round((val - min_val) / (max_val - min_val) * 255)
                    norm = max(0, min(255, norm))  # 限制到 [0,255]
            else:
                norm = 0  # 未知电机ID，默认为0
            result.append(norm)
        return result
    # ------------------------------------------------------------------------------------------
    # def _receive_frame(self, timeout_ms=100) -> Optional[Tuple[int, bytes]]:
    #     msg_array = CanFD_Msg_ARRAY(500)
    #     ret = self.comm.canDLL.CANFD_Receive(0, 0, byref(msg_array.ADDR), 500, timeout_ms)
    #     if ret <= 0:
    #         return None
    #     for i in range(ret):
    #         msg = msg_array.STRUCT_ARRAY[i]
    #         if ((msg.ID >> self.DEVICE_ID_SHIFT) & 0xFF) == 0x01:
    #             # 过滤本设备ID消息
    #             data_len = self.dlc2len[msg.DLC] if msg.DLC < len(self.dlc2len) else 0
    #             data_bytes = bytes(msg.Data[:data_len])
    #             return msg.ID, data_bytes
    #     return None
    # def _get_dlc_from_length(self, length: int) -> int:
    #     """根据数据长度获取DLC值"""
    #     if length <= 8:
    #         return length
    #     elif length <= 12:
    #         return 9
    #     elif length <= 16:
    #         return 10
    #     elif length <= 20:
    #         return 11
    #     elif length <= 24:
    #         return 12
    #     elif length <= 32:
    #         return 13
    #     elif length <= 48:
    #         return 14
    #     else:
    #         return 15  # 64字节
    # def _build_can_id(self, reg_addr: int, rw_flag: int) -> int:
    #     device_part = (0x01 & 0xFF) << self.DEVICE_ID_SHIFT
    #     reg_part = (reg_addr & 0xFF) << self.REG_ADDR_SHIFT
    #     rw_part = (rw_flag & 0x1) << self.RW_FLAG_SHIFT
    #     return device_part | reg_part | rw_part
    # def _send_frame(self, reg_addr: int, rw_flag: int, data: bytes) -> bool:
    #     can_id = self._build_can_id(reg_addr, rw_flag)
    #     # 限制数据长度
    #     data_len = min(len(data), 64)

    #     # 创建数据数组并初始化为0
    #     data_array = (c_ubyte * 64)()
    #     for i in range(64):
    #         data_array[i] = 0

    #     # 填充实际数据
    #     for i, byte_val in enumerate(data[:data_len]):
    #         data_array[i] = byte_val
    #     # 计算DLC (Data Length Code)
    #     dlc = self._get_dlc_from_length(data_len)
    #     msg = CanFD_Msg(
    #         ID=can_id,
    #         TimeStamp=0,
    #         FrameType=0x04,   # CANFD 帧
    #         DLC=dlc,
    #         ExternFlag=1,     # 扩展帧
    #         RemoteFlag=0,
    #         BusSatus=0,
    #         ErrSatus=0,
    #         TECounter=0,
    #         RECounter=0,
    #         Data=data_array
    #     )
    #     # print("_-" * 30, flush=True)
    #     # print(data_len, flush=True)
    #     # print(' '.join(f'{b:02X}' for b in data), flush=True)
    #     # print("*" * 30, flush=True)
    #     ret = self.comm.canDLL.CANFD_Transmit(0, 0, byref(msg), 1, 100)
    #     if ret != 1:
    #         print(f"发送CANFD帧失败，寄存器0x{reg_addr:02X} 返回值: {ret}")
    #         return False
        
    #     return True
    # def read_register(self, reg_addr: int, length: int) -> Optional[bytes]:
    #     if not self._send_frame(reg_addr, self.RW_READ, b''):
    #         print(f"发送读寄存器0x{reg_addr:02X}命令失败")
    #         return None
    #     start_time = time.time()
    #     while time.time() - start_time < 0.3:
    #         res = self._receive_frame(timeout_ms=50)
    #         if res is None:
    #             continue
    #         can_id, data = res
    #         # 检查回复ID是否匹配
    #         if (((can_id >> self.DEVICE_ID_SHIFT) & 0xFF) == self.device_id and
    #             ((can_id >> self.REG_ADDR_SHIFT) & 0xFF) == reg_addr and
    #             ((can_id >> self.RW_FLAG_SHIFT) & 0x1) == self.RW_READ):
    #             if len(data) >= length:
    #                 return data[:length]
    #     print(f"读寄存器0x{reg_addr:02X}超时")
    #     return None

    # def get_current_positions(self) -> Optional[Dict[int, float]]:
    #     data = self.read_register(self.REG_SYS_CURRENT_POS, 34)
    #     if data is None or len(data) != 34:
    #         return None
    #     #pos = {}
    #     # pose = [0] * 17
    #     # for i in range(17):
    #     #     val = int.from_bytes(data[i * 2:i * 2 + 2], byteorder='little', signed=True)
    #     #     pose[i] = val
    #         #pos[i + 1] = val * 0.087  # 单位：度
    #     pose = self.hex_to_dec(data)
    #     print(pose)
    #     return pose
    
    # def hex_to_dec(self, byte_data):
    #     """
    #     将字节数据转换为有符号整数列表(小端序)
    #     参数:
    #         byte_data: bytes对象 如 b'\xC7\x00\x6F\x01...'
    #     返回:
    #         有符号整数列表
    #     """
    #     print(byte_data)
    #     if not isinstance(byte_data, bytes):
    #         raise TypeError(f"Expected bytes, got {type(byte_data)}")
        
    #     if len(byte_data) % 2 != 0:
    #         raise ValueError(f"Byte data length must be even, got {len(byte_data)}")
        
    #     results = []
    #     for i in range(0, len(byte_data), 2):
    #         # 提取两个字节并转换为有符号short(小端序)
    #         value = struct.unpack('<h', byte_data[i:i+2])[0]
    #         results.append(value)
        
    #     return results
    
# 结构体数组辅助类，用于接收多个CANFD消息
class CanFD_Msg_ARRAY(Structure):
    _fields_ = [('SIZE', c_uint16), ('STRUCT_ARRAY', POINTER(CanFD_Msg))]

    def __init__(self, num_of_structs):
        self.STRUCT_ARRAY = cast((CanFD_Msg * num_of_structs)(), POINTER(CanFD_Msg))
        self.SIZE = num_of_structs
        self.ADDR = self.STRUCT_ARRAY[0]

