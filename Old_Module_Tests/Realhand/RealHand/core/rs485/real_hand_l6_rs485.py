#!/usr/bin/env python3
import os
import time
from pymodbus.client import ModbusSerialClient
from typing import List, Dict
import numpy as np

_INTERVAL = 0.006  # 8 ms

class RealHandL6RS485:
    """L6 robotic hand Modbus-RTU control class"""
    
    # 6 joint names
    JOINT_NAMES = ["thumb_pitch", "thumb_yaw", "index_pitch", 
                   "middle_pitch", "ring_pitch", "little_pitch"]
    
    # Finger names
    FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]
    
    def __init__(self, hand_id=0x27, modbus_port="/dev/ttyUSB0", baudrate=115200):
        """
        Initialize L6 robotic hand
        hand_id: right hand 0x27(39), left hand 0x28(40)
        modbus_port: serial device path
        baudrate: baud rate, fixed at 115200
        """
        self.slave = hand_id
        self.cli = ModbusSerialClient(
            port=modbus_port, 
            baudrate=baudrate,
            bytesize=8, 
            parity="N", 
            stopbits=1,
            timeout=0.05
        )
        # pymodbus 3.5.1 requires explicit connect
        self.connected = self.cli.connect()
        if not self.connected:
            raise ConnectionError(f"RS485 connection failed, port: {modbus_port}")

    def _read_input_registers(self, address: int, count: int) -> List[int]:
        """Read input registers"""
        time.sleep(_INTERVAL)
        result = self.cli.read_input_registers(address=address, count=count, slave=self.slave)
        if result.isError():
            raise RuntimeError(f"Failed to read input registers: address={address}, count={count}")
        return result.registers

    def _write_register(self, address: int, value: int):
        """Write single register"""
        time.sleep(_INTERVAL)
        result = self.cli.write_register(address=address, value=value, slave=self.slave)
        if result.isError():
            raise RuntimeError(f"Failed to write register: address={address}, value={value}")

    def _write_registers(self, address: int, values: List[int]):
        """Write multiple registers"""
        time.sleep(_INTERVAL)
        result = self.cli.write_registers(address=address, values=values, slave=self.slave)
        if result.isError():
            raise RuntimeError(f"Failed to write multiple registers: address={address}, values={values}")

    # --------------------------------------------------
    # Basic read interfaces
    # --------------------------------------------------
    
    def read_angles(self) -> List[int]:
        """Read 6 joint angles (input registers 0-5)"""
        return self._read_input_registers(0, 6)

    def read_torques(self) -> List[int]:
        """Read 6 joint torques (input registers 6-11)"""
        return self._read_input_registers(6, 6)

    def read_speeds(self) -> List[int]:
        """Read 6 joint speeds (input registers 12-17)"""
        return self._read_input_registers(12, 6)

    def read_temperatures(self) -> List[int]:
        """Read 6 joint temperatures (input registers 18-23)"""
        return self._read_input_registers(18, 6)

    def read_error_codes(self) -> List[int]:
        """Read 6 joint error codes (input registers 24-29)"""
        return self._read_input_registers(24, 6)

    # --------------------------------------------------
    # Pressure sensor interfaces
    # --------------------------------------------------
    
    # def _pressure(self, finger: int) -> List[int]:
    #     """Internal: select finger → read pressure data"""
    #     # Select finger (holding register 36)
    #     self._write_register(36, finger)
    #     time.sleep(_INTERVAL)
    #     # Read pressure data (input registers 52-122)
    #     return np.array(self._read_input_registers(52, 71))

    def _pressure(self, finger: int) -> np.ndarray:
        """
        6x12 (72-point) matrix size.
        Modbus addresses 60/62.
        """
        rows = 12  # 12 rows
        cols = 6   # 6 columns
        finger_size = rows * cols  # 72 data points
        
        # Modbus addresses and counts
        write_address = 60  # write finger selection
        read_address = 62   # read pressure data
        read_count = 96     # read 96 registers
        skip_count = 10     # skip first 10 check points
        
        # 0. Parameter validation and finger write value
        if finger < 1 or finger > 5:
            raise ValueError(f"Invalid finger index: {finger}. Finger index should be between 1 and 5.")
            
        finger_write_value = finger 
        
        # 1. Write finger selection register (address 60)
        time.sleep(0.008)
        wrsp = self.cli.write_register(address=write_address, value=finger_write_value, slave=self.slave)
        if wrsp.isError():
            raise RuntimeError(f"Failed to write finger selection {finger} to address {write_address}: {wrsp}")

        # Wait a moment after writing
        time.sleep(0.008) 
        
        # 2. Read data from address 62
        rrsp = self.cli.read_input_registers(address=read_address, count=read_count, slave=self.slave)
        
        if rrsp.isError():
            raise RuntimeError(f"Failed to read pressure data from address {read_address}: {rrsp}")
            
        registers_16bit: List[int] = rrsp.registers 
        
        # 3. Core data processing
        # a. Extract lower 8 bits (get 96 8-bit data points)
        final_data_96 = [reg_value & 255 for reg_value in registers_16bit]
        
        # b. Skip first 10 check/header data points (get 86 valid data points)
        effective_data = np.array(final_data_96[skip_count:], dtype=np.uint8)
        # c. Slice current finger matrix data (take 72 points from 86 valid points)
        start_idx = 0 
        end_idx = finger_size  # 72
        
        finger_data_flat = effective_data[start_idx:end_idx]
        
        # d. Validate data length
        if finger_data_flat.size != finger_size:
            raise ValueError(
                f"Data extraction failed. Expected {finger_size} points ({rows}x{cols}), "
                f"but only sliced {finger_data_flat.size} points. Please check the protocol and "
                f"confirm whether address 62 returns all finger data at once."
            )
            
        # e. Reshape to 2D matrix (12 rows, 6 columns)
        finger_matrix = finger_data_flat.reshape((rows, cols))
                
        return finger_matrix

    def read_pressure_thumb(self) -> np.ndarray:
        """Read thumb pressure data"""
        return np.array(self._pressure(1), dtype=np.uint8)

    def read_pressure_index(self) -> np.ndarray:
        """Read index finger pressure data"""
        return np.array(self._pressure(2), dtype=np.uint8)

    def read_pressure_middle(self) -> np.ndarray:
        """Read middle finger pressure data"""
        return np.array(self._pressure(3), dtype=np.uint8)

    def read_pressure_ring(self) -> np.ndarray:
        """Read ring finger pressure data"""
        return np.array(self._pressure(4), dtype=np.uint8)

    def read_pressure_little(self) -> np.ndarray:
        """Read little finger pressure data"""
        return np.array(self._pressure(5), dtype=np.uint8)

    # --------------------------------------------------
    # Version info interfaces
    # --------------------------------------------------
    
    def read_versions(self) -> Dict[str, int]:
        """Read version info (input registers 148-155)"""
        result = self._read_input_registers(148, 8)
        
        return {
            "hand_freedom": result[0],
            "hand_version": result[1],
            "hand_number": result[2],
            "hand_direction": result[3],
            "software_version_major": result[4],
            "software_version_minor": result[5] if len(result) > 5 else 0,
            "software_version_revision": result[6] if len(result) > 6 else 0,
            "hardware_version": result[7] if len(result) > 7 else 0
        }

    # --------------------------------------------------
    # Write interfaces
    # --------------------------------------------------
    
    def write_angles(self, vals: List[int]):
        """Set 6 joint angles (holding registers 0-5)"""
        vals = [int(x) for x in vals]
        if not self.is_valid_6xuint8(vals):
            raise ValueError("Requires six integers between 0 and 255")
        self._write_registers(0, vals)

    def write_torques(self, vals: List[int]):
        """Set 6 joint torques (holding registers 6-11)"""
        vals = [int(x) for x in vals]
        if not self.is_valid_6xuint8(vals):
            raise ValueError("Requires six integers between 0 and 255")
        self._write_registers(6, vals)

    def write_speeds(self, vals: List[int]):
        """Set 6 joint speeds (holding registers 12-17)"""
        vals = [int(x) for x in vals]
        if not self.is_valid_6xuint8(vals):
            raise ValueError("Requires six integers between 0 and 255")
        self._write_registers(12, vals)

    # --------------------------------------------------
    # Context management
    # --------------------------------------------------
    
    def close(self):
        """Close connection"""
        if self.connected:
            self.cli.close()
            self.connected = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # --------------------------------------------------
    # Fixed API interface functions
    # --------------------------------------------------
    
    def is_valid_6xuint8(self, lst) -> bool:
        """Validate list of six integers between 0 and 255"""
        if len(lst) != 6:
            return False
        return all(isinstance(x, int) and 0 <= x <= 255 for x in lst)
    
    def set_joint_positions(self, joint_angles=None):
        """Set joint positions"""
        joint_angles = joint_angles or [0] * 6
        self.write_angles(joint_angles)

    def set_speed(self, speed=None):
        """Set speed"""
        speed = speed or [200] * 6
        self.write_speeds(speed)
    
    def set_torque(self, torque=None):
        """Set torque"""
        torque = torque or [200] * 6
        self.write_torques(torque)

    def set_current(self, current=None):
        """Set current (L6 not supported)"""
        print("Current L6 does not support setting current", flush=True)

    def get_version(self) -> list:
        """Get version info"""
        versions = self.read_versions()
        return [
            versions.get("hand_freedom", 0),
            versions.get("hand_version", 0),
            versions.get("hand_number", 0),
            versions.get("hand_direction", 0),
            versions.get("software_version_major", 0),
            versions.get("hardware_version", 0)
        ]

    def get_current(self):
        """Get current (L6 not supported)"""
        print("Current L6 does not support getting current", flush=True)
        return []

    def get_state(self) -> list:
        """Get joint state"""
        return self.read_angles()
    
    def get_state_for_pub(self) -> list:
        return self.get_state()

    def get_current_status(self) -> list:
        return self.get_state()
    
    def get_speed(self) -> list:
        """Get current speeds"""
        return self.read_speeds()
    
    def get_joint_speed(self) -> list:
        return self.get_speed()
    
    def get_touch_type(self) -> int:
        """Get tactile type (2=matrix)"""
        return 2
    
    def get_normal_force(self) -> list:
        """Get tactile data: point type"""
        return [-1] * 5
    
    def get_tangential_force(self) -> list:
        """Get tactile data: point type"""
        return [-1] * 5
    
    def get_approach_inc(self) -> list:
        """Get tactile data: point type"""
        return [-1] * 5
    
    def get_touch(self) -> list:
        return [-1] * 5

    def get_thumb_matrix_touch(self,sleep_time=0):
        return self._pressure(1)

    def get_index_matrix_touch(self,sleep_time=0):
        return self._pressure(2)

    def get_middle_matrix_touch(self,sleep_time=0):
        return self._pressure(3)

    def get_ring_matrix_touch(self,sleep_time=0):
        return self._pressure(4)

    def get_little_matrix_touch(self,sleep_time=0):
        return self._pressure(5)
        
    def get_matrix_touch(self) -> list:
        """Get tactile data: matrix type"""
        return [self._pressure(1), self._pressure(2), self._pressure(3), 
                self._pressure(4), self._pressure(5)]
    
    def get_matrix_touch_v2(self) -> list:
        """Get tactile data: matrix type"""
        return self.get_matrix_touch()
    
    def get_torque(self) -> list:
        """Get current torques"""
        return self.read_torques()
    
    def get_temperature(self) -> list:
        """Get current motor temperatures"""
        return self.read_temperatures()
    
    def get_fault(self) -> list:
        """Get current motor fault codes"""
        return self.read_error_codes()

    # --------------------------------------------------
    # Convenience methods
    # --------------------------------------------------
    
    def relax(self):
        """Extend all fingers"""
        self.set_joint_positions([255] * 6)
    
    def fist(self):
        """Make a fist with all fingers"""
        self.set_joint_positions([0] * 6)
    
    def dump_status(self):
        """Print status information"""
        print("=" * 50)
        print("L6 robotic hand status")
        print("=" * 50)
        
        try:
            # Joint state
            angles = self.read_angles()
            torques = self.read_torques()
            speeds = self.read_speeds()
            temps = self.read_temperatures()
            errors = self.read_error_codes()
            
            print("Joint state:")
            for i, name in enumerate(self.JOINT_NAMES):
                print(f"  {name:15s}: angle={angles[i]:3d}, torque={torques[i]:3d}, "
                      f"speed={speeds[i]:3d}, temp={temps[i]:2d}℃, error={errors[i]:2d}")
            
            # Version info
            versions = self.read_versions()
            print("\nVersion info:")
            for key, value in versions.items():
                print(f"  {key:20s}: {value}")
            
            # Pressure sensor test
            print("\nPressure sensor test:")
            thumb_pressure = self.read_pressure_thumb()
            print(f"Thumb pressure data length: {len(thumb_pressure)}")
            
        except Exception as e:
            print(f"Error while reading status: {e}")
        
        print("=" * 50)


# ------------------- Demo program -------------------
if __name__ == "__main__":
    # Usage example
    try:
        with RealHandL6RS485(hand_id=0x27, modbus_port="/dev/ttyUSB0", baudrate=115200) as hand:
            print("Connected successfully!")
            
            # Print status info
            hand.dump_status()
            
            # Test basic control
            print("\nTesting control functions...")
            print("Extending fingers...")
            hand.relax()
            time.sleep(2)
            
            print("Making a fist...")
            hand.fist()
            time.sleep(2)
            
            print("Returning to extended...")
            hand.relax()
            
            # Test pressure sensor
            print("\nTesting pressure sensor...")
            thumb_matrix = hand.get_thumb_matrix_touch()
            print(f"Thumb pressure data: {len(thumb_matrix)} points")
            
            # Get pressure data for all fingers
            all_matrices = hand.get_matrix_touch()
            for i, name in enumerate(hand.FINGER_NAMES):
                matrix = all_matrices[i]
                print(f"{name} finger pressure data length: {len(matrix)}")
            
    except Exception as e:
        print(f"Error: {e}")
