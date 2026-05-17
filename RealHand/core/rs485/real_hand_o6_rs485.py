#!/usr/bin/env python3
"""
O6 robotic hand Modbus-RTU control class (based on pymodbus 3.5.1)
"""

import os
import time
from typing import List, Dict, Any # introduce Any to represent flexible input types
import numpy as np
import logging
from threading import Lock # used for thread safety and bus arbitration

# Import pymodbus client
from pymodbus.client import ModbusSerialClient
from struct import error as StructError 

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S"
)

# ------------------------------------------------------------------
# Input register address enumeration (function code 04, read-only) - keep unchanged
# ------------------------------------------------------------------
REG_RD_CURRENT_THUMB_PITCH      = 0   # Thumb flexion angle (0-255, small=bent, large=extended)
REG_RD_CURRENT_THUMB_YAW        = 1   # Thumb yaw angle (0-255, small=close to palm, large=away)
REG_RD_CURRENT_INDEX_PITCH      = 2   # Index finger flexion angle
REG_RD_CURRENT_MIDDLE_PITCH     = 3   # Middle finger flexion angle
REG_RD_CURRENT_RING_PITCH       = 4   # Ring finger flexion angle
REG_RD_CURRENT_LITTLE_PITCH     = 5   # Little finger flexion angle
REG_RD_CURRENT_THUMB_TORQUE     = 6   # Thumb flexion torque (0-255)
REG_RD_CURRENT_THUMB_YAW_TORQUE = 7   # Thumb yaw torque
REG_RD_CURRENT_INDEX_TORQUE     = 8   # Index finger torque
REG_RD_CURRENT_MIDDLE_TORQUE    = 9   # Middle finger torque
REG_RD_CURRENT_RING_TORQUE      = 10  # Ring finger torque
REG_RD_CURRENT_LITTLE_TORQUE    = 11  # Little finger torque
REG_RD_CURRENT_THUMB_SPEED      = 12  # Thumb flexion speed (0-255)
REG_RD_CURRENT_THUMB_YAW_SPEED  = 13  # Thumb yaw speed
REG_RD_CURRENT_INDEX_SPEED      = 14  # Index finger speed
REG_RD_CURRENT_MIDDLE_SPEED     = 15  # Middle finger speed
REG_RD_CURRENT_RING_SPEED       = 16  # Ring finger speed
REG_RD_CURRENT_LITTLE_SPEED     = 17  # Little finger speed
REG_RD_THUMB_TEMP               = 18  # Thumb flexion temperature (0-70℃)
REG_RD_THUMB_YAW_TEMP           = 19  # Thumb yaw temperature
REG_RD_INDEX_TEMP               = 20  # Index finger temperature
REG_RD_MIDDLE_TEMP              = 21  # Middle finger temperature
REG_RD_RING_TEMP                = 22  # Ring finger temperature
REG_RD_LITTLE_TEMP              = 23  # Little finger temperature
REG_RD_THUMB_ERROR              = 24  # Thumb error code
REG_RD_THUMB_YAW_ERROR          = 25  # Thumb yaw error code
REG_RD_INDEX_ERROR              = 26  # Index finger error code
REG_RD_MIDDLE_ERROR             = 27  # Middle finger error code
REG_RD_RING_ERROR               = 28  # Ring finger error code
REG_RD_LITTLE_ERROR             = 29  # Little finger error code
REG_RD_HAND_FREEDOM             = 30  # Degrees of freedom (same as hand label)
REG_RD_HAND_VERSION             = 31  # Hand version
REG_RD_HAND_NUMBER              = 32  # Hand number
REG_RD_HAND_DIRECTION           = 33  # Hand direction (left/right)
REG_RD_SOFTWARE_VERSION         = 34  # Software version
REG_RD_HARDWARE_VERSION         = 35  # Hardware version


# ------------------------------------------------------------------
# Write holding register address enumeration (function code 16, read/write) - keep unchanged
# ------------------------------------------------------------------
REG_WR_THUMB_PITCH      = 0   # Thumb flexion angle (0-255)
REG_WR_THUMB_YAW        = 1   # Thumb yaw angle
REG_WR_INDEX_PITCH      = 2   # Index finger flexion angle
REG_WR_MIDDLE_PITCH     = 3   # Middle finger flexion angle
REG_WR_RING_PITCH       = 4   # Ring finger flexion angle
REG_WR_LITTLE_PITCH     = 5   # Little finger flexion angle
REG_WR_THUMB_TORQUE     = 6   # Thumb flexion torque
REG_WR_THUMB_YAW_TORQUE = 7   # Thumb yaw torque
REG_WR_INDEX_TORQUE     = 8   # Index finger torque
REG_WR_MIDDLE_TORQUE    = 9   # Middle finger torque
REG_WR_RING_TORQUE      = 10  # Ring finger torque
REG_WR_LITTLE_TORQUE    = 11  # Little finger torque
REG_WR_THUMB_SPEED      = 12  # Thumb flexion speed
REG_WR_THUMB_YAW_SPEED  = 13  # Thumb yaw speed
REG_WR_INDEX_SPEED      = 14  # Index finger speed
REG_WR_MIDDLE_SPEED     = 15  # Middle finger speed
REG_WR_RING_SPEED       = 16  # Ring finger speed
REG_WR_LITTLE_SPEED     = 17  # Little finger speed


class RealHandO6RS485:
    """O6 robotic hand Modbus-RTU control class, using pymodbus 3.5.1"""

    TTL_TIMEOUT = 0.15     # Serial timeout
    FRAME_GAP = 0.030      # 30 ms
    
    # KEYS for easy indexing
    JOINT_KEYS = ["thumb_pitch", "thumb_yaw", "index_pitch", 
                  "middle_pitch", "ring_pitch", "little_pitch"]

    def __init__(self, hand_id=0x27, modbus_port="/dev/ttyUSB0", baudrate=115200):
        self._id = hand_id
        self._last_ts = 0.0  # Last frame end time
        self._lock = Lock()  # Bus access lock

        # Use pymodbus 3.x client
        self.cli = ModbusSerialClient(
            port=modbus_port,
            baudrate=baudrate,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=self.TTL_TIMEOUT,
            handle_local_echo=False
        )

        try:
            logging.info(f"Connecting to Modbus RTU on {modbus_port}...")
            self.connected = self.cli.connect()
            if not self.connected:
                raise ConnectionError(f"RS485 connect fail to {modbus_port}")
            logging.info("Connection successful.")
        except Exception as e:
            logging.error(f"Initialization failed: {e}")
            raise

    # ----------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------
    def _bus_free(self):
        """Ensure interval from last frame ≥ 30 ms"""
        with self._lock:
            elapse = time.perf_counter() - self._last_ts
            if elapse < self.FRAME_GAP:
                time.sleep(self.FRAME_GAP - elapse)

    def _execute_read(self, address: int, count: int) -> List[int]:
        """Execute Modbus read operation (function code 04) with bus arbitration."""
        self._bus_free()
        
        rsp = self.cli.read_input_registers(
            address=address, 
            count=count, 
            slave=self._id
        )
        
        self._last_ts = time.perf_counter()
        
        if rsp.isError():
            raise RuntimeError(f"Modbus Read Failed (Addr={address}, Count={count}): {rsp}")
        
        # Ensure the returned values are native Python integers
        return [int(reg) for reg in rsp.registers]

    def _execute_write(self, address: int, values: List[int]):
        """Execute Modbus batch write operation (function code 16) with bus arbitration."""
        self._bus_free()
        
        # values must be a list of native Python integers
        rsp = self.cli.write_registers(
            address=address, 
            values=values, 
            slave=self._id
        )
        
        self._last_ts = time.perf_counter()
        
        if rsp.isError():
            raise RuntimeError(f"Modbus Write Failed (Addr={address}, Values={values}): {rsp}")

    # ----------------------------------------------------------
    # Batch reading and data packaging (optimize communication efficiency)
    # ----------------------------------------------------------
    def read_all_angles(self) -> List[int]:
        return self._execute_read(REG_RD_CURRENT_THUMB_PITCH, 6)
    
    def read_all_torques(self) -> List[int]:
        return self._execute_read(REG_RD_CURRENT_THUMB_TORQUE, 6)

    def read_all_speeds(self) -> List[int]:
        return self._execute_read(REG_RD_CURRENT_THUMB_SPEED, 6)
    
    def read_all_temperatures(self) -> List[int]:
        return self._execute_read(REG_RD_THUMB_TEMP, 6)
    
    def read_all_errors(self) -> List[int]:
        return self._execute_read(REG_RD_THUMB_ERROR, 6)

    def read_all_versions(self) -> List[int]:
        return self._execute_read(REG_RD_HAND_FREEDOM, 6)


    # ----------------------------------------------------------
    # Read-only properties (single register read)
    # ----------------------------------------------------------
    def _read_reg(self, addr: int) -> int:
        """Read single input register (function code 04) with 30 ms frame interval"""
        return self._execute_read(addr, 1)[0]
    
    def get_thumb_pitch(self) -> int:       return self._read_reg(REG_RD_CURRENT_THUMB_PITCH)
    def get_thumb_yaw(self) -> int:         return self._read_reg(REG_RD_CURRENT_THUMB_YAW)
    def get_index_pitch(self) -> int:       return self._read_reg(REG_RD_CURRENT_INDEX_PITCH)
    def get_middle_pitch(self) -> int:      return self._read_reg(REG_RD_CURRENT_MIDDLE_PITCH)
    def get_ring_pitch(self) -> int:        return self._read_reg(REG_RD_CURRENT_RING_PITCH)
    def get_little_pitch(self) -> int:      return self._read_reg(REG_RD_CURRENT_LITTLE_PITCH)

    def get_thumb_torque(self) -> int:      return self._read_reg(REG_RD_CURRENT_THUMB_TORQUE)
    def get_thumb_yaw_torque(self) -> int:  return self._read_reg(REG_RD_CURRENT_THUMB_YAW_TORQUE)
    def get_index_torque(self) -> int:      return self._read_reg(REG_RD_CURRENT_INDEX_TORQUE)
    def get_middle_torque(self) -> int:     return self._read_reg(REG_RD_CURRENT_MIDDLE_TORQUE)
    def get_ring_torque(self) -> int:       return self._read_reg(REG_RD_CURRENT_RING_TORQUE)
    def get_little_torque(self) -> int:     return self._read_reg(REG_RD_CURRENT_LITTLE_TORQUE)

    def get_thumb_speed(self) -> int:       return self._read_reg(REG_RD_CURRENT_THUMB_SPEED)
    def get_thumb_yaw_speed(self) -> int:   return self._read_reg(REG_RD_CURRENT_THUMB_YAW_SPEED)
    def get_index_speed(self) -> int:       return self._read_reg(REG_RD_CURRENT_INDEX_SPEED)
    def get_middle_speed(self) -> int:      return self._read_reg(REG_RD_CURRENT_MIDDLE_SPEED)
    def get_ring_speed(self) -> int:        return self._read_reg(REG_RD_CURRENT_RING_SPEED)
    def get_little_speed(self) -> int:      return self._read_reg(REG_RD_CURRENT_LITTLE_SPEED)

    def get_thumb_temp(self) -> int:        return self._read_reg(REG_RD_THUMB_TEMP)
    def get_thumb_yaw_temp(self) -> int:    return self._read_reg(REG_RD_THUMB_YAW_TEMP)
    def get_index_temp(self) -> int:        return self._read_reg(REG_RD_INDEX_TEMP)
    def get_middle_temp(self) -> int:       return self._read_reg(REG_RD_MIDDLE_TEMP)
    def get_ring_temp(self) -> int:         return self._read_reg(REG_RD_RING_TEMP)
    def get_little_temp(self) -> int:       return self._read_reg(REG_RD_LITTLE_TEMP)

    def get_thumb_error(self) -> int:       return self._read_reg(REG_RD_THUMB_ERROR)
    def get_thumb_yaw_error(self) -> int:   return self._read_reg(REG_RD_THUMB_YAW_ERROR)
    def get_index_error(self) -> int:       return self._read_reg(REG_RD_INDEX_ERROR)
    def get_middle_error(self) -> int:      return self._read_reg(REG_RD_MIDDLE_ERROR)
    def get_ring_error(self) -> int:        return self._read_reg(REG_RD_RING_ERROR)
    def get_little_error(self) -> int:      return self._read_reg(REG_RD_LITTLE_ERROR)

    def get_hand_freedom(self) -> int:      return self._read_reg(REG_RD_HAND_FREEDOM)
    def get_hand_version(self) -> int:      return self._read_reg(REG_RD_HAND_VERSION)
    def get_hand_number(self) -> int:       return self._read_reg(REG_RD_HAND_NUMBER)
    def get_hand_direction(self) -> int:    return self._read_reg(REG_RD_HAND_DIRECTION)
    def get_software_version(self) -> int:  return self._read_reg(REG_RD_SOFTWARE_VERSION)
    def get_hardware_version(self) -> int:  return self._read_reg(REG_RD_HARDWARE_VERSION)
    
    # ----------------------------------------------------------
    # Batch getters (using read_all_... methods)
    # ----------------------------------------------------------
    def get_state(self) -> List[int]:
        """Get finger motor state (angles)"""
        return self.read_all_angles()

    def get_torque(self) -> List[int]:
        """Get current torque"""
        return self.read_all_torques()

    def get_speed(self) -> List[int]:
        """Get current speed"""
        return self.read_all_speeds()

    def get_temperature(self) -> List[int]:
        """Get current motor temperature"""
        return self.read_all_temperatures()
    
    def get_fault(self) -> List[int]:
        """Get current motor fault codes"""
        return self.read_all_errors()
    
    def get_version(self) -> list:
        """Get current firmware version numbers"""
        return self.read_all_versions()


    # ----------------------------------------------------------
    # Write holding registers (single register write)
    # ----------------------------------------------------------
    def _write_reg(self, addr: int, value: int):
        """Write single holding register (function code 16) with 30 ms frame interval"""
        if not 0 <= value <= 255:
            raise ValueError("value must be 0-255")
        
        # Ensure value is a native Python int
        self._execute_write(addr, [int(value)])

    def _write_regs(self, addr: int, values: List[int]):
        """Write multiple holding registers (function code 16) with 30 ms frame interval"""
        # At this point values should already be validated by is_valid_6xuint8 and converted to Python ints
        if not all(0 <= v <= 255 for v in values):
            # This should theoretically not trigger because the upper-level call has validated
            raise ValueError("All values must be 0-255")
        self._execute_write(addr, values)


    def set_thumb_pitch(self, v: int):           self._write_reg(REG_WR_THUMB_PITCH, v)
    def set_thumb_yaw(self, v: int):             self._write_reg(REG_WR_THUMB_YAW, v)
    def set_index_pitch(self, v: int):           self._write_reg(REG_WR_INDEX_PITCH, v)
    def set_middle_pitch(self, v: int):          self._write_reg(REG_WR_MIDDLE_PITCH, v)
    def set_ring_pitch(self, v: int):            self._write_reg(REG_WR_RING_PITCH, v)
    def set_little_pitch(self, v: int):          self._write_reg(REG_WR_LITTLE_PITCH, v)

    def set_thumb_torque(self, v: int):          self._write_reg(REG_WR_THUMB_TORQUE, v)
    def set_thumb_yaw_torque(self, v: int):      self._write_reg(REG_WR_THUMB_YAW_TORQUE, v)
    def set_index_torque(self, v: int):          self._write_reg(REG_WR_INDEX_TORQUE, v)
    def set_middle_torque(self, v: int):         self._write_reg(REG_WR_MIDDLE_TORQUE, v)
    def set_ring_torque(self, v: int):           self._write_reg(REG_WR_RING_TORQUE, v)
    def set_little_torque(self, v: int):         self._write_reg(REG_WR_LITTLE_TORQUE, v)

    def set_thumb_speed(self, v: int):           self._write_reg(REG_WR_THUMB_SPEED, v)
    def set_thumb_yaw_speed(self, v: int):       self._write_reg(REG_WR_THUMB_YAW_SPEED, v)
    def set_index_speed(self, v: int):           self._write_reg(REG_WR_INDEX_SPEED, v)
    def set_middle_speed(self, v: int):          self._write_reg(REG_WR_MIDDLE_SPEED, v)
    def set_ring_speed(self, v: int):            self._write_reg(REG_WR_RING_SPEED, v)
    def set_little_speed(self, v: int):          self._write_reg(REG_WR_LITTLE_SPEED, v)

    # ----------------------------------------------------------
    # Fixed functions (using batch write optimization)
    # ----------------------------------------------------------
    def is_valid_6xuint8(self, lst: List[Any]) -> bool:
        """
        Validate a list of six integers between 0 and 255.
        Allows inputs containing floats, NumPy integers, etc. that can be converted to int, with range checks.
        """
        if not (isinstance(lst, list) and len(lst) == 6):
            return False
            
        try:
            # Key: try converting all elements to native Python int
            int_values = [int(v) for v in lst]
        except (ValueError, TypeError):
            # Conversion failed; list contains elements that cannot be converted
            return False

        # Check whether converted integer list is in range 0-255
        return all(0 <= x <= 255 for x in int_values)
    
    def set_joint_positions(self, joint_angles: List[Any] = None):
        joint_angles = joint_angles or [0] * 6
        
        if not self.is_valid_6xuint8(joint_angles):
            logging.error(f"Invalid joint angles received: {joint_angles}")
            raise ValueError("Joint angles must be a list of 6 values between 0 and 255 (convertible to int).")

        # Force conversion to Modbus-compatible native Python int list
        int_angles = [int(v) for v in joint_angles] 
        
        # Batch write 6 angle registers (starting from REG_WR_THUMB_PITCH address 0, count=6)
        self._write_regs(REG_WR_THUMB_PITCH, int_angles)

    def set_speed(self, speed: List[Any] = None):
        speed = speed or [200] * 6
        if not self.is_valid_6xuint8(speed):
            logging.error(f"Invalid speed values received: {speed}")
            raise ValueError("Speed values must be a list of 6 values between 0 and 255 (convertible to int).")
        
        int_speed = [int(v) for v in speed]
        self._write_regs(REG_WR_THUMB_SPEED, int_speed)
    
    def set_torque(self, torque: List[Any] = None):
        torque = torque or [200] * 6
        if not self.is_valid_6xuint8(torque):
            logging.error(f"Invalid torque values received: {torque}")
            raise ValueError("Torque values must be a list of 6 values between 0 and 255 (convertible to int).")
            
        int_torque = [int(v) for v in torque]
        self._write_regs(REG_WR_THUMB_TORQUE, int_torque)

    # ... (other fixed functions remain unchanged) ...

    def set_current(self, current: List[int] = None):
        print("Current O6 does not support setting current", flush=True)
        pass

    def get_state_for_pub(self) -> list:
        return self.get_state()

    def get_current_status(self) -> list:
        return self.get_state()
    
    def get_joint_speed(self) -> list:
        return self.get_speed()
    
    def get_touch_type(self) -> list:
        return -1
    
    def get_normal_force(self) -> list:
        return [-1] * 5
    
    def get_tangential_force(self) -> list:
        return [-1] * 5
    
    def get_approach_inc(self) -> list:
        return [-1] * 5
    
    def get_touch(self) -> list:
        return [-1] * 5
    
    def get_thumb_matrix_touch(self,sleep_time=0):
        return np.full((12, 6), -1)

    def get_index_matrix_touch(self,sleep_time=0):
        return np.full((12, 6), -1)

    def get_middle_matrix_touch(self,sleep_time=0):
        return np.full((12, 6), -1)

    def get_ring_matrix_touch(self,sleep_time=0):
        return np.full((12, 6), -1)

    def get_little_matrix_touch(self,sleep_time=0):
        return np.full((12, 6), -1)

    def get_matrix_touch(self) -> list:
        thumb_matrix = np.full((12, 6), -1)
        index_matrix = np.full((12, 6), -1)
        middle_matrix = np.full((12, 6), -1)
        ring_matrix = np.full((12, 6), -1)
        little_matrix = np.full((12, 6), -1)
        return thumb_matrix , index_matrix , middle_matrix , ring_matrix , little_matrix
    
    def get_matrix_touch_v2(self) -> list:
        return self.get_matrix_touch()
    
    # ----------------------------------------------------------
    # Context management
    # ----------------------------------------------------------
    def close(self):
        if hasattr(self, 'connected') and self.connected:
            self.cli.close()
            self.connected = False
            logging.info("Modbus connection closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ----------------------------------------------------------
    # Convenience functions
    # ----------------------------------------------------------
    def set_all_fingers(self, pitch: int):
        """Set flexion angle (0-255) of all five fingers at the same time using batch write"""
        # Allow passing float/numpy int, etc. that can be converted to int
        try:
            pitch_int = int(pitch)
        except (ValueError, TypeError):
             raise ValueError("Pitch value must be a number convertible to int (0-255)")

        if not 0 <= pitch_int <= 255:
            raise ValueError("Pitch value must be 0-255")
        
        # Batch set the angles of all 6 joints
        self.set_joint_positions([pitch_int] * 6)

    def relax(self):
        """Extend all fingers (255)"""
        self.set_all_fingers(255)

    def fist(self):
        """Flex all fingers (0)"""
        self.set_all_fingers(0)

    def dump_status(self):
        """Print all current readable states (using batch read optimization)"""
        print("--------- O6 Hand Status ---------")
        
        angles = self.get_state()
        temps = self.get_temperature()
        errors = self.get_fault()
        versions = self.get_version()

        print(f"Joint Angles: {angles}")
        print(f"Temperature:  {temps}℃")
        print(f"Error Codes:  {errors}")
        print(f"Versions:     {versions}")
        print("----------------------------------")

# ------------------------------------------------------------------
# Command-line quick test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    
    # Assume default station ID is 0x27 (39)
    DEFAULT_HAND_ID = 0x27

    parser = argparse.ArgumentParser(description="O6 Hand Modbus tester (using pymodbus 3.5.1)")
    parser.add_argument("-p", "--port", required=True, help="Serial port, e.g. /dev/ttyUSB0")
    parser.add_argument("-l", "--left", action="store_const", const=0x28, default=DEFAULT_HAND_ID, dest='hand_id', help="Left hand (0x28), default right hand (0x27)")
    
    args = parser.parse_args()

    try:
        # Use with-statement to ensure connection closure, which is recommended by pymodbus
        with RealHandO6RS485(hand_id=args.hand_id, modbus_port=args.port, baudrate=115200) as hand:
            hand.dump_status()
            print("Executing relax → extend")
            hand.relax()
            time.sleep(1)
            print("Executing fist → make fist")
            hand.fist()
            time.sleep(1)
            hand.relax()
            print("Demo finished")
            
    except ConnectionError as e:
        print(f"Connection error: {e}")
    except RuntimeError as e:
        print(f"Modbus runtime error: {e}")
    except StructError as e:
        print(f"Data structure error (please check whether input data type is native int): {e}")
    except Exception as e:
        print(f"Other error occurred: {e}")
