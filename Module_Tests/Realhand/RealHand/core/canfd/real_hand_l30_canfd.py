#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from .l30_canfd_controller import DexterousHandController


class RealHandL30Canfd:
    def __init__(self, hand_type="right"):
        self.hand_type = hand_type
        if self.hand_type == "right":
            device_id = DexterousHandController.DEVICE_ID_RIGHT
        else:
            device_id = DexterousHandController.DEVICE_ID_LEFT
        self.controller = DexterousHandController(device_id=device_id)
        ok, _device_type = self.controller.connect()
        if not ok:
            raise RuntimeError("L30 CANFD connect failed")
        # Set reasonable defaults once connected
        try:
            self.controller.set_default_velocity(100)
            time.sleep(0.05)
            self.controller.set_default_torque(500)
            time.sleep(0.05)
        except Exception:
            # Keep API resilient; caller can still control manually
            pass

    def set_joint_positions(self, pose):
        if len(pose) != 17:
            raise ValueError("L30 requires 17 joint values")
        # RealHand API uses 0-255
        norm = [int(max(0, min(255, v))) for v in pose]
        angles = self.controller.denormalize_motor_values(norm)
        return self.controller.set_joint_positions(angles)

    def set_speed(self, speed):
        if len(speed) == 0:
            return False
        if len(speed) == 1:
            speed = speed * 17
        if len(speed) != 17:
            raise ValueError("L30 speed requires 17 values")
        # Map 0-255 to 0-1000 (protocol expects raw velocity values)
        scaled = [int(max(0, min(1000, round(v / 255.0 * 1000)))) for v in speed]
        return self.controller.set_joint_velocities(scaled)

    def set_torque(self, torque):
        if len(torque) == 0:
            return False
        if len(torque) == 1:
            torque = torque * 17
        if len(torque) != 17:
            raise ValueError("L30 torque requires 17 values")
        # Map 0-255 to 0-1000
        scaled = [int(max(0, min(1000, round(v / 255.0 * 1000)))) for v in torque]
        return self.controller.set_joint_torques(scaled)

    def get_version(self):
        # No embedded version query exposed in controller; return software/hardware if available
        try:
            product_model, serial_number, software_version, hardware_version, hand_type = self.controller.get_device_info()
            return f"{product_model}:{software_version}:{hardware_version}:{hand_type}" if product_model else None
        except Exception:
            return None

    def get_current_status(self):
        state = self.controller.get_current_state()
        if state is None:
            return [0] * 17
        _state_arc, state_range = state
        return state_range

    def get_current_pub_status(self):
        return self.get_current_status()

    def get_speed(self):
        try:
            return self.controller.get_current_speed()
        except Exception:
            return [0] * 17

    def get_force(self):
        # Not supported via CANFD controller
        return None

    def get_touch_type(self):
        return None

    def get_touch(self):
        return None

    def get_matrix_touch(self):
        return None

    def get_matrix_touch_v2(self):
        return None

    def get_thumb_matrix_touch(self, *args, **kwargs):
        return None

    def get_index_matrix_touch(self, *args, **kwargs):
        return None

    def get_middle_matrix_touch(self, *args, **kwargs):
        return None

    def get_ring_matrix_touch(self, *args, **kwargs):
        return None

    def get_little_matrix_touch(self, *args, **kwargs):
        return None

    def get_torque(self):
        return None

    def get_temperature(self):
        return None

    def get_fault(self):
        try:
            return self.controller.get_error_status()
        except Exception:
            return None

    def clear_faults(self):
        return None

    def set_enable_mode(self):
        return None

    def set_disability_mode(self):
        return None

    def get_finger_order(self):
        return []

    def close(self):
        try:
            self.controller.disconnect()
        except Exception:
            pass
