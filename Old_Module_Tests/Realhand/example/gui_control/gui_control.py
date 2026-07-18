#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys
import time, json
import threading
from dataclasses import dataclass
from typing import List, Dict
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QObject, QEvent
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QSlider, QLabel, QPushButton, QGroupBox, QScrollArea, QTabWidget, 
    QFrame, QSplitter, QMessageBox, QTextEdit
)
from PyQt5.QtGui import QFont, QPainter, QColor, QBrush
from PyQt5.QtCore import QRect
from config.constants import _HAND_CONFIGS
current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(target_dir)

from RealHand.real_hand_api import RealHandApi
from RealHand.utils.load_write_yaml import LoadWriteYaml
from RealHand.utils.color_msg import ColorMsg


LOOP_TIME = 1000 # Cycle action interval in milliseconds

class DotMatrixWidget(QWidget):
    """Dot Matrix Display Widget - White to Dark Red Gradient Version"""
    
    def __init__(self, parent=None, rows=12, cols=6):
        super().__init__(parent)
        self.rows = rows
        self.cols = cols
        self.dot_size = 6
        self.spacing = 3
        self.data = None
        
        # Calculate widget size - add some extra space to ensure full display
        width = cols * (self.dot_size + self.spacing) + self.spacing + 2
        height = rows * (self.dot_size + self.spacing) + self.spacing + 2
        self.setMinimumSize(width, height)
        self.setMaximumSize(width, height)
        
    def set_data(self, data):
        """Set dot matrix data"""
        if data is not None:
            try:
                # If data is a numpy array, convert to list
                if hasattr(data, 'tolist'):
                    self.data = data.tolist()
                else:
                    self.data = data
            except Exception as e:
                print(f"Error converting data: {e}")
                self.data = None
        else:
            self.data = None
        self.update()
        
    def paintEvent(self, event):
        """Draw dot matrix"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background
        painter.fillRect(self.rect(), QColor('white'))
        
        for row in range(self.rows):
            for col in range(self.cols):
                x = self.spacing + col * (self.dot_size + self.spacing)
                y = self.spacing + row * (self.dot_size + self.spacing)
                
                if self.data is not None:
                    try:
                        # Handle numpy array or normal list
                        if hasattr(self.data, 'flatten'):
                            flat_data = self.data.flatten()
                        else:
                            flat_data = self.data
                        
                        index = row * self.cols + col
                        if index < len(flat_data):
                            value = flat_data[index]
                            # Ensure value is a scalar number
                            if hasattr(value, 'item'):
                                value = value.item()
                                
                            if value > 0:
                                # White to dark red gradient logic
                                intensity = min(255, max(0, int(value)))
                                
                                if intensity < 128:
                                    # White to light red gradient
                                    red = 255
                                    green = 255 - (intensity * 55 // 128)
                                    blue = 255 - (intensity * 55 // 128)
                                    color = QColor(red, green, blue)
                                else:
                                    # Light red to dark red gradient
                                    red = 255
                                    green = 200 - ((intensity - 128) * 200 // 127)
                                    blue = 200 - ((intensity - 128) * 200 // 127)
                                    color = QColor(red, green, blue)
                            else:
                                color = QColor('#c8c8c8')  # Default gray
                        else:
                            color = QColor('#c8c8c8')  # Default gray
                    except Exception as e:
                        print(f"Error drawing matrix: {e}")
                        color = QColor('#c8c8c8')  # Default gray
                else:
                    color = QColor('#c8c8c8')  # Default gray
                
                # Draw dots
                painter.setBrush(QBrush(color))
                painter.setPen(QColor('#666666'))
                painter.drawEllipse(x, y, self.dot_size, self.dot_size)

class MatrixDisplayWidget(QWidget):
    """Matrix display widget, containing dot matrices for five fingers - New arrangement"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.finger_matrices = {}
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI - New arrangement"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # First row: Thumb, Index, Middle
        first_row_layout = QHBoxLayout()
        first_row_layout.setSpacing(15)
        
        # Thumb
        thumb_frame = self.create_finger_frame("Thumb", "thumb_matrix")
        first_row_layout.addWidget(thumb_frame)
        
        # Index
        index_frame = self.create_finger_frame("Index", "index_matrix")
        first_row_layout.addWidget(index_frame)
        
        # Middle
        middle_frame = self.create_finger_frame("Middle", "middle_matrix")
        first_row_layout.addWidget(middle_frame)
        
        first_row_layout.addStretch()
        main_layout.addLayout(first_row_layout)
        
        # Second row: Ring, Pinky
        second_row_layout = QHBoxLayout()
        second_row_layout.setSpacing(15)
        
        # Ring
        ring_frame = self.create_finger_frame("Ring", "ring_matrix")
        second_row_layout.addWidget(ring_frame)
        
        # Pinky
        little_frame = self.create_finger_frame("Pinky", "little_matrix")
        second_row_layout.addWidget(little_frame)
        
        second_row_layout.addStretch()
        main_layout.addLayout(second_row_layout)
        
        main_layout.addStretch()
        
    def create_finger_frame(self, display_name, finger_name):
        """Create frame for a single finger"""
        finger_frame = QWidget()
        finger_layout = QVBoxLayout(finger_frame)
        finger_layout.setSpacing(5)
        finger_layout.setContentsMargins(5, 5, 5, 5)
        
        # Finger Label - Center aligned
        label = QLabel(display_name)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-weight: bold;")
        finger_layout.addWidget(label)
        
        # Create dot matrix widget
        matrix = DotMatrixWidget()
        finger_layout.addWidget(matrix, 0, Qt.AlignCenter)
        
        self.finger_matrices[finger_name] = matrix
        return finger_frame
        
    def update_matrix_data(self, finger_name, data):
        """Update dot matrix data for specified finger"""
        if finger_name in self.finger_matrices:
            # Handle numpy array or normal list
            if data is not None:
                try:
                    # If data is numpy array, convert to list first
                    if hasattr(data, 'tolist'):
                        data = data.tolist()
                    
                    # Flatten data (if data is 2D list or 2D array)
                    if data and (isinstance(data[0], list) or 
                            (hasattr(data, 'ndim') and data.ndim > 1)):
                        # If 2D structure, flatten to 1D
                        flattened = []
                        for item in data:
                            if isinstance(item, (list, tuple)) or (hasattr(item, '__iter__') and not isinstance(item, str)):
                                flattened.extend(item)
                            else:
                                flattened.append(item)
                    else:
                        # If already 1D, use directly
                        flattened = data
                        
                    self.finger_matrices[finger_name].set_data(flattened)
                except Exception as e:
                    print(f"Error processing matrix data: {e}")
                    # Use default gray matrix on error
                    default_data = [0] * 72
                    self.finger_matrices[finger_name].set_data(default_data)
            else:
                # Use default gray matrix when data is empty
                default_data = [0] * 72
                self.finger_matrices[finger_name].set_data(default_data)
            
    def initialize_default_matrices(self):
        """Initialize display with default matrix (all dots gray)"""
        default_data = [0] * 72  # 12x6 zero matrix
        for finger_name in self.finger_matrices.keys():
            self.finger_matrices[finger_name].set_data(default_data)

class HandApiManager(QObject):
    """Hand API Manager, handles communication with RealHandApi"""
    status_updated = pyqtSignal(str, str)  # Status type, Message content
    matrix_data_updated = pyqtSignal(dict)  # Matrix data update signal

    def __init__(self):
        super().__init__()
        self.hand_joint = None
        self.hand_type = None
        self.api = None
        self._init_real_hand_type()
        # Initialize API
        self.init_api()
        
        # Matrix data update timer
        self.matrix_timer = QTimer(self)
        self.matrix_timer.timeout.connect(self.update_matrix_data)
        self.matrix_timer.start(500)  # Update matrix data every 500ms
        self.lock = False

    def _init_real_hand_type(self):
        try:
            self.yaml = LoadWriteYaml() # Initialize configuration file
            # Read configuration file
            self.setting = self.yaml.load_setting_yaml()
            time.sleep(1)
            self.left_hand = False
            self.right_hand = False
            if self.setting['REAL_HAND']['LEFT_HAND']['EXISTS'] == True:
                self.left_hand = True
            elif self.setting['REAL_HAND']['RIGHT_HAND']['EXISTS'] == True:
                self.right_hand = True
            # GUI control only supports single hand, mutual exclusion for left/right hand here
            if self.left_hand == True and self.right_hand == True:
                self.left_hand = True
                self.right_hand = False
            if self.left_hand == True:
                self.hand_exists = True
                self.hand_joint = self.setting['REAL_HAND']['LEFT_HAND']['JOINT']
                self.hand_type = "left"
                self.is_touch = self.setting['REAL_HAND']['LEFT_HAND']['TOUCH']
                self.can = self.setting['REAL_HAND']['LEFT_HAND']['CAN']
                self.modbus = self.setting['REAL_HAND']['LEFT_HAND']['MODBUS']
            if self.right_hand == True:
                self.hand_exists = True
                self.hand_joint = self.setting['REAL_HAND']['RIGHT_HAND']['JOINT']
                self.hand_type = "right"
                self.is_touch = self.setting['REAL_HAND']['RIGHT_HAND']['TOUCH']
                self.can = self.setting['REAL_HAND']['RIGHT_HAND']['CAN']
                self.modbus = self.setting['REAL_HAND']['RIGHT_HAND']['MODBUS']
        except Exception as e:
            ColorMsg(msg=f"Error Failed to read config file: {str(e)}", color="red")
            self.status_updated.emit("error", f"Failed to read config file: {str(e)}")
        ColorMsg(msg=f"Current config: Real Hand {self.hand_type} {self.hand_joint} Pressure Sensor:{self.is_touch} modbus:{self.modbus} CAN:{self.can}", color="green")

    def init_api(self):
        """Initialize RealHandApi"""
        try:
            if self.hand_joint == "L30":
                self._validate_l30_canfd_deps()
            self.api = RealHandApi(hand_joint=self.hand_joint, hand_type=self.hand_type, modbus=self.modbus, can=self.can)
            self.status_updated.emit("info", f"Hand API initialized successfully: {self.hand_type} {self.hand_joint}")
        except Exception as e:
            self.status_updated.emit("error", f"API initialization failed: {str(e)}")
            raise

    def _validate_l30_canfd_deps(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "RealHand", "third_party", "canfd"))
        libcanbus = os.path.join(base_dir, "libcanbus.so")
        libusb = os.path.join(base_dir, "libusb-1.0.so")
        missing = []
        if not os.path.exists(libcanbus):
            missing.append(libcanbus)
        if not os.path.exists(libusb):
            missing.append(libusb)
        if missing:
            self.status_updated.emit(
                "error",
                f"L30 CANFD deps missing: {', '.join(missing)}. Install canfd libs into RealHand/third_party/canfd."
            )

    def update_matrix_data(self):
        """Update matrix data"""
        if not self.api:
            return
            
        try:
            # Get matrix touch data
            matrix_data = {}
            # Get corresponding matrix data based on hand model
            if self.is_touch == True and self.lock == False:
                # Get matrix touch data for each finger
                thumb_data = self.api.get_thumb_matrix_touch()
                index_data = self.api.get_index_matrix_touch()
                middle_data = self.api.get_middle_matrix_touch()
                ring_data = self.api.get_ring_matrix_touch()
                little_data = self.api.get_little_matrix_touch()
                matrix_data = {
                    "thumb_matrix": thumb_data,
                    "index_matrix": index_data,
                    "middle_matrix": middle_data,
                    "ring_matrix": ring_data,
                    "little_matrix": little_data
                }
                
                # Emit matrix data update signal
                self.matrix_data_updated.emit(matrix_data)
                
        except Exception as e:
            # If data retrieval fails, send empty data
            default_data = [0] * 72
            matrix_data = {
                "thumb_matrix": default_data,
                "index_matrix": default_data,
                "middle_matrix": default_data,
                "ring_matrix": default_data,
                "little_matrix": default_data
            }
            self.matrix_data_updated.emit(matrix_data)

    def publish_joint_state(self, positions: List[int]):
        """Publish joint state message"""
        if not self.api:
            self.status_updated.emit("error", "Hand API not initialized")
            return
            
        try:
            self.lock = True
            # Call API to send joint positions
            self.api.finger_move(positions)
            self.status_updated.emit("info", "Joint state sent")
        except Exception as e:
            self.status_updated.emit("error", f"Send failed: {str(e)}")
        self.lock = False

    def publish_speed(self, val: int):
        """Publish speed setting"""
        if not self.api:
            self.status_updated.emit("error", "Hand API not initialized")
            return
            
        try:
            joint_len = 0
            if (self.hand_joint.upper() == "O6" or self.hand_joint.upper() == "L6"):
                joint_len = 6
            elif self.hand_joint == "L7":
                joint_len = 7
            elif self.hand_joint == "L10":
                joint_len = 10
            elif self.hand_joint == "L30":
                joint_len = 17
            else:
                joint_len = 5
                
            speed_values = [val] * joint_len
            self.api.set_speed(speed_values)
            
            self.status_updated.emit("info", f"Speed set to {speed_values}")
            print(f"Speed values: {speed_values}", flush=True)
        except Exception as e:
            self.status_updated.emit("error", f"Failed to set speed: {str(e)}")

    def publish_torque(self, val: int):
        """Publish torque setting"""
        if not self.api:
            self.status_updated.emit("error", "Hand API not initialized")
            return
            
        try:
            joint_len = 0
            if (self.hand_joint.upper() == "O6" or self.hand_joint.upper() == "L6"):
                joint_len = 6
            elif self.hand_joint == "L7":
                joint_len = 7
            elif self.hand_joint == "L10":
                joint_len = 10
            elif self.hand_joint == "L30":
                joint_len = 17
            else:
                joint_len = 5
                
            torque_values = [val] * joint_len
            self.api.set_torque(torque_values)
            
            self.status_updated.emit("info", f"Torque set to {torque_values}")
            print(f"Torque values: {torque_values}", flush=True)
        except Exception as e:
            self.status_updated.emit("error", f"Failed to set torque: {str(e)}")

    def shutdown(self):
        """Close API connection"""
        if self.api:
            try:
                self.api.close_can()
                self.status_updated.emit("info", "API connection closed")
            except Exception as e:
                self.status_updated.emit("error", f"Failed to close API: {str(e)}")
        if self.matrix_timer.isActive():
            self.matrix_timer.stop()

class HandControlGUI(QWidget):
    """Dexterous Hand Control Interface"""
    status_updated = pyqtSignal(str, str)  # Status type, Message content

    def __init__(self, api_manager: HandApiManager):
        super().__init__()
        
        # Loop control variables
        self.cycle_timer = None  # Loop timer
        self.current_action_index = -1  # Current action index
        self.preset_buttons = []  # Store preset action button references
        
        # Set API manager
        self.api_manager = api_manager
        self.api_manager.status_updated.connect(self.update_status)
        self.api_manager.matrix_data_updated.connect(self.update_matrix_display)
        
        # Get hand configuration
        self.hand_joint = self.api_manager.hand_joint
        self.hand_type = self.api_manager.hand_type
        self.hand_config = _HAND_CONFIGS[self.hand_joint]
        
        # Initialize UI
        self.init_ui()
        
        # Set timer to publish joint state
        self.publish_timer = QTimer(self)
        self.publish_timer.setInterval(30)  # 10Hz publish frequency
        self.publish_timer.timeout.connect(self.publish_joint_state)
        self.publish_timer.start()

    def init_ui(self):
        """Initialize User Interface"""
        # Set window properties
        self.setWindowTitle(f'Dexterous Hand Control Interface - {self.hand_type} {self.hand_joint}')
        self.setMinimumSize(1200, 900)
        
        # Set styles
        self.setStyleSheet("""
            QWidget {
                font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
                font-size: 12px;
            }
            QGroupBox {
                border: 1px solid #CCCCCC;
                border-radius: 6px;
                margin-top: 6px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #165DFF;
                font-weight: bold;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                border-radius: 4px;
                background: #CCCCCC;
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #165DFF, stop:1 #0E42D2);
                border: 1px solid #5C8AFF;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QPushButton {
                background-color: #E0E0E0;
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                padding: 5px 10px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #F0F0F0;
            }
            QPushButton:pressed {
                background-color: #D0D0D0;
            }
            QPushButton[category="preset"] {
                background-color: #E6F7FF;
                color: #1890FF;
                border-color: #91D5FF;
            }
            QPushButton[category="preset"]:hover {
                background-color: #B3E0FF;
            }
            QPushButton[category="action"] {
                background-color: #FFF7E6;
                color: #FA8C16;
                border-color: #FFD591;
            }
            QPushButton[category="action"]:hover {
                background-color: #FFE6B3;
            }
            QPushButton[category="danger"] {
                background-color: #FFF1F0;
                color: #F5222D;
                border-color: #FFCCC7;
            }
            QPushButton[category="danger"]:hover {
                background-color: #FFE8E6;
            }
            QLabel#StatusLabel {
                padding: 5px;
                border-radius: 4px;
            }
            QLabel#StatusInfo {
                background-color: #F0F7FF;
                color: #0066CC;
            }
            QLabel#StatusError {
                background-color: #FFF0F0;
                color: #CC0000;
            }
            /* Value display panel style */
            QTextEdit#ValueDisplay {
                background-color: #F8F8F8;
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                padding: 10px;
                font-family: Consolas, monospace;
                font-size: 12px;
            }
        """)
        
        # Create main vertical layout
        main_layout = QVBoxLayout(self)
        
        # Create horizontal splitter (originally three panels)
        splitter = QSplitter(Qt.Horizontal)
        
        # Create left joint control panel
        self.joint_control_panel = self.create_joint_control_panel()
        splitter.addWidget(self.joint_control_panel)
        
        # Create middle preset actions panel
        self.preset_actions_panel = self.create_preset_actions_panel()
        splitter.addWidget(self.preset_actions_panel)
        
        # Create right status monitor panel
        self.status_monitor_panel = self.create_status_monitor_panel()
        splitter.addWidget(self.status_monitor_panel)
        
        # Set splitter sizes
        splitter.setSizes([500, 300, 400])
        
        # Add splitter to main layout, set stretch factor to 1 (resizable)
        main_layout.addWidget(splitter, stretch=1)
        
        # Create and add value display panel, set stretch factor to 0 (non-resizable)
        self.value_display_panel = self.create_value_display_panel()
        main_layout.addWidget(self.value_display_panel, stretch=0)
        
        # Initial update of value display
        self.update_value_display()

    def create_joint_control_panel(self):
        """Create Joint Control Panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create title
        title_label = QLabel(f"Joint Control - {self.hand_joint}")
        title_label.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        layout.addWidget(title_label)

        # Create slider scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        
        scroll_content = QWidget()
        self.sliders_layout = QGridLayout(scroll_content)
        self.sliders_layout.setSpacing(10)
        
        # Create sliders
        self.create_joint_sliders()
        
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)
        
        return panel

    def create_joint_sliders(self):
        """Create joint sliders"""
        # Clear existing sliders
        for i in reversed(range(self.sliders_layout.count())):
            item = self.sliders_layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
        
        # Create new sliders
        self.sliders = []
        self.slider_labels = []
        
        for i, (name, value) in enumerate(zip(
            self.hand_config.joint_names, self.hand_config.init_pos
        )):
            # Create label
            label = QLabel(f"{name}: {value}")
            label.setMinimumWidth(120)
            
            # Create slider
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 255)
            slider.setValue(value)
            slider.valueChanged.connect(
                lambda val, idx=i: self.on_slider_value_changed(idx, val)
            )
            
            # Add to layout
            row, col = divmod(i, 1)
            self.sliders_layout.addWidget(label, row, 0)
            self.sliders_layout.addWidget(slider, row, 1)
            
            self.sliders.append(slider)
            self.slider_labels.append(label)

    def create_preset_actions_panel(self):
        """Create Preset Actions Panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # System Presets
        sys_preset_group = QGroupBox("System Presets")
        sys_preset_layout = QGridLayout(sys_preset_group)
        sys_preset_layout.setSpacing(8)
        
        # Add system preset action buttons
        self.create_system_preset_buttons(sys_preset_layout)
        layout.addWidget(sys_preset_group)
        
        # Add action buttons
        actions_layout = QHBoxLayout()
        
        # Add cycle run button
        self.cycle_button = QPushButton("Cycle Preset Actions")
        self.cycle_button.setProperty("category", "action")
        self.cycle_button.clicked.connect(self.on_cycle_clicked)
        actions_layout.addWidget(self.cycle_button)
        
        self.home_button = QPushButton("Return to Home")
        self.home_button.setProperty("category", "action")
        self.home_button.clicked.connect(self.on_home_clicked)
        actions_layout.addWidget(self.home_button)
        
        self.stop_button = QPushButton("Stop All Actions")
        self.stop_button.setProperty("category", "danger")
        self.stop_button.clicked.connect(self.on_stop_clicked)
        actions_layout.addWidget(self.stop_button)
        
        layout.addLayout(actions_layout)
        
        return panel

    def create_system_preset_buttons(self, parent_layout):
        """Create system preset action buttons"""
        self.preset_buttons = []  # Clear button list
        if self.hand_config.preset_actions:
            buttons = []
            for idx, (name, positions) in enumerate(self.hand_config.preset_actions.items()):
                button = QPushButton(name)
                button.setProperty("category", "preset")
                button.clicked.connect(
                    lambda checked, pos=positions: self.on_preset_action_clicked(pos)
                )
                buttons.append(button)
                self.preset_buttons.append(button)  # Store button references
                
            # Add to grid layout
            cols = 2
            for i, button in enumerate(buttons):
                row, col = divmod(i, cols)
                parent_layout.addWidget(button, row, col)

    def create_status_monitor_panel(self):
        """Create Status Monitor Panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Title
        title_label = QLabel("Status Monitor")
        title_label.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        layout.addWidget(title_label)

        # Quick Settings
        quick_set_gb = QGroupBox("Quick Settings")
        qv_layout = QVBoxLayout(quick_set_gb)

        # Speed Row
        speed_hbox = QHBoxLayout()
        speed_hbox.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(0, 255)
        self.speed_slider.setValue(255)
        self.speed_slider.setMinimumWidth(150)
        speed_hbox.addWidget(self.speed_slider)
        self.speed_val_lbl = QLabel("255")
        self.speed_val_lbl.setMinimumWidth(30)
        speed_hbox.addWidget(self.speed_val_lbl)
        self.speed_btn = QPushButton("Set Speed")
        self.speed_btn.clicked.connect(
            lambda: (
                self.api_manager.publish_speed(self.speed_slider.value()),
                self.status_updated.emit(
                    "info", f"Speed set to {self.speed_slider.value()}")
            ))
        speed_hbox.addWidget(self.speed_btn)
        speed_hbox.addStretch()
        qv_layout.addLayout(speed_hbox)

        # Torque Row
        torque_hbox = QHBoxLayout()
        torque_hbox.addWidget(QLabel("Torque:"))
        self.torque_slider = QSlider(Qt.Horizontal)
        self.torque_slider.setRange(0, 255)
        self.torque_slider.setValue(255)
        self.torque_slider.setMinimumWidth(150)
        torque_hbox.addWidget(self.torque_slider)
        self.torque_val_lbl = QLabel("255")
        self.torque_val_lbl.setMinimumWidth(30)
        torque_hbox.addWidget(self.torque_val_lbl)
        self.torque_btn = QPushButton("Set Torque")
        self.torque_btn.clicked.connect(
            lambda: (
                self.api_manager.publish_torque(self.torque_slider.value()),
                self.status_updated.emit(
                    "info", f"Torque set to {self.torque_slider.value()}")
            ))
        torque_hbox.addWidget(self.torque_btn)
        torque_hbox.addStretch()
        qv_layout.addLayout(torque_hbox)

        layout.addWidget(quick_set_gb)

        # Tab Widget Section
        tab_widget = QTabWidget()

        # System Info Tab
        sys_info_widget = QWidget()
        sys_info_layout = QVBoxLayout(sys_info_widget)

        conn_group = QGroupBox("Connection Status")
        conn_layout = QVBoxLayout(conn_group)
        self.connection_status = QLabel("Hand API Connected")
        self.connection_status.setObjectName("StatusLabel")
        self.connection_status.setObjectName("StatusInfo")
        conn_layout.addWidget(self.connection_status)

        hand_info_group = QGroupBox("Hand Info")
        hand_info_layout = QVBoxLayout(hand_info_group)
        info_text = f"""Hand Type: {self.hand_type}
Joint Model: {self.hand_joint}
Joint Count: {len(self.hand_config.joint_names)}"""
        self.hand_info_label = QLabel(info_text)
        self.hand_info_label.setWordWrap(True)
        hand_info_layout.addWidget(self.hand_info_label)

        sys_info_layout.addWidget(conn_group)
        sys_info_layout.addWidget(hand_info_group)
        
        # Add matrix heatmap display
        matrix_group = QGroupBox("Finger Matrix Heatmap")
        matrix_layout = QVBoxLayout(matrix_group)
        self.matrix_display = MatrixDisplayWidget()
        matrix_layout.addWidget(self.matrix_display)
        sys_info_layout.addWidget(matrix_group)

        sys_info_layout.addStretch()
        tab_widget.addTab(sys_info_widget, "System Info")

        # Status Log Tab
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        self.status_log = QLabel("Waiting for system startup...")
        self.status_log.setObjectName("StatusLabel")
        self.status_log.setObjectName("StatusInfo")
        self.status_log.setWordWrap(True)
        self.status_log.setMinimumHeight(300)
        log_layout.addWidget(self.status_log)
        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(self.clear_status_log)
        log_layout.addWidget(clear_log_btn)
        tab_widget.addTab(log_widget, "Status Log")

        layout.addWidget(tab_widget)

        # Real-time slider value update
        self.speed_slider.valueChanged.connect(
            lambda v: self.speed_val_lbl.setText(str(v)))
        self.torque_slider.valueChanged.connect(
            lambda v: self.torque_val_lbl.setText(str(v)))
        return panel

    def create_value_display_panel(self):
        """Create slider value display panel"""
        panel = QGroupBox("Joint Value List")
        layout = QVBoxLayout(panel)
        
        # Set layout vertical margins to 20 pixels
        layout.setContentsMargins(10, 20, 10, 20)
        
        self.value_display = QTextEdit()
        self.value_display.setObjectName("ValueDisplay")
        self.value_display.setReadOnly(True)
        self.value_display.setMinimumHeight(60)
        self.value_display.setMaximumHeight(80)
        self.value_display.setText("[]")
        
        layout.addWidget(self.value_display)
        
        return panel

    def update_matrix_display(self, matrix_data):
        """Update Matrix Display"""
        for finger_name, data in matrix_data.items():
            self.matrix_display.update_matrix_data(finger_name, data)

    def on_slider_value_changed(self, index: int, value: int):
        """Slider value changed event handler"""
        if 0 <= index < len(self.slider_labels):
            joint_name = self.hand_config.joint_names[index]
            self.slider_labels[index].setText(f"{joint_name}: {value}")
            
        # Update value display
        self.update_value_display()

    def update_value_display(self):
        """Update value display panel content"""
        values = [slider.value() for slider in self.sliders]
        self.value_display.setText(f"{values}")

    def on_preset_action_clicked(self, positions: List[int]):
        """Preset action button click event handler"""
        if len(positions) != len(self.sliders):
            QMessageBox.warning(
                self, "Action Mismatch", 
                f"Preset action joint count ({len(positions)}) does not match current joint count ({len(self.sliders)})"
            )
            return
            
        # Update sliders
        for i, (slider, pos) in enumerate(zip(self.sliders, positions)):
            slider.setValue(pos)
            self.on_slider_value_changed(i, pos)
            
        # Publish joint state
        self.publish_joint_state()

    def on_home_clicked(self):
        """Return to Home button click event handler"""
        for slider, pos in zip(self.sliders, self.hand_config.init_pos):
            slider.setValue(pos)
            
        self.publish_joint_state()
        self.status_updated.emit("info", "Return to Home")
        
        # Update value display
        self.update_value_display()

    def on_stop_clicked(self):
        """Stop All Actions button click event handler"""
        # Stop cycle timer
        if self.cycle_timer and self.cycle_timer.isActive():
            self.cycle_timer.stop()
            self.cycle_timer = None
            self.cycle_button.setText("Cycle Preset Actions")
            self.reset_preset_buttons_color()
            
        self.status_updated.emit("warning", "All actions stopped")

    def on_cycle_clicked(self):
        """Cycle Preset Actions button click event handler"""
        if not self.hand_config.preset_actions:
            QMessageBox.warning(self, "No Preset Actions", "Current hand model has no preset actions to cycle")
            return
            
        if self.cycle_timer and self.cycle_timer.isActive():
            # Stop cycle
            self.cycle_timer.stop()
            self.cycle_timer = None
            self.cycle_button.setText("Cycle Preset Actions")
            self.reset_preset_buttons_color()
            self.status_updated.emit("info", "Stopped cycling preset actions")
        else:
            # Start cycle
            self.current_action_index = -1
            self.cycle_timer = QTimer(self)
            self.cycle_timer.timeout.connect(self.run_next_action)
            self.cycle_timer.start(LOOP_TIME)
            self.cycle_button.setText("Stop Cycling")
            self.status_updated.emit("info", "Started cycling preset actions")
            self.run_next_action()

    def run_next_action(self):
        """Run next preset action"""
        if not self.hand_config.preset_actions:
            return
            
        # Reset all button colors
        self.reset_preset_buttons_color()
        
        # Calculate next action index
        self.current_action_index = (self.current_action_index + 1) % len(self.hand_config.preset_actions)
        
        # Get next action
        action_names = list(self.hand_config.preset_actions.keys())
        action_name = action_names[self.current_action_index]
        action_positions = self.hand_config.preset_actions[action_name]
        
        # Execute action
        self.on_preset_action_clicked(action_positions)
        
        # Highlight current action button
        if 0 <= self.current_action_index < len(self.preset_buttons):
            button = self.preset_buttons[self.current_action_index]
            button.setStyleSheet("background-color: green; color: white; border-color: #91D5FF;")
            
        self.status_updated.emit("info", f"Running preset action: {action_name}")

    def reset_preset_buttons_color(self):
        """Reset all preset button colors"""
        for button in self.preset_buttons:
            button.setStyleSheet("")
            button.setProperty("category", "preset")
            button.style().unpolish(button)
            button.style().polish(button)

    def publish_joint_state(self):
        """Publish current joint state"""
        positions = [slider.value() for slider in self.sliders]
        self.api_manager.publish_joint_state(positions)

    def update_status(self, status_type: str, message: str):
        """Update status display"""
        # Update connection status
        if status_type == "info" and "Hand API initialized successfully" in message:
            self.connection_status.setText("Hand API Connected")
            self.connection_status.setObjectName("StatusLabel")
            self.connection_status.setObjectName("StatusInfo")
            
        # Update log
        current_time = time.strftime("%H:%M:%S")
        log_entry = f"[{current_time}] {message}\n"
        current_log = self.status_log.text()
        
        if len(current_log) > 10000:
            current_log = current_log[-10000:]
            
        self.status_log.setText(log_entry + current_log)
        
        # Set log style
        self.status_log.setObjectName("StatusLabel")
        if status_type == "error":
            self.status_log.setObjectName("StatusError")
        else:
            self.status_log.setObjectName("StatusInfo")

    def clear_status_log(self):
        """Clear status log"""
        self.status_log.setText("Log cleared")
        self.status_log.setObjectName("StatusLabel")
        self.status_log.setObjectName("StatusInfo")

    def closeEvent(self, event):
        """Window close event handler"""
        if self.cycle_timer and self.cycle_timer.isActive():
            self.cycle_timer.stop()
        if self.publish_timer and self.publish_timer.isActive():
            self.publish_timer.stop()
        self.api_manager.shutdown()
        super().closeEvent(event)

def main():
    """Main function"""
    try:
        # Create Qt application
        app = QApplication(sys.argv)
        
        # Create API manager
        api_manager = HandApiManager()
        
        # Create GUI
        window = HandControlGUI(api_manager)
        
        # Connect status update signal
        api_manager.status_updated.connect(window.update_status)
        window.status_updated = api_manager.status_updated
        
        # Show window
        window.show()
        
        # Run application
        exit_code = app.exec_()
        
        # Cleanup
        api_manager.shutdown()
            
        sys.exit(exit_code)
    except Exception as e:
        print(f"Application startup failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
