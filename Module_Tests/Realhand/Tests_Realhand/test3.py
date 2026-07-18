#!/usr/bin/env python3

"""
RealHand Grip & Log Test (Test 3) - Time-Series Data Collection

Records pressure sensor readings and joint positions over time during a full
grasp cycle, then saves the data to CSV and generates a matplotlib plot.

Procedure:
  1. Start with hand fully open (all joints at position 250)
  2. Close all fingers simultaneously (all joints to position 0)
  3. Poll max pressure across all fingers and all 6 joint positions at ~2 Hz
  4. Stop logging when any finger's max pressure reaches threshold of 10
  5. Open hand and log 2 more seconds of release data
  6. Save time-series data to CSV in Module_Tests/Test_Outputs/PressureState_JointState_Time_Plot_Test3/
  7. Generate dual-panel plot: pressure over time (top) + all joint states over time (bottom)

Output files (saved to Test_Outputs/<script_name>/):
  - realhand_log_<timestamp>.csv: time, max_pressure, j1-j6 columns
  - realhand_plot_<timestamp>.png: dual-panel visualization at 150 DPI

Joint mapping (L6 hand model):
  j1 = Thumb Flexion
  j2 = Thumb Adduction/Abduction
  j3 = Index Finger Flexion
  j4 = Middle Finger Flexion
  j5 = Ring Finger Flexion
  j6 = Pinky Finger Flexion

Pressure threshold: 10 (absolute maximum across all finger sensor grids)
"""

import time, os, sys, csv
from datetime import datetime
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(target_dir)

# Determine output directory: Module_Tests/Test_Outputs/<script_name>/
SCRIPT_NAME = Path(__file__).stem
MODULE_TESTS_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = MODULE_TESTS_ROOT / "Test_Outputs" / SCRIPT_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from RealHand.real_hand_api import RealHandApi
from RealHand.utils.load_write_yaml import LoadWriteYaml

# =========================================================
# Configuration: detect hand and load settings
# =========================================================
configyaml = LoadWriteYaml()
setting = configyaml.load_setting_yaml()

left_hand = False
right_hand = False
if setting['REAL_HAND']['LEFT_HAND']['EXISTS'] == True:
    left_hand = True
elif setting['REAL_HAND']['RIGHT_HAND']['EXISTS'] == True:
    right_hand = True

# Prefer left hand if both are configured (SDK is single-hand only)
if left_hand == True and right_hand == True:
    left_hand = True
    right_hand = False

if left_hand == True:
    hand_exists = True
    hand_joint = setting['REAL_HAND']['LEFT_HAND']['JOINT']
    hand_type = "left"
    is_touch = setting['REAL_HAND']['LEFT_HAND']['TOUCH']
    can = setting['REAL_HAND']['LEFT_HAND']['CAN']
    modbus = setting['REAL_HAND']['LEFT_HAND']['MODBUS']
if right_hand == True:
    hand_exists = True
    hand_joint = setting['REAL_HAND']['RIGHT_HAND']['JOINT']
    hand_type = "right"
    is_touch = setting['REAL_HAND']['RIGHT_HAND']['TOUCH']
    can = setting['REAL_HAND']['RIGHT_HAND']['CAN']
    modbus = setting['REAL_HAND']['RIGHT_HAND']['MODBUS']

real_hand = RealHandApi(hand_joint=hand_joint, hand_type=hand_type, modbus=modbus, can=can)

# Configure speed and torque parameters
speed = [10, 10, 10, 10, 10, 10]
real_hand.set_speed(speed)

torque = [10, 10, 10, 10, 10, 10]
real_hand.set_torque(torque)

# =========================================================
# Helper: compute absolute maximum pressure across all fingers
# =========================================================
def get_max_pressure(rh_api):
    """Read pressure matrices from all five fingers and return the global maximum.

    Each finger's get_*_matrix_touch() returns a 2D array of sensor readings.
    This function flattens each matrix and finds the single highest value across
    all fingers, providing a scalar contact indicator.

    Args:
        rh_api: RealHandApi instance connected to the hand.

    Returns:
        int: The maximum pressure value found across all finger sensor grids.
             Returns 0 if no sensors report contact or if any read fails.
    """
    pressures = [
        rh_api.get_thumb_matrix_touch(),
        rh_api.get_index_matrix_touch(),
        rh_api.get_middle_matrix_touch(),
        rh_api.get_ring_matrix_touch(),
        rh_api.get_little_matrix_touch()
    ]

    max_p = 0
    for p in pressures:
        # Flatten the 2D matrix and find the max value
        try:
            current_max = max(item for row in p for item in row)
            if current_max > max_p:
                max_p = current_max
        except:
            pass  # Handle empty arrays if no sensors are triggered on a finger
    return max_p

# =========================================================
# Phase 1: Close hand and log time-series data
# =========================================================
log_data = []
start_time = time.time()

print("Sending CLOSE command...")
# Close all fingers simultaneously (position 0 = fully closed)
position_closed = [0, 0, 0, 0, 0, 0]
real_hand.finger_move(position_closed)

print("Monitoring pressure until max reaches 10...")
while True:
    current_time = time.time() - start_time

    # Read current max pressure across all fingers
    max_pressure = get_max_pressure(real_hand)
    # Read current joint positions (6 values)
    joint_state = real_hand.get_state()

    # Append timestamped readings to the log buffer
    # Joint positions are flattened into individual columns for CSV export
    log_data.append({
        'time': round(current_time, 3),
        'max_pressure': max_pressure,
        'j1': joint_state[0], 'j2': joint_state[1], 'j3': joint_state[2],
        'j4': joint_state[3], 'j5': joint_state[4], 'j6': joint_state[5]
    })

    print(f"Time: {current_time:.2f}s | Max Pressure: {max_pressure} | Joints: {joint_state}")

    # Stop when any finger detects contact (pressure threshold reached)
    if max_pressure >= 10:
        print("\nThreshold of 10 reached!")
        break

    time.sleep(0.5)  # ~2 Hz polling rate to avoid CAN bus overload

# =========================================================
# Phase 2: Open hand and log release data
# =========================================================
print("Sending OPEN command...")
position_openhand = [250, 250, 250, 250, 250, 250]
real_hand.finger_move(position_openhand)

# Continue logging for 2 more seconds to capture the release trajectory
# This ensures the plot shows the pressure dropping to zero after opening
time.sleep(2.0)
current_time = time.time() - start_time
max_pressure = get_max_pressure(real_hand)
joint_state = real_hand.get_state()
log_data.append({
    'time': round(current_time, 3),
    'max_pressure': max_pressure,
    'j1': joint_state[0], 'j2': joint_state[1], 'j3': joint_state[2],
    'j4': joint_state[3], 'j5': joint_state[4], 'j6': joint_state[5]
})
print("Hand opened.")

# =========================================================
# Phase 3: Save collected data to CSV
# =========================================================
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
csv_filename = f"realhand_log_{timestamp_str}.csv"
csv_filepath = OUTPUT_DIR / csv_filename

with open(csv_filepath, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['time', 'max_pressure', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6'])
    writer.writeheader()
    writer.writerows(log_data)
print(f"Data saved to {csv_filepath}")

# =========================================================
# Phase 4: Generate dual-panel plot (pressure + joint states)
# =========================================================
try:
    import matplotlib.pyplot as plt

    # Anatomical joint names for the L6 hand model
    l6_joint_names = [
        "Thumb Flexion",
        "Thumb Adduction/Abduction",
        "Index Finger Flexion",
        "Middle Finger Flexion",
        "Ring Finger Flexion",
        "Pinky Finger Flexion"
    ]

    times = [d['time'] for d in log_data]
    pressures = [d['max_pressure'] for d in log_data]

    # Create two vertically stacked subplots sharing the x-axis (time)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top panel: max pressure over time with threshold line
    ax1.plot(times, pressures, color='red', linewidth=2, label='Max Pressure')
    ax1.axhline(y=10, color='grey', linestyle='--', label='Threshold (10)')
    ax1.set_ylabel('Pressure Value')
    ax1.set_title('Max Pressure Over Time')
    ax1.legend()
    ax1.grid(True)

    # Bottom panel: all 6 joint positions over time with anatomical labels
    for i, name in enumerate(l6_joint_names):
        joint_vals = [d[f'j{i+1}'] for d in log_data]
        ax2.plot(times, joint_vals, linewidth=2, label=name)

    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Joint Position (0-250)')
    ax2.set_title('Joint States Over Time')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    ax2.grid(True)

    plt.tight_layout()

    # Save plot as PNG image
    image_filename = f"realhand_plot_{timestamp_str}.png"
    image_filepath = OUTPUT_DIR / image_filename
    plt.savefig(image_filepath, dpi=150, bbox_inches='tight')
    print(f"Plot saved as image to {image_filepath}")

    plt.show()

except ImportError:
    print("\n[WARNING] matplotlib is not installed. Skipping plot.")
    print("Install it using: pip install matplotlib")
    print("You can still view the raw data in the generated CSV file.")
