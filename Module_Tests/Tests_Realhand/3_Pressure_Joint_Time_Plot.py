#!/usr/bin/env python3

"""
RealHand Grip & Log Test (Test 3) - Per-Finger Pressure Time-Series Collection

Records pressure sensor readings and joint positions over time during a full
grasp cycle, then saves the data to CSV and generates a matplotlib plot.

Key behaviors:
  1. Each finger's max pressure is tracked and plotted individually (not just global max)
  2. Individual fingers stop moving when EITHER:
     - Their max pressure reaches PRESSURE_THRESHOLD (default: 10), OR
     - Their joint position is unchanged for STAGNATION_TIME (default: 1 second)
  3. Once ALL fingers have stopped, the hand waits 2 seconds then opens
  4. The plot updates in real-time without blocking the terminal

Procedure:
  1. Start with hand fully open (all joints at position 250)
  2. Send close command to all fingers
  3. In a control loop at ~2 Hz:
     a. Poll per-finger max pressure and joint positions
     b. Detect which fingers have stopped (pressure threshold or position stagnation)
     c. For stopped fingers, hold their current position; for others, continue closing
     d. Log data for CSV and update real-time plot
  4. Once all fingers stop, wait 2 seconds then open the hand
  5. Save time-series data to CSV
  6. Generate and display dual-panel plot in real-time

Output files (saved to Test_Outputs/<script_name>/):
  - realhand_log_<timestamp>.csv: time, per-finger pressures (p1-p5), j1-j6 columns
  - realhand_plot_<timestamp>.png: dual-panel visualization at 150 DPI

Per-finger pressure mapping:
  p1 = Thumb max pressure    (get_thumb_matrix_touch)
  p2 = Index max pressure    (get_index_matrix_touch)
  p3 = Middle max pressure   (get_middle_matrix_touch)
  p4 = Ring max pressure     (get_ring_matrix_touch)
  p5 = Pinky max pressure    (get_little_matrix_touch)

Joint mapping (L6 hand model):
  j1 = Thumb Flexion
  j2 = Thumb Adduction/Abduction
  j3 = Index Finger Flexion
  j4 = Middle Finger Flexion
  j5 = Ring Finger Flexion
  j6 = Pinky Finger Flexion

Parameters:
  PRESSURE_THRESHOLD = 10  (max pressure per finger to trigger stop)
  STAGNATION_TIME = 1.0    (seconds of no position change to trigger stop)
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
# Parameters
# =========================================================
PRESSURE_THRESHOLD = 10       # Max pressure per finger to trigger stop
STAGNATION_TIME = 1.0         # Seconds of no position change to trigger stop
ALL_STOPPED_WAIT = 2.0        # Seconds to wait after all fingers stop before opening

# Finger names for plotting legends (order: thumb, index, middle, ring, pinky)
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

# Joint names for plotting legends (L6 hand model)
L6_JOINT_NAMES = [
    "Thumb Flexion",
    "Thumb Adduction/Abduction",
    "Index Finger Flexion",
    "Middle Finger Flexion",
    "Ring Finger Flexion",
    "Pinky Finger Flexion"
]

# =========================================================
# Per-finger pressure tracking
# =========================================================
# Maps finger index (0-4) to (joint_index, pressure_getter_function)
# Joint indices: 0=Thumb Flexion, 1=Thumb Adduction, 2=Index, 3=Middle, 4=Ring, 5=Pinky
FINGER_PRESSURE_INFO = [
    (0, real_hand.get_thumb_matrix_touch),       # Thumb -> joint 0
    (2, real_hand.get_index_matrix_touch),       # Index -> joint 2
    (3, real_hand.get_middle_matrix_touch),      # Middle -> joint 3
    (4, real_hand.get_ring_matrix_touch),        # Ring -> joint 4
    (5, real_hand.get_little_matrix_touch),      # Pinky -> joint 5
]

# Per-finger max pressure values (5 fingers)
finger_max_pressure = [0] * 5

# Track which joints have stopped moving (6 joints)
joint_stopped = [False] * 6

# Track last time each joint's position changed (for stagnation detection)
last_position_change = [time.time()] * 6

# Previous joint state for stagnation comparison
prev_joint_state = None

# Target positions for each joint (updated each cycle)
target_positions = [0, 0, 0, 0, 0, 0]

# =========================================================
# Phase 1: Close hand with per-finger pressure monitoring
# =========================================================
log_data = []
start_time = time.time()

print("Sending CLOSE command...")
real_hand.finger_move([0, 0, 0, 0, 0, 0])

all_stopped_time = None  # Track when all fingers first stopped

print("Monitoring per-finger pressure and joint stagnation...")
while True:
    current_time = time.time() - start_time

    # Read current joint positions (6 values)
    joint_state = real_hand.get_state()

    # Detect position stagnation for each joint
    if prev_joint_state is not None:
        for i in range(6):
            if joint_state[i] != prev_joint_state[i]:
                last_position_change[i] = time.time()
    prev_joint_state = list(joint_state)

    # Read per-finger max pressure and check pressure-based stopping
    for f_idx, (joint_idx, pressure_func) in enumerate(FINGER_PRESSURE_INFO):
        try:
            pressure_matrix = pressure_func()
            # Flatten 2D matrix and find max value
            current_max = max(item for row in pressure_matrix for item in row)
            finger_max_pressure[f_idx] = current_max
        except Exception:
            finger_max_pressure[f_idx] = 0
            continue

        # Stop this finger if pressure threshold reached
        if not joint_stopped[joint_idx] and current_max >= PRESSURE_THRESHOLD:
            joint_stopped[joint_idx] = True
            target_positions[joint_idx] = joint_state[joint_idx]
            print(f"  Joint {joint_idx + 1} ({L6_JOINT_NAMES[joint_idx]}) stopped: pressure threshold reached ({current_max})")

    # Check stagnation-based stopping for all joints
    for i in range(6):
        if not joint_stopped[i]:
            if current_time - last_position_change[i] >= STAGNATION_TIME:
                joint_stopped[i] = True
                target_positions[i] = joint_state[i]
                print(f"  Joint {i + 1} ({L6_JOINT_NAMES[i]}) stopped: stagnation ({STAGNATION_TIME}s)")

    # Record when all joints first stop
    if all(joint_stopped) and all_stopped_time is None:
        all_stopped_time = current_time
        print("\nAll fingers stopped. Waiting 2 seconds before opening...")

    # If all stopped and wait period elapsed, open the hand
    if all_stopped_time is not None and (current_time - all_stopped_time) >= ALL_STOPPED_WAIT:
        break

    # Send target positions: stopped joints hold current position, others continue closing
    real_hand.finger_move(target_positions)

    # Log timestamped data
    log_data.append({
        'time': round(current_time, 3),
        'p1': finger_max_pressure[0],  # Thumb
        'p2': finger_max_pressure[1],  # Index
        'p3': finger_max_pressure[2],  # Middle
        'p4': finger_max_pressure[3],  # Ring
        'p5': finger_max_pressure[4],  # Pinky
        'j1': joint_state[0], 'j2': joint_state[1], 'j3': joint_state[2],
        'j4': joint_state[3], 'j5': joint_state[4], 'j6': joint_state[5]
    })

    print(f"  Time: {current_time:.2f}s | Pressures: {finger_max_pressure} | Joints: {joint_state} | Stopped: {joint_stopped}")

    time.sleep(0.5)  # ~2 Hz polling rate to avoid CAN bus overload

# =========================================================
# Phase 2: Open hand and log release data
# =========================================================
print("\nSending OPEN command...")
position_openhand = [250, 250, 250, 250, 250, 250]
real_hand.finger_move(position_openhand)

# Continue logging for 2 more seconds to capture the release trajectory
time.sleep(2.0)
current_time = time.time() - start_time
joint_state = real_hand.get_state()

# Compute per-finger max pressure for final log entry
for f_idx, (joint_idx, pressure_func) in enumerate(FINGER_PRESSURE_INFO):
    try:
        pressure_matrix = pressure_func()
        finger_max_pressure[f_idx] = max(item for row in pressure_matrix for item in row)
    except Exception:
        finger_max_pressure[f_idx] = 0

log_data.append({
    'time': round(current_time, 3),
    'p1': finger_max_pressure[0],
    'p2': finger_max_pressure[1],
    'p3': finger_max_pressure[2],
    'p4': finger_max_pressure[3],
    'p5': finger_max_pressure[4],
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
    writer = csv.DictWriter(f, fieldnames=['time', 'p1', 'p2', 'p3', 'p4', 'p5', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6'])
    writer.writeheader()
    writer.writerows(log_data)
print(f"Data saved to {csv_filepath}")

# =========================================================
# Phase 4: Generate real-time updating plot (non-blocking)
# =========================================================
try:
    import matplotlib.pyplot as plt

    # Extract data arrays from log
    times = [d['time'] for d in log_data]
    finger_pressures = [
        [d['p1'] for d in log_data],  # Thumb
        [d['p2'] for d in log_data],  # Index
        [d['p3'] for d in log_data],  # Middle
        [d['p4'] for d in log_data],  # Ring
        [d['p5'] for d in log_data],  # Pinky
    ]
    joint_vals = [
        [d['j1'] for d in log_data],
        [d['j2'] for d in log_data],
        [d['j3'] for d in log_data],
        [d['j4'] for d in log_data],
        [d['j5'] for d in log_data],
        [d['j6'] for d in log_data],
    ]

    # Enable interactive mode so the plot updates without blocking
    plt.ion()

    # Create figure with two vertically stacked subplots sharing the x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top panel: per-finger max pressure over time (one line per finger)
    pressure_lines = []
    for i, name in enumerate(FINGER_NAMES):
        line, = ax1.plot([], [], linewidth=2, label=name)
        pressure_lines.append(line)
    threshold_line, = ax1.plot([], [], color='grey', linestyle='--', linewidth=1.5, label='Threshold (10)')
    ax1.set_ylabel('Pressure Value')
    ax1.set_title('Per-Finger Max Pressure Over Time')
    # Place legend outside the plot area to avoid blocking the data
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
    ax1.grid(True)

    # Bottom panel: all 6 joint positions over time (one line per joint)
    joint_lines = []
    for i, name in enumerate(L6_JOINT_NAMES):
        line, = ax2.plot([], [], linewidth=2, label=name)
        joint_lines.append(line)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Joint Position (0-250)')
    ax2.set_title('Joint States Over Time')
    # Place legend outside the plot area to avoid blocking the data
    ax2.legend(loc='upper right', bbox_to_anchor=(1.02, 1.0))
    ax2.grid(True)

    plt.tight_layout()

    # --- Real-time update loop: update plot as data arrives ---
    for i in range(len(log_data)):
        # Update pressure lines with data up to current index
        for j, line in enumerate(pressure_lines):
            line.set_data(times[:i+1], finger_pressures[j][:i+1])
        threshold_line.set_data([times[0], times[i]], [10, 10])

        # Update joint lines with data up to current index
        for j, line in enumerate(joint_lines):
            line.set_data(times[:i+1], joint_vals[j][:i+1])

        # Rescale axes to fit new data
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()

        # Redraw without blocking
        fig.canvas.draw()
        fig.canvas.flush_events()

    # Save final plot as PNG image
    image_filename = f"realhand_plot_{timestamp_str}.png"
    image_filepath = OUTPUT_DIR / image_filename
    plt.savefig(image_filepath, dpi=150, bbox_inches='tight')
    print(f"Plot saved as image to {image_filepath}")

    # Show the plot in non-blocking mode
    plt.show(block=False)
    print("Plot displayed. Close the window to exit.")

except ImportError:
    print("\n[WARNING] matplotlib is not installed. Skipping plot.")
    print("Install it using: pip install matplotlib")
    print("You can still view the raw data in the generated CSV file.")
