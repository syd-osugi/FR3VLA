
###########################
# # RealHand Grip & Log Test
# Measures pressure state and joint state over time
# Joints are currently set to start in a fully open position, then close, and then open once a pressure threshold of any finger reaches 10
###########################

import time, os, sys, csv
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(target_dir)

from RealHand.real_hand_api import RealHandApi
from RealHand.utils.load_write_yaml import LoadWriteYaml

# --- Initialization ---
configyaml = LoadWriteYaml()
setting = configyaml.load_setting_yaml()

left_hand = False
right_hand = False
if setting['REAL_HAND']['LEFT_HAND']['EXISTS'] == True:
    left_hand = True
elif setting['REAL_HAND']['RIGHT_HAND']['EXISTS'] == True:
    right_hand = True

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

# Set speed and torque
speed = [10, 10, 10, 10, 10, 10]
real_hand.set_speed(speed)

torque = [10, 10, 10, 10, 10, 10]
real_hand.set_torque(torque)

# --- Helper Function to get Max Pressure ---
def get_max_pressure(rh_api):
    """Reads all finger matrices and returns the absolute maximum pressure value."""
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
            pass # Handle empty arrays if no sensors are triggered
    return max_p

# --- Execution Logic ---
log_data = []
start_time = time.time()

print("Sending CLOSE command...")
position_closed = [0, 0, 0, 0, 0, 0] # Assuming 0 is fully closed
real_hand.finger_move(position_closed)

print("Monitoring pressure until max reaches 10...")
while True:
    current_time = time.time() - start_time
    
    # Get states
    max_pressure = get_max_pressure(real_hand)
    joint_state = real_hand.get_state()
    
    # Log data (flattening joint state into j1 through j6 for CSV columns)
    log_data.append({
        'time': round(current_time, 3),
        'max_pressure': max_pressure,
        'j1': joint_state[0], 'j2': joint_state[1], 'j3': joint_state[2],
        'j4': joint_state[3], 'j5': joint_state[4], 'j6': joint_state[5]
    })
    
    print(f"Time: {current_time:.2f}s | Max Pressure: {max_pressure} | Joints: {joint_state}")
    
    # Check threshold
    if max_pressure >= 10:
        print("\nThreshold of 10 reached!")
        break
        
    time.sleep(.5) # 20Hz polling rate to prevent CAN bus overload

print("Sending OPEN command...")
position_openhand = [250, 250, 250, 250, 250, 250]
real_hand.finger_move(position_openhand)

# Allow hand to open, continue logging briefly so the plot shows the release
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

# --- Save Data to CSV ---
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
csv_filename = f"realhand_log_{timestamp_str}.csv"
csv_filepath = os.path.join(PressureState_JointState_Time_Plot_Test3_Output, csv_filename)

with open(csv_filepath, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['time', 'max_pressure', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6'])
    writer.writeheader()
    writer.writerows(log_data)
print(f"Data saved to {csv_filepath}")

# --- Plot Data and SAVE IMAGE ---
try:
    import matplotlib.pyplot as plt
    
    # --- Define the custom L6 joint names ---
    l6_joint_names = [
        "Thumb Flexion", 
        "Thumb Adduction/Abduction",
        "Index Finger Flexion", 
        "Middle Finger Flexion", 
        "Ring Finger Flexion",
        "Pinky Finger Flexion"
    ]
    # --------------------------------------------
    
    times = [d['time'] for d in log_data]
    pressures = [d['max_pressure'] for d in log_data]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot Pressure
    ax1.plot(times, pressures, color='red', linewidth=2, label='Max Pressure')
    ax1.axhline(y=10, color='grey', linestyle='--', label='Threshold (10)')
    ax1.set_ylabel('Pressure Value')
    ax1.set_title('Max Pressure Over Time')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Joints using the custom names
    for i, name in enumerate(l6_joint_names):
        joint_vals = [d[f'j{i+1}'] for d in log_data]
        ax2.plot(times, joint_vals, linewidth=2, label=name)
        
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Joint Position (0-250)')
    ax2.set_title('Joint States Over Time')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0)) # Adjusted legend position so long names don't get cut off
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save the image
    image_filename = f"realhand_plot_{timestamp_str}.png"
    csv_filepath = os.path.join(PressureState_JointState_Time_Plot_Test3_Output, csv_filename)
    plt.savefig(image_filepath, dpi=150, bbox_inches='tight')
    print(f"Plot saved as image to {image_filepath}")
    
    plt.show()

except ImportError:
    print("\n[WARNING] matplotlib is not installed. Skipping plot.")
    print("Install it using: pip install matplotlib")
    print("You can still view the raw data in the generated CSV file.")