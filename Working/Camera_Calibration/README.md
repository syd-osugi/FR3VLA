# Camera Calibration

Tools for calibrating the Franka FR3VLA vision system cameras. Supports intrinsic calibration (lens distortion + focal length) and extrinsic calibration (camera-to-robot transforms) for three cameras:

| Camera | Mount | Role | Output |
|--------|-------|------|--------|
| **D405** | End-effector (wrist) | Eye-in-hand, follows the robot | `calibration_data/d405_to_ee.json` |
| **D435 (left)** | Table/ceiling, 45° left of center | Eye-to-hand, fixed overhead | `calibration_data/d435_left_to_base.json` |
| **D435 (right)** | Table/ceiling, 45° right of center | Eye-to-hand, fixed overhead | `calibration_data/d435_right_to_base.json` |

## Prerequisites

- Franka FR3 robot connected and controllable via `pylibfranka`
- Intel RealSense D405 and two D435 cameras connected via USB 3.0
- `pyrealsense2`, `opencv-contrib-python`, `pyyaml` installed
- ChArUco board printed with the geometry specified in `calibration_config.yaml`

## Quick Start

### 1. Configure Cameras

Edit `calibration_config.yaml` and set the serial numbers:

```yaml
d435_left_serial: "YOUR_LEFT_D435_SERIAL"
d435_right_serial: "YOUR_RIGHT_D435_SERIAL"
d405_serial: "YOUR_D405_SERIAL"
```

### 2. Intrinsic Calibration (per camera)

Calibrate each camera's lens distortion independently. Run the script for each camera, waving the ChArUco board in front of it.

```bash
# Calibrate D405 (eye-in-hand)
python calibrate_intrinsics.py --camera d405

# Calibrate left D435
python calibrate_intrinsics.py --camera d435_left

# Calibrate right D435
python calibrate_intrinsics.py --camera d435_right
```

Each camera saves to its own file:
- D405 -> `calibration_data/d405_intrinsics.json`
- D435 -> `calibration_data/d435_intrinsics.json`

### 3. Extrinsic Calibration

#### D405 Eye-in-Hand

Mount the ChArUco board flat on the table. Move the robot so the wrist-mounted D405 sees the board from many viewpoints.

```bash
# Manual mode: move robot by hand or command, press 's' at each good pose
python calibrate_d405_hand_eye.py

# Planned-pose mode: use saved poses from pose_planner
python calibrate_d405_hand_eye.py --pose-plan /path/to/pose_plan.json
```

The script saves `calibration_data/d405_to_ee.json` containing T_d405_to_ee.

#### D435 Eye-to-Hand (Fixed Cameras)

Mount the ChArUco board rigidly to the end-effector. Move the robot so the fixed D435 sees the board from many viewpoints. Calibrate each camera separately.

```bash
# Calibrate left D435
python calibrate_d435_bird_eye.py --camera left

# Calibrate right D435
python calibrate_d435_bird_eye.py --camera right
```

Each camera saves to its own file:
- Left -> `calibration_data/d435_left_to_base.json`
- Right -> `calibration_data/d435_right_to_base.json`

### 4. Pose Planner (Optional)

For D405 calibration, you can pre-plan robot poses before running calibration:

```bash
python pose_planner.py --camera d405
```

This opens a live D405 preview window. Move the robot to viewpoints where the ChArUco board is clearly visible, then press `s` to save each pose. After collecting poses, run calibration with `--pose-plan`.

### 5. State Recorder (Pose Planner)

For D435 calibration (and any calibration requiring varied robot poses), use the state recorder to save and recall robot positions:

```bash
# Record states for left D435 calibration
python state_recorder.py --camera left

# Record states for right D435 calibration
python state_recorder.py --camera right

# Record states for D405 calibration
python state_recorder.py --camera d405
```

See [State Recorder](#state-recorder) below for details.

## State Recorder

The state recorder lets you save Franka robot joint states and end-effector poses during calibration, then recall them later to return to known-good viewpoints. This is essential for extrinsic calibration where you need to revisit specific poses.

### How It Works

1. Connect to the robot and start a live camera preview
2. Move the robot to a good viewpoint (visible in the camera preview)
3. Save the current state with a label (e.g., "pose_1", "pose_2")
4. Repeat until you have enough poses
5. Recall any saved pose to return to that viewpoint
6. Export the saved poses to a JSON file for use with `calibrate_d405_hand_eye.py --pose-plan`

### Camera Preview

The state recorder opens a live camera preview window so you can verify each saved pose shows the ChArUco board clearly. The preview updates in real-time as you move the robot.

### Keyboard Controls

| Key | Action |
|-----|--------|
| `s` | Save current robot state (prompts for label) |
| `r` | Recall a saved pose (prompts for index) |
| `d` | Delete a saved pose (prompts for index) |
| `e` | Export saved poses to JSON file |
| `q` | Quit and save the pose list |

### Per-Camera Output

Each camera has its own pose file to keep calibration data organized:

| Camera | Pose File |
|--------|-----------|
| D405 | `calibration_data/d405_poses.json` |
| D435 (left) | `calibration_data/d435_left_poses.json` |
| D435 (right) | `calibration_data/d435_right_poses.json` |

### Pose File Format

```json
{
  "camera": "d405",
  "saved_at": "2026-07-16T23:00:00",
  "poses": [
    {
      "label": "pose_1",
      "timestamp": 1234567890.0,
      "q": [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5],
      "O_T_EE": [[...], [...], [...], [...]],
      "ee_xyz": [0.3, 0.1, 0.5]
    }
  ]
}
```

### Using Saved Poses with Calibration

After exporting poses, run the D405 calibration script with the `--pose-plan` flag:

```bash
python calibrate_d405_hand_eye.py --pose-plan calibration_data/d405_poses.json
```

The calibration script will move the robot to each saved pose and prompt you to capture when the ChArUco board is visible.

## Configuration

All parameters live in `calibration_config.yaml`:

| Section | Key Parameters |
|---------|---------------|
| `d435_left_serial` / `d435_right_serial` / `d405_serial` | Camera serial numbers |
| `charuco_board_inner_corners` | ChArUco board grid size |
| `charuco_square_size_m` / `charuco_marker_size_m` | Board geometry |
| `hand_eye_d405.*` | D405 calibration settings |
| `bird_eye_d435.left.*` / `bird_eye_d435.right.*` | D435 calibration settings |
| `franka_ip` | Robot IP address |
| `workspace_min_m` / `workspace_max_m` | Safe workspace bounds |

## Troubleshooting

- **Camera not found**: Verify serial numbers in `calibration_config.yaml` match the connected cameras. Use `pyrealsense2` to list devices:
  ```python
  import pyrealsense2 as rs
  ctx = rs.context()
  for d in ctx.query_devices():
      print(d.get_info(rs.camera_info.serial_number))
  ```
- **ChArUco not detected**: Ensure the board is printed at the correct size (check `charuco_square_size_m` and `charuco_marker_size_m`). Verify the board is flat and well-lit.
- **Hand-eye validation fails**: Capture more poses with greater rotational diversity. Avoid poses that only translate in a straight line.
- **Robot connection fails**: Verify the Franka IP in `calibration_config.yaml` matches the robot controller. Ensure the robot is in a controllable mode.

## File Structure

```
Working/Camera_Calibration/
├── calibration_config.yaml          # All calibration parameters
├── calibration_math.py              # Shared math (ChArUco, transforms, validation)
├── calibrate_intrinsics.py          # Intrinsic calibration (per camera)
├── calibrate_d405_hand_eye.py       # D405 eye-in-hand extrinsic calibration
├── calibrate_d435_bird_eye.py       # D435 eye-to-hand extrinsic calibration
├── state_recorder.py                # Save/recall robot states during calibration
├── pose_planner.py                  # Interactive pose planner for D405 calibration
├── calibration_data/                # Output directory for all calibration files
├── __init__.py
├── package.xml                      # ROS package manifest
└── CMakeLists.txt                   # ROS build config
```
