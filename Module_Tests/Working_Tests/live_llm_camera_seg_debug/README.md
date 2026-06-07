# Live LLM / Camera / Segmentation Debug

Use this diagnostic to test the real D435/D405 camera hardware and the configured local LLM server without connecting to the Franka robot and without commanding any robot motion.

For a `main.py`-style interactive loop, run:

```bash
python3 Module_Tests/Working_Tests/live_llm_camera_seg_debug/interactive_no_robot_llm_debug.py
```

Then type commands at `Enter command:`. The LLM decides what to inspect from your command, requests real camera images, localizes object pixels with the normal XYZ tools, fuses camera estimates when useful, and can plan a trajectory without executing it.

Interactive outputs go to:

```text
Module_Tests/Test_Outputs/interactive_no_robot_llm_debug/<timestamp>/
```

Each command gets its own folder, for example `command_001/`. After the LLM finishes, the runner saves:

- `side_by_side_marker_trajectory.png`: D435/D405 side-by-side image with localization markers and projected planned waypoints when projection is possible.
- `side_by_side_marker_trajectory_summary.json`: recorded tool results, latest localization data, trajectory JSON, and projected display points.
- `conversation_sanitized.json`: LLM conversation with image data URLs removed.

The interactive script uses the normal `Working` LLM interface and tool dispatcher. It exposes everything from `vision.tools.tool_json_list` except physical execution:

- `get_birds_eye_view`
- `get_eye_in_hand_view`
- `get_xyz_d435`
- `get_xyz_d405`
- `get_xyz_fused`
- `plan_robot_trajectory`

`execute_robot_waypoints` is intentionally removed. There is no Franka connection, no live robot state read, and no motion.

D405 robot-frame localization and trajectory math need `robot_ee_pose`. Because this runtime cannot read Franka state, it supplies a static debug pose:

- default: first pose from `Working/camera_calibration/calibration_data/d405_hand_eye_pose_plan.json`, if present;
- fallback: identity pose;
- override: `--ee-pose-file /path/to/pose.json`;
- disable static pose: `--ee-pose-source none`.

That means D435 calibration math, D405 hand-eye math, fusion, and trajectory planning can run, but anything using D405/base-frame math is only as accurate as the static pose you choose.

For a single-shot diagnostic with a command-line target, run:

```bash
python3 Module_Tests/Working_Tests/live_llm_camera_seg_debug/live_llm_camera_seg_debug.py \
  --target "red block" \
  --target-color red
```

Single-shot outputs go to:

```text
Module_Tests/Test_Outputs/live_llm_camera_seg_debug/<timestamp>/
```

Useful files in each run:

- `summary.json`: LLM pixels, segmentation pixels, pixel deltas, depth stats, and camera-frame XYZ results.
- `llm_request.json` / `llm_response.json` / `llm_reply.txt`: exact LLM request and response artifacts.
- `d435_rgb.png` / `d405_rgb.png`: real camera frames sent to the LLM.
- `d435_llm_vs_segmentation_overlay.png` / `d405_llm_vs_segmentation_overlay.png`: green LLM pixel/box vs. yellow segmentation pixel/box.
- `*_depth_colormap.png` and `*_depth_raw.npy`: real aligned depth data from the cameras.

Run only one camera:

```bash
python3 Module_Tests/Working_Tests/live_llm_camera_seg_debug/live_llm_camera_seg_debug.py \
  --camera d435 \
  --target "mug" \
  --target-color white
```

Disable color segmentation when you only want LLM and depth behavior:

```bash
python3 Module_Tests/Working_Tests/live_llm_camera_seg_debug/live_llm_camera_seg_debug.py \
  --target "the object on the table" \
  --target-color none
```

This script reports camera-frame XYZ only. It intentionally does not use robot-frame transforms, trajectory planning, robot state, or the Franka interface.
