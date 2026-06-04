# Working Test Scripts

This folder has two kinds of scripts:

- `unit_*.py`: no-hardware checks for small pieces of `Working`.
- the older numbered scripts such as `camera_access_test1.py`: integration checks that may need RealSense cameras, calibration files, a local LLM server, or robot hardware.

Run the lightweight suite:

```bash
python3 Module_Tests/Working_Tests/run_unit_tests.py
```

Run one slice while debugging:

```bash
python3 Module_Tests/Working_Tests/unit_trajectory_math_test17.py
python3 Module_Tests/Working_Tests/run_unit_tests.py --pattern "unit_charuco*"
```

The unit scripts use fake cameras, fake depth frames, monkeypatched transforms,
and temporary calibration JSON files. Use them first when changing math or tool
routing, then run the hardware integration scripts once the small pieces pass.

Calibration hardware checks:

```bash
python3 Module_Tests/Working_Tests/intrinsics_check_test12.py
python3 Module_Tests/Working_Tests/d405_hand_eye_check_test13.py
python3 Module_Tests/Working_Tests/d435_bird_eye_check_test23.py
python3 Module_Tests/Working_Tests/charuco_detection_video_test24.py --camera D435
```
