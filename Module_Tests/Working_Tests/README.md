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

LLM/server integration checks:

```bash
python3 Module_Tests/Working_Tests/llm_image_capability_test29.py
```

This image-capability test does not need cameras. It sends synthetic red/green
square images to the configured `LLM_API_URL` and reports whether the local
multimodal model can see one image and two images in a single request. It uses
JPEG by default when Pillow is available, matching the runtime camera path, and
falls back to PNG otherwise. Each run saves a timestamped folder under
`Module_Tests/Test_Outputs/llm_image_capability` with `inputs/`, `requests/`,
and `outputs/` subfolders.
Single-shot LLM pixel annotation (live cameras):

`ash
python3 Module_Tests/Working_Tests/llm_pixel_annotation_test30.py --prompt "describe the red block"
Single-shot LLM pixel annotation (live cameras):

`ash
python3 Module_Tests/Working_Tests/llm_pixel_annotation_test30.py --prompt "describe the red block"
`

This script mirrors the interactive no-robot debug loop for one command. It captures synchronized D435/D405 frames, lets the LLM pick pixel coordinates, and stores the annotated side-by-side visualization, sanitized conversation, and run summary under Module_Tests/Test_Outputs/llm_pixel_annotation_test/<timestamp>/. Use --ee-pose-source, --ee-pose-file, and --pose-index to override the static D405 pose when validating hand-eye math.
