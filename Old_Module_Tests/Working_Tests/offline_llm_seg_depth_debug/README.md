# Offline LLM / Segmentation / Depth Math Debug

This folder is the synthetic math-only fixture. It simulates the LLM response and camera depth so you can test JSON parsing, segmentation comparison, and depth-coordinate plumbing when no hardware is available.

For real camera hardware plus the real configured LLM server, use:

```bash
python3 Module_Tests/Working_Tests/live_llm_camera_seg_debug/live_llm_camera_seg_debug.py \
  --target "red block" \
  --target-color red
```

Run the offline synthetic fixture:

```bash
python3 Module_Tests/Working_Tests/offline_llm_seg_depth_debug/offline_llm_seg_depth_debug.py
```

Outputs go to:

```text
Module_Tests/Test_Outputs/offline_llm_seg_depth_debug/<timestamp>/
```

Useful files in each offline run:

- `report.json`: parsed mock LLM tool call, segmentation pixels, depth stats, XYZ results, and fusion comparison.
- `d435_segmentation_llm_overlay.png` / `d405_segmentation_llm_overlay.png`: segmentation center vs. mock LLM-selected pixel.
- `*_depth_colormap.png`: fake or loaded depth map visualization.

Replay a saved LLM response into the synthetic fixture:

```bash
python3 Module_Tests/Working_Tests/offline_llm_seg_depth_debug/offline_llm_seg_depth_debug.py \
  --llm-response-file /path/to/llm_response.json
```
