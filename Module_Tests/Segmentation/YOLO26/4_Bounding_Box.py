# YOLO Segmentation - Bounding Box Center Coordinate Extraction
# Extracts the center (x, y) coordinates of detected object bounding boxes
# Purpose: Coordinates will be passed to a depth camera to obtain object depth values
# Limitation: The bounding box center may not always fall on the actual object

from ultralytics import YOLO
import cv2
import torch
import numpy as np
from pathlib import Path

# ---- Model Loading ----
# Load the YOLO26 nano segmentation model
SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_TESTS_ROOT = SCRIPT_DIR.parents[1]
OUTPUT_DIR = MODULE_TESTS_ROOT / "Test_Outputs" / Path(__file__).stem
model = YOLO(str(SCRIPT_DIR / "yolo26n-seg.pt"))

print(f"Script directory: {SCRIPT_DIR}")

# ---- Image Path Setup ----
# Define the path to the test image containing the object to detect
file_path = SCRIPT_DIR / "Example_Images" / "Donut.jpg"
print(f"Image path: {file_path}")

# ---- Inference ----
# Run segmentation on the specified image
results = model(str(file_path))

# Ensure the output directory exists before saving
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Process Detections ----
for r in results:
    # Save the raw YOLO output image (boxes + masks drawn)
    save_path = OUTPUT_DIR / "donut_bounding_box_result.jpg"
    r.save(filename=str(save_path))

    # Reload the saved image so we can draw additional annotations with OpenCV
    image = cv2.imread(str(save_path))

    # Check whether YOLO detected any objects (boxes) in the image
    if len(r.boxes) > 0:
        # Iterate over each detected bounding box
        for box in r.boxes:
            # Extract the bounding box in (center_x, center_y, width, height) format
            # r.boxes is a tensor batch; [0] selects the first (only) detection
            b_xywh = box.xywh[0]

            # Convert PyTorch tensor values to Python integers
            b_x_center = int(b_xywh[0].item())
            b_y_center = int(b_xywh[1].item())

            # Print the center coordinates for debugging/depth mapping
            print(f"Found object at X: {b_x_center}, Y: {b_y_center}")

            # Draw a filled red circle at the bounding box center
            # cv2.circle(image, center, radius, color(BGR), thickness=-1 means filled)
            cv2.circle(image, (b_x_center, b_y_center), radius=20, color=(0, 0, 255), thickness=-1)

        # Save the final annotated image with the red center marker
        final_save_path = OUTPUT_DIR / "center_donut_bounding_box_result.jpg"
        cv2.imwrite(str(final_save_path), image)
        print(f"Saved final image to {final_save_path}")

    else:
        # No objects were detected in this image
        print("Warning: No objects were detected in the image.")

print("yay")
