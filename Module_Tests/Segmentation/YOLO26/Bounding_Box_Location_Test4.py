###########################
# YOLO segmentation to retrieve center coordinate of object bounding box
# Goal: Bounding box segmentation and pass object center coordinate through the depth camera to obtain depth of the object
# Issue: Center of bounding box may not always be on the object, YOLO has limits on identifying objects
###########################

from ultralytics import YOLO
import cv2
import torch
import numpy as np
from pathlib import Path
import os

# Load a model. Current model is nano segmentation.
model = YOLO("yolo26n-seg.pt")

# Get the directory of the current script
script_dir = Path(__file__).resolve().parent
print(f"Script directory: {script_dir}")

# Construct the path to your file
file_path = "Example_Images/FishingRod.png"
print(f"Image path: {file_path}")

# Predict on provided image
results = model(file_path)

# Create the output folder so it doesn't crash if the folder is missing
os.makedirs("Yolo_Output", exist_ok=True)

for r in results:
    # Save the base YOLO result
    save_path = os.path.join("Yolo_Output", "FishingRod_bounding_box_result.jpg")
    r.save(filename=save_path)
    
    # Read the image back so we can draw on it with OpenCV
    image = cv2.imread(save_path)
    
    # Check if YOLO actually found anything before trying to loop through boxes
    if len(r.boxes) > 0:
        for box in r.boxes:
            # Get center coordinates
            b_xywh = box.xywh[0]  
            
            # Convert tensor to integer
            b_x_center = int(b_xywh[0].item())
            b_y_center = int(b_xywh[1].item())
            
            print(f"Found object at X: {b_x_center}, Y: {b_y_center}")
            
            # Locate the center of the bounding box with a red circle
            cv2.circle(image, (b_x_center, b_y_center), radius=20, color=(0, 0, 255), thickness=-1)
            
        # Save the final image with the circle drawn on it
        final_save_path = os.path.join("Yolo_Output", "center_fishingrod_bounding_box_result.jpg")
        cv2.imwrite(final_save_path, image)
        print(f"Saved final image to {final_save_path}")
        
    else:
        print("Warning: No objects were detected in the image.")

print("yay")