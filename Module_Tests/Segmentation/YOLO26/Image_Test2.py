###########################
# YOLO Segmentation Test
# General segmentation using an image
###########################

from ultralytics import YOLO
import os

# Load a model. Current model is nano segmentation.
model = YOLO("yolo26n-seg.pt")  # load an official model from https://docs.ultralytics.com/models/

results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

for r in results:
    # This allows you to specify any valid system path and filename
    save_path = os.path.join("Yolo_Output", "bus_result.jpg")
    r.save(filename=save_path)

print("yay")