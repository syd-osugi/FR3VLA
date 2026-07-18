# YOLO Segmentation Test - Image Inference
# Tests YOLO26 nano segmentation model on a static image
# Pipeline: load model -> run inference on remote image -> save annotated output

from ultralytics import YOLO
from pathlib import Path


# Resolve the directory containing this script file
SCRIPT_DIR = Path(__file__).resolve().parent
# Navigate up two levels to reach the project root (Module_Tests root)
MODULE_TESTS_ROOT = SCRIPT_DIR.parents[1]
# Define output directory: <root>/Test_Outputs/<this_script_name>
OUTPUT_DIR = MODULE_TESTS_ROOT / "Test_Outputs" / Path(__file__).stem

# Load the YOLO26 nano segmentation model from a .pt file
# The nano model is the smallest/fastest variant in the YOLO26 family
model = YOLO(str(SCRIPT_DIR / "yolo26n-seg.pt"))  # load an official model from https://docs.ultralytics.com/models/

# Run inference on a remote image URL
# YOLO automatically downloads the image and processes it
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

# Create the output directory (and any missing parent directories)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Iterate over each result batch returned by YOLO
for r in results:
    # Specify the output file path for the annotated image
    save_path = OUTPUT_DIR / "bus_result.jpg"
    # Save the image with bounding boxes, masks, and labels drawn on it
    r.save(filename=str(save_path))

print("yay")
