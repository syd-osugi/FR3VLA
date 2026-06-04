###########################
# YOLO Segmentation Test
# General segmentation using an image
###########################

from ultralytics import YOLO
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_TESTS_ROOT = SCRIPT_DIR.parents[1]
OUTPUT_DIR = MODULE_TESTS_ROOT / "Test_Outputs" / Path(__file__).stem

# Load a model. Current model is nano segmentation.
model = YOLO(str(SCRIPT_DIR / "yolo26n-seg.pt"))  # load an official model from https://docs.ultralytics.com/models/

results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for r in results:
    # This allows you to specify any valid system path and filename
    save_path = OUTPUT_DIR / "bus_result.jpg"
    r.save(filename=str(save_path))

print("yay")
