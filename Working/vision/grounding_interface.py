"""
SEGMENTATION PLACEHOLDER FILE

Currently, we are relying on the LLM (Qwen) to look at the image and guess the [u, v] 
pixel coordinates of an object. This works, but LLMs are historically bad at exact 
pixel math.

In the future, you will replace this with YOLO, SAM (Segment Anything), or another model.
Because of this placeholder file, when you DO switch to YOLO, you will only need to edit 
THIS file. The rest of your robot code (tools.py, main.py) won't need to change at all.
"""
from vision.base_classes import BaseDetector

class LLMVisualDetector(BaseDetector):
    """
    FAKE DETECTOR: Returns an empty list because the LLM handles vision natively.
    Because it inherits from BaseDetector, it is legally allowed to be used anywhere 
    in the code that expects a Detector.
    """
    def detect(self, image, classes_to_find=None):
        return []

# --- EXAMPLE OF HOW YOU WILL IMPLEMENT YOLO LATER ---
# class YOLODetector(BaseDetector):
#     def __init__(self, model_path):
#         from ultralytics import YOLO
#         self.model = YOLO(model_path)
# 
#     def detect(self, image, classes_to_find=None):
#         # ... YOLO logic ...
#         return detections