# YOLO Segmentation Test - Live Webcam Feed
# Performs real-time object segmentation on a webcam video stream
# Known issues: high latency, detection accuracy varies by model choice

import cv2
from ultralytics import YOLO

# ---- Model Selection ----
# Option 1: yoloe-26l-seg (promptable/empty-prompt model)
#   - Higher accuracy for detecting specific object classes
#   - Slower inference due to larger model size
#   - Uses model.set_classes() to restrict detection to a predefined list
model = YOLO("yoloe-26l-seg.pt")
# Set the classes to detect. YOLOE filters predictions to only these categories.
model.set_classes(["person", "mouse", "keyboard", "guitar", "waterbottle", "can"])

# ---- Option 2: yolo26n-seg (nano model, commented out) ----
# Uncomment the lines below to switch to the nano model for faster inference.
# The nano model detects all COCO classes (no class filtering) but is less accurate.
# # ================================
# model = YOLO("yolo26n-seg.pt")
# # ================================

# ---- Webcam Initialization ----
# Open the default camera (index 0) using the V4L2 backend (Linux)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Verify the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
else:
    print("Success: Webcam opened successfully.")
    print("Press 'q' to exit the webcam feed.")

# ---- Main Inference Loop ----
# Continuously read frames from the webcam, run segmentation, and display results
while cap.isOpened():
    # Read the next frame from the camera
    # ret: boolean indicating success; frame: the captured image (numpy array)
    ret, frame = cap.read()

    # Exit if the frame could not be read (e.g., camera disconnected)
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # Run YOLO segmentation inference on the current frame
    # results[0] contains the detections for this single frame
    results = model(frame)

    # Annotate the frame with bounding boxes, masks, and class labels
    # plot() returns a numpy array with all detections drawn on the image
    annotated_frame = results[0].plot()

    # Display the annotated frame in a window
    cv2.imshow('Webcam Object Segmentation', annotated_frame)

    # Wait 10ms for a key press; if 'q' is pressed, break the loop
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# ---- Cleanup ----
# Release the camera resource and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
