###########################
# Yolo Segmentation Test using Webcam for a live feed
# Issue: Poor latency, object identification isn't great
###########################

import cv2
from ultralytics import YOLO

# Load desired model
# Using prompt-based model for better accuracy in detecting specific classes (e.g., person)
# Uncomment lines below to use yoloe-26l-seg
# ================================
model = YOLO("yoloe-26l-seg.pt")
# Add prompt to only detect the certain items
model.set_classes(["person", "mouse", "keyboard", "guitar", "waterbottle", "can"])
# ================================

# # Use yolo26n-seg for faster inference, but it may be less accurate and will detect all classes
# # Uncomment line below to use yolo26n-seg
# # ================================
# model = YOLO("yolo26n-seg.pt")
# # ================================

# Initalize the webcam
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
else:
    print("Success: Webcam opened successfully.")
    print("Press 'q' to exit the webcam feed.")

# Perform segmentaion while webcam is open
while cap.isOpened():
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # Perform object detection and segmentation
    results = model(frame)

    # Display the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow('Webcam Object Segmentation', annotated_frame)

    # Break the loop is 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()