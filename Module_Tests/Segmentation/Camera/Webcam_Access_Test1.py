###########################
# Webcam Access Test
# Press 'q' to close window
###########################

import cv2 # import Open Source Computer Vision Library

# Initialize camera using DirectShow backend (0 is usually the default built-in camera)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2) # or cv2.CAP_MSMF


# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam successfully opened.")

while True:
    # Capture frame-by-frame
    # 'ret' is a boolean (True if frame is captured), 'frame' is the image data
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Display the resulting frame in a window named 'Webcam'
    cv2.imshow('Webcam', frame)

    # Press 'q' on the keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("yay")