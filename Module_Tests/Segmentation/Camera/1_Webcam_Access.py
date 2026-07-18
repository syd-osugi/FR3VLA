# Webcam Access Test - Basic Camera Stream
# Opens the default webcam and displays the live video feed in a window
# Press 'q' to close the window and exit

import cv2 # import Open Source Computer Vision Library

# Initialize the camera capture object
# Index 0 = default camera (built-in or first USB camera)
# CAP_V4L2 = Linux V4L2 backend; alternatives: CAP_MSMF (Windows), CAP_ANY (auto-detect)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2) # or cv2.CAP_MSMF


# Verify the camera was opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam successfully opened.")

# ---- Main Loop ----
# Continuously grab frames from the camera and display them
while True:
    # Grab the next frame from the camera
    # ret: boolean (True if frame was captured successfully)
    # frame: numpy array containing the image data (H x W x 3, BGR format)
    ret, frame = cap.read()

    # Exit if the frame could not be read (camera disconnected or error)
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Display the frame in a window titled 'Webcam'
    cv2.imshow('Webcam', frame)

    # Wait 1ms for a keyboard event; if 'q' is pressed, exit the loop
    # cv2.waitKey() returns the ASCII code of the pressed key, or -1 if no key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---- Cleanup ----
# Release the camera so other applications can use it
cap.release()
# Close all OpenCV windows that were created
cv2.destroyAllWindows()

print("yay")
