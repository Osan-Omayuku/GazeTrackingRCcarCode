import cv2
from gaze_tracking import GazeTracking

# Initialize the gaze tracking object
gaze = GazeTracking()

# Start the webcam feed (0 is typically the default webcam)
webcam = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = webcam.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Send the frame to GazeTracking object to analyze
    gaze.refresh(frame)

    # Get the annotated frame (with pupils highlighted)
    frame = gaze.annotated_frame()

    # Determine the direction of the gaze
    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"
    else:
        text = "No gaze detected"

    # Display the result on the frame
    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    # Show the frame in a window
    cv2.imshow("Gaze Tracking", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
webcam.release()
cv2.destroyAllWindows()
