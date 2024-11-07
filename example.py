import cv2
import dlib
import serial  # For serial communication
from gaze_tracking import GazeTracking
import numpy as np
import time

# Initialize GazeTracking, webcam, and dlib's face detector
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()
model_path = r"C:\Users\RON-DON\AppData\Local\Programs\Python\Python39\Lib\site-packages\shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(model_path)

# Attempt to connect to Arduino on COM7
try:
    arduino = serial.Serial('COM7', 9600)  # Connect to Arduino on COM7
    arduino_connected = True
    print("Arduino connected.")
except serial.SerialException:
    arduino_connected = False
    print("Arduino not connected. Running without Arduino control.")

print("Starting gaze tracking with head movement detection...")

# Calibration parameters
calibrated = False
calibration_data = {
    "center": None,
    "left": None,
    "right": None,
    "up": None,
    "down": None
}

# Calibration function to capture head positions for different directions
def calibrate():
    directions = ["center", "left", "right", "up", "down"]
    for direction in directions:
        print(f"Please look {direction}. Press Enter to start capturing...")
        input()  # Wait for user to press Enter
        print(f"Capturing {direction} position... Please hold still.")

        # Capture the face position for the current direction
        captured = False
        while not captured:
            _, frame = webcam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray)

            if faces:
                face = faces[0]
                landmarks = predictor(gray, face)
                
                # Extract key facial landmark positions
                nose_tip = (landmarks.part(30).x, landmarks.part(30).y)  # Nose tip
                chin = (landmarks.part(8).x, landmarks.part(8).y)        # Chin
                left_eye = (landmarks.part(36).x, landmarks.part(36).y)  # Left eye corner
                right_eye = (landmarks.part(45).x, landmarks.part(45).y) # Right eye corner
                
                # Store baseline positions for each direction
                calibration_data[direction] = {
                    "nose_tip": nose_tip,
                    "chin": chin,
                    "left_eye": left_eye,
                    "right_eye": right_eye,
                    "face_width": right_eye[0] - left_eye[0],
                    "face_height": chin[1] - nose_tip[1]
                }

                # Confirm calibration for the current direction
                cv2.putText(frame, f"{direction.capitalize()} captured", (50, 50),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("Calibration", frame)
                cv2.waitKey(1000)
                captured = True

            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) == 27:
                break
    return True

# Start calibration process
calibrated = calibrate()
cv2.destroyAllWindows()

# Set a 15-second timer before starting to use calibrated values
start_time = time.time()
use_calibrated_values = False

# Main tracking loop with head movement detection
while calibrated:
    _, frame = webcam.read()
    gaze.refresh(frame)
    frame = gaze.annotated_frame()
    text = ""

    # Check if 15 seconds have passed since calibration
    if time.time() - start_time > 15:
        use_calibrated_values = True

    # Detect face and facial landmarks
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    if faces:
        face = faces[0]
        landmarks = predictor(gray, face)

        # Extract current positions
        nose_tip = (landmarks.part(30).x, landmarks.part(30).y)
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)

        # Use calibrated values if 15 seconds have passed
        if use_calibrated_values:
            center_data = calibration_data["center"]
            face_width = center_data["face_width"]
            face_height = center_data["face_height"]

            # Determine head movements based on relative position to the calibrated center
            if nose_tip[0] < center_data["nose_tip"][0] - 0.15 * face_width:
                head_movement = "L"  # Head turned left
                text = "Head turned left"
            elif nose_tip[0] > center_data["nose_tip"][0] + 0.15 * face_width:
                head_movement = "R"  # Head turned right
                text = "Head turned right"
            elif nose_tip[1] < center_data["nose_tip"][1] - 0.2 * face_height:
                head_movement = "F"  # Head tilted up
                text = "Head tilted up"
            elif nose_tip[1] > center_data["nose_tip"][1] + 0.2 * face_height:
                head_movement = "B"  # Head tilted down
                text = "Head tilted down"
            else:
                head_movement = "c"  # Centered
                text = "Head centered"
        else:
            # During the initial 15 seconds, show that calibration values are not yet in use
            text = "Calibration values not yet in use"

        # Send the head movement command to Arduino if connected
        if arduino_connected and use_calibrated_values:
            try:
                arduino.write(head_movement.encode('utf-8'))  # Send as a single character
            except serial.SerialException:
                print("Lost connection to Arduino.")
                arduino_connected = False

    else:
        head_movement = "c"
        text = "No face detected"

    # Display the head movement direction on the frame
    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    cv2.imshow("Calibrated Head and Gaze Tracking", frame)

    if cv2.waitKey(1) == 27:
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
if arduino_connected:
    arduino.close()
