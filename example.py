import cv2
import dlib
from gaze_tracking import GazeTracking
import numpy as np

# Initialize GazeTracking, webcam, and dlib's face detector
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()
model_path = r"C:\Users\RON-DON\AppData\Local\Programs\Python\Python39\Lib\site-packages\shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(model_path)

# Calibration data storage
calibration_data = {"center": [], "left": [], "right": [], "up": [], "down": []}

def calculate_ranges(data):
    ranges = {}
    for direction, coords in data.items():
        if coords:
            x_coords = [p[0] for p in coords]
            y_coords = [p[1] for p in coords]
            ranges[direction] = {
                "x_range": (min(x_coords) - 10, max(x_coords) + 10),
                "y_range": (min(y_coords) - 10, max(y_coords) + 10)
            }
    return ranges

# Calibration step for each direction
print("Starting calibration. Look in each direction when prompted.")
directions = ["center", "left", "right", "up", "down"]

for direction in directions:
    input(f"Press Enter and look {direction} for 3 seconds...")
    start_time = cv2.getTickCount()
    while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < 3:
        _, frame = webcam.read()
        gaze.refresh(frame)
        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        if left_pupil and right_pupil:
            avg_pupil_x = (left_pupil[0] + right_pupil[0]) / 2
            avg_pupil_y = (left_pupil[1] + right_pupil[1]) / 2
            calibration_data[direction].append((avg_pupil_x, avg_pupil_y))
        cv2.putText(frame, f"Calibrating: Look {direction}", (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) == 27:
            break

# Check and recalibrate if any direction has missing data
for direction in directions:
    if not calibration_data[direction]:
        print(f"Warning: No data captured for {direction}. Recalibrating {direction}...")
        input(f"Press Enter and look {direction} for 3 seconds...")
        start_time = cv2.getTickCount()
        while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < 3:
            _, frame = webcam.read()
            gaze.refresh(frame)
            left_pupil = gaze.pupil_left_coords()
            right_pupil = gaze.pupil_right_coords()
            if left_pupil and right_pupil:
                avg_pupil_x = (left_pupil[0] + right_pupil[0]) / 2
                avg_pupil_y = (left_pupil[1] + right_pupil[1]) / 2
                calibration_data[direction].append((avg_pupil_x, avg_pupil_y))
            cv2.putText(frame, f"Recalibrating: Look {direction}", (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) == 27:
                break

# Calculate gaze ranges from calibration data
ranges = calculate_ranges(calibration_data)
print("Calibration completed. Starting gaze tracking with head movement detection...")

# Main tracking loop with head movement detection
while True:
    _, frame = webcam.read()
    gaze.refresh(frame)
    frame = gaze.annotated_frame()
    text = ""

    # Get gaze coordinates
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    
    # Detect face and facial landmarks
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    if faces:
        face = faces[0]
        landmarks = predictor(gray, face)
        nose = landmarks.part(30)  # Nose bridge midpoint as reference

        # Calculate head movement based on nose position relative to the face bounding box
        face_center_x = (face.left() + face.right()) / 2
        face_center_y = (face.top() + face.bottom()) / 2

        if nose.x < face_center_x - 10:  # Adjust threshold for sensitivity
            head_movement = "left"
        elif nose.x > face_center_x + 10:
            head_movement = "right"
        elif nose.y < face_center_y - 10:
            head_movement = "up"
        elif nose.y > face_center_y + 10:
            head_movement = "down"
        else:
            head_movement = "center"
    else:
        head_movement = "center"

    # Determine gaze direction based on dynamic ranges and head movement
    if left_pupil and right_pupil:
        avg_pupil_x = (left_pupil[0] + right_pupil[0]) / 2
        avg_pupil_y = (left_pupil[1] + right_pupil[1]) / 2

        # Check combined gaze and head direction
        if ranges["center"]["x_range"][0] <= avg_pupil_x <= ranges["center"]["x_range"][1] and \
           ranges["center"]["y_range"][0] <= avg_pupil_y <= ranges["center"]["y_range"][1] and \
           head_movement == "center":
            text = "Looking center"
        elif ranges["left"]["x_range"][0] <= avg_pupil_x <= ranges["left"]["x_range"][1] and \
             ranges["left"]["y_range"][0] <= avg_pupil_y <= ranges["left"]["y_range"][1] or head_movement == "left":
            text = "Looking left"
        elif ranges["right"]["x_range"][0] <= avg_pupil_x <= ranges["right"]["x_range"][1] and \
             ranges["right"]["y_range"][0] <= avg_pupil_y <= ranges["right"]["y_range"][1] or head_movement == "right":
            text = "Looking right"
        elif ranges["up"]["x_range"][0] <= avg_pupil_x <= ranges["up"]["x_range"][1] and \
             ranges["up"]["y_range"][0] <= avg_pupil_y <= ranges["up"]["y_range"][1] or head_movement == "up":
            text = "Looking up"
        elif ranges["down"]["x_range"][0] <= avg_pupil_x <= ranges["down"]["x_range"][1] and \
             ranges["down"]["y_range"][0] <= avg_pupil_y <= ranges["down"]["y_range"][1] or head_movement == "down":
            text = "Looking down"

    # Display the direction on the frame
    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    # Display pupil and head movement coordinates
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Head movement: " + head_movement, (90, 200), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Gaze Tracking with Head Movement", frame)

    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()
