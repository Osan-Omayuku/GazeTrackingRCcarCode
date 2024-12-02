import cv2
import dlib
import serial
import logging
from gaze_tracking import GazeTracking
from collections import deque
import time

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize GazeTracking, webcam, and dlib's face detector
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()
model_path = r"C:\Users\RON-DON\AppData\Local\Programs\Python\Python39\Lib\site-packages\shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(model_path)

# Attempt to connect to Arduino on COM7
def connect_arduino(port='COM7', baudrate=9600):
    try:
        arduino = serial.Serial(port, baudrate, timeout=1)
        logging.info("Arduino connected.")
        return arduino
    except serial.SerialException as e:
        logging.warning("Arduino connection failed: %s", e)
        return None

arduino = connect_arduino()
if not arduino:
    logging.warning("Proceeding without Arduino connection. Commands will not be sent.")

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
        input(f"Please look {direction}. Press Enter to start capturing...")
        logging.info(f"Starting capture for {direction} position.")

        attempts, captured = 0, False
        while not captured and attempts < 10:  # Retry up to 10 times
            attempts += 1
            ret, frame = webcam.read()
            if not ret:
                logging.warning("Frame not captured. Retrying...")
                time.sleep(0.5)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray)

            if faces:
                face = faces[0]
                landmarks = predictor(gray, face)

                # Store positions
                calibration_data[direction] = {
                    "nose_tip": (landmarks.part(30).x, landmarks.part(30).y),
                    "chin": (landmarks.part(8).x, landmarks.part(8).y),
                    "left_eye": (landmarks.part(36).x, landmarks.part(36).y),
                    "right_eye": (landmarks.part(45).x, landmarks.part(45).y),
                    "face_width": landmarks.part(45).x - landmarks.part(36).x,
                    "face_height": landmarks.part(8).y - landmarks.part(30).y
                }

                logging.info(f"{direction.capitalize()} position captured.")
                captured = True
            else:
                logging.warning(f"No face detected for {direction}. Attempt {attempts}.")
                time.sleep(1)  # Add delay between attempts

        if not captured:
            logging.error(f"Calibration failed for {direction} after {attempts} attempts.")
            return False  # Exit calibration if a critical direction fails
    return True

# Start calibration process
calibrated = calibrate()
cv2.destroyAllWindows()

# Exit if calibration is incomplete
if not calibrated or calibration_data.get("center") is None:
    logging.error("Calibration incomplete. Please ensure all directions are calibrated.")
    webcam.release()
    if arduino:
        arduino.close()
    exit()

# Moving average parameters
window_size = 5  # Number of frames to average over
horizontal_ratios = deque(maxlen=window_size)
vertical_ratios = deque(maxlen=window_size)
x_deviation_avg = deque(maxlen=window_size)
y_deviation_avg = deque(maxlen=window_size)

# Main tracking loop with moving average
while True:
    ret, frame = webcam.read()
    if not ret:
        logging.error("Frame capture failed in main loop.")
        break

    # Refresh gaze tracking
    gaze.refresh(frame)
    horizontal_ratio = gaze.horizontal_ratio()
    vertical_ratio = gaze.vertical_ratio()

    # Add gaze ratios to moving average queues
    if horizontal_ratio is not None:
        horizontal_ratios.append(horizontal_ratio)
    if vertical_ratio is not None:
        vertical_ratios.append(vertical_ratio)

    # Calculate smoothed averages
    smoothed_horizontal = sum(horizontal_ratios) / len(horizontal_ratios) if horizontal_ratios else None
    smoothed_vertical = sum(vertical_ratios) / len(vertical_ratios) if vertical_ratios else None

    # Process head movement
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    if faces:
        face = faces[0]
        landmarks = predictor(gray, face)
        nose_tip = (landmarks.part(30).x, landmarks.part(30).y)

        # Calculate deviations
        x_deviation = nose_tip[0] - calibration_data["center"]["nose_tip"][0]
        y_deviation = nose_tip[1] - calibration_data["center"]["nose_tip"][1]

        # Add deviations to moving average queues
        x_deviation_avg.append(x_deviation)
        y_deviation_avg.append(y_deviation)

        # Calculate smoothed deviations
        smoothed_x_deviation = sum(x_deviation_avg) / len(x_deviation_avg)
        smoothed_y_deviation = sum(y_deviation_avg) / len(y_deviation_avg)

        # Determine head movement using smoothed deviations
        if smoothed_x_deviation < -0.15 * calibration_data["center"]["face_width"]:
            head_movement = "L"
            text = "Head turned left"
        elif smoothed_x_deviation > 0.15 * calibration_data["center"]["face_width"]:
            head_movement = "R"
            text = "Head turned right"
        elif smoothed_y_deviation < -0.2 * calibration_data["center"]["face_height"]:
            head_movement = "U"
            text = "Head tilted up"
        elif smoothed_y_deviation > 0.2 * calibration_data["center"]["face_height"]:
            head_movement = "D"
            text = "Head tilted down"
        else:
            head_movement = "C"
            text = "Head centered"

        # Output results
        print(f"Smoothed Horizontal Ratio: {smoothed_horizontal}")
        print(f"Smoothed Vertical Ratio: {smoothed_vertical}")
        print(f"Smoothed Deviations: x={smoothed_x_deviation}, y={smoothed_y_deviation}")
        print(f"Head Movement Detected: {head_movement}")

        # Send head movement command to Arduino
        if arduino:
            try:
                arduino.write(head_movement.encode())
                logging.info(f"Sent to Arduino: {head_movement}")
            except serial.SerialException as e:
                logging.warning("Lost connection to Arduino: %s", e)
                arduino = connect_arduino()

    # Display results on frame
    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Gaze and Head Tracking", frame)

    if cv2.waitKey(10) == 27:
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
logging.info("Resources released and program terminated.")
