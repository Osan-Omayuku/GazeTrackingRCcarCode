import cv2
import dlib
import serial
import logging
from gaze_tracking import GazeTracking
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
                cv2.putText(frame, f"{direction.capitalize()} captured", (50, 50),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("Calibration", frame)
                cv2.waitKey(1000)
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

# Timer for FPS calculation
fps = 0
fps_start_time = time.time()
frame_count = 0

# Main tracking loop with fallback to head movement detection
while True:
    ret, frame = webcam.read()
    if not ret:
        logging.error("Frame capture failed in main loop.")
        break

    frame_count += 1

    # Refresh gaze tracking
    gaze.refresh(frame)
    frame = gaze.annotated_frame()
    text = ""
    head_movement = "C"  # Default to "Centered"
    accuracy_degrees = 0  # Default accuracy in case no face is detected

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    if faces:
        face = faces[0]
        landmarks = predictor(gray, face)
        nose_tip = (landmarks.part(30).x, landmarks.part(30).y)

        # Calculate deviations for head movement detection
        x_deviation = nose_tip[0] - calibration_data["center"]["nose_tip"][0]
        y_deviation = nose_tip[1] - calibration_data["center"]["nose_tip"][1]
        face_width = calibration_data["center"]["face_width"]
        face_height = calibration_data["center"]["face_height"]

        # Determine head movement
        if x_deviation < -0.15 * face_width:
            head_movement = "L"
            text = "Head turned left"
        elif x_deviation > 0.15 * face_width:
            head_movement = "R"
            text = "Head turned right"
        elif y_deviation < -0.2 * face_height:
            head_movement = "U"
            text = "Head tilted up"
        elif y_deviation > 0.2 * face_height:
            head_movement = "D"
            text = "Head tilted down"
        else:
            text = "Head centered"

        # Fallback to head movement if gaze tracking fails
        if gaze.horizontal_ratio() is None or gaze.vertical_ratio() is None:
            logging.info("Gaze tracking failed; using head movement as fallback.")
            if arduino:
                try:
                    arduino.write(head_movement.encode())
                    logging.info(f"Sent to Arduino: {head_movement}")
                except serial.SerialException as e:
                    logging.warning("Lost connection to Arduino: %s", e)
                    arduino = connect_arduino()
            text += " (Fallback to head movement)"

    else:
        # No face detected
        text = "No face detected. Fallback not available."
        logging.warning("No face detected; gaze tracking and head movement detection unavailable.")

    # FPS Calculation
    if time.time() - fps_start_time >= 1.0:
        fps = frame_count / (time.time() - fps_start_time)
        frame_count = 0
        fps_start_time = time.time()

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Calibrated Head and Gaze Tracking", frame)

    if cv2.waitKey(10) == 27:
        break

webcam.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
logging.info("Resources released and program terminated.")
