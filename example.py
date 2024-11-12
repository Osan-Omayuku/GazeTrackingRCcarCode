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

            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) == 27:
                logging.info("Calibration interrupted by user.")
                return False

        if not captured:
            logging.error(f"Calibration failed for {direction} after {attempts} attempts.")
    return True

# Start calibration process
calibrated = calibrate()
cv2.destroyAllWindows()

# Timer before using calibrated values
start_time = time.time()
use_calibrated_values = False

# Initialize the last_print_time variable
last_print_time = time.time()

# Main tracking loop with head movement detection
while calibrated:
    ret, frame = webcam.read()
    if not ret:
        logging.error("Frame capture failed in main loop.")
        break

    # Limit processing to every 30 ms to reduce CPU load
    time.sleep(0.03)

    # Refresh gaze tracking
    gaze.refresh(frame)
    frame = gaze.annotated_frame()
    text = "Calibration values not yet in use" if not use_calibrated_values else ""

    # Check if 15 seconds have passed since calibration
    if time.time() - start_time > 15:
        use_calibrated_values = True

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    if faces:
        face = faces[0]
        landmarks = predictor(gray, face)
        nose_tip = (landmarks.part(30).x, landmarks.part(30).y)

        # Initialize default deviation values
        x_deviation = 0
        y_deviation = 0
        x_deviation_percent = 0
        y_deviation_percent = 0
        head_movement = "c"  # Default to centered if no movement detected

        if use_calibrated_values and "center" in calibration_data:
            center_data = calibration_data["center"]
            face_width = center_data["face_width"]
            face_height = center_data["face_height"]

            # Calculate deviations only if calibrated center data is available
            x_deviation = nose_tip[0] - center_data["nose_tip"][0]
            y_deviation = nose_tip[1] - center_data["nose_tip"][1]
            x_deviation_percent = x_deviation / face_width if face_width != 0 else 0
            y_deviation_percent = y_deviation / face_height if face_height != 0 else 0

            # Determine head movement
            if x_deviation < -0.15 * face_width:
                head_movement = "L"
                text = "Head turned left"
            elif x_deviation > 0.15 * face_width:
                head_movement = "R"
                text = "Head turned right"
            elif y_deviation < -0.2 * face_height:
                head_movement = "F"
                text = "Head tilted up"
            elif y_deviation > 0.2 * face_height:
                head_movement = "B"
                text = "Head tilted down"
            else:
                head_movement = "c"
                text = "Head centered"
        else:
            text = "Calibration values not yet in use"

        # Send head movement command to Arduino if connected
        if arduino and use_calibrated_values:
            try:
                arduino.write(head_movement.encode())
                logging.info(f"Sent to Arduino: {head_movement}")
            except serial.SerialException as e:
                logging.warning("Lost connection to Arduino: %s", e)
                arduino = connect_arduino()  # Attempt reconnection

        # Get current time
        current_time = time.time()
        if current_time - last_print_time >= 1.0:
            # Calculate gaze ratios
            horizontal_ratio = gaze.horizontal_ratio()
            vertical_ratio = gaze.vertical_ratio()

            # Determine gaze direction
            if gaze.is_right():
                gaze_direction = "Looking right"
            elif gaze.is_left():
                gaze_direction = "Looking left"
            elif gaze.is_center():
                gaze_direction = "Looking center"
            else:
                gaze_direction = "Gaze direction unknown"

            # Print gaze ratios and direction
            print(f"\n--- Gaze and Head Movement Data ---")
            print(f"Gaze horizontal ratio: {horizontal_ratio}, vertical ratio: {vertical_ratio}")
            print(f"Gaze direction: {gaze_direction}")

            # Print head movement values
            print(f"Nose tip position: {nose_tip}")
            print(f"Deviation from center (pixels): x={x_deviation}, y={y_deviation}")
            print(f"Deviation from center (%): x={x_deviation_percent*100:.2f}%, y={y_deviation_percent*100:.2f}%")
            print(f"Head movement detected: {head_movement}")
            print(f"-----------------------------------\n")
            last_print_time = current_time

    else:
        head_movement = "c"
        text = "No face detected"
        current_time = time.time()
        if current_time - last_print_time >= 1.0:
            print("No face detected.")
            last_print_time = current_time

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
    cv2.imshow("Calibrated Head and Gaze Tracking", frame)

    if cv2.waitKey(10) == 27:  # Adjust wait time to ensure OpenCV can refresh the window
        logging.info("Exiting tracking loop.")
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
logging.info("Resources released and program terminated.")
