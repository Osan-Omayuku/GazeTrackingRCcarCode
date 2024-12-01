import cv2

# Open webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Webcam not detected or failed to initialize.")
    exit()

while True:
    ret, frame = webcam.read()
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow("Webcam Test", frame)

    # Exit on pressing ESC
    if cv2.waitKey(10) == 27:
        break

webcam.release()
cv2.destroyAllWindows()
