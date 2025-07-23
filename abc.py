import cv2
from ultralytics import YOLO
from playsound import playsound
import threading

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Replace with 'yolov8s.pt' for better accuracy

# Threshold for crowd alert
CROWD_LIMIT = 10
ALERT_PLAYING = False

def play_alert():
    global ALERT_PLAYING
    ALERT_PLAYING = True
    playsound('alert.mp3')  # Make sure alert.mp3 is in the same directory
    ALERT_PLAYING = False

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model.predict(frame, classes=[0], conf=0.5)  # Class 0 = person

    # Draw boxes
    crowd_count = 0
    for r in results:
        boxes = r.boxes.xyxy
        for box in boxes:
            crowd_count += 1
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display count
    cv2.putText(frame, f"People Detected: {crowd_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Alert if limit exceeded
    if crowd_count > CROWD_LIMIT and not ALERT_PLAYING:
        threading.Thread(target=play_alert).start()
        cv2.putText(frame, "⚠️ Crowd Limit Exceeded!", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Show output
    cv2.imshow("Crowd Monitor", frame)

    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
