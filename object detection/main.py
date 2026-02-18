import cv2
import numpy as np
import sqlite3
from datetime import datetime
from ultralytics import YOLO
from sort import Sort

# =========================
# DATABASE SETUP
# =========================
conn = sqlite3.connect("vehicle_data.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS vehicles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id INTEGER,
    vehicle_type TEXT,
    timestamp TEXT
)
""")
conn.commit()

# =========================
# YOLO + SORT SETUP
# =========================
model = YOLO("yolov8n.pt")
tracker = Sort()

# Vehicle class mapping (COCO)
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Counting structures
counted_ids = set()
vehicle_counts = {
    "Car": 0,
    "Motorcycle": 0,
    "Bus": 0,
    "Truck": 0
}

# =========================
# VIDEO INPUT
# =========================
video_path = r"C:\Users\punit\OneDrive\Desktop\object detection\video.mp4.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Video not found")
    exit()

print("✅ Video loaded successfully")

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = []
    detection_classes = []

    results = model(frame, stream=True, verbose=False)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls in VEHICLE_CLASSES and conf > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append([x1, y1, x2, y2])
                detection_classes.append(cls)

    detections = np.array(detections)
    if len(detections) == 0:
        detections = np.empty((0, 4))

    tracks = tracker.update(detections)

    # =========================
    # TRACKING + COUNTING
    # =========================
    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)

        # Assign vehicle type (simple frame-wise)
        vehicle_type = "Unknown"
        if len(detection_classes) > 0:
            vehicle_type = VEHICLE_CLASSES.get(detection_classes[0], "Unknown")

        # Count only once per vehicle
        if track_id not in counted_ids and vehicle_type != "Unknown":
            counted_ids.add(track_id)
            vehicle_counts[vehicle_type] += 1

            cursor.execute(
                "INSERT INTO vehicles (vehicle_id, vehicle_type, timestamp) VALUES (?, ?, ?)",
                (track_id, vehicle_type, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            conn.commit()

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 🔥 ONLY VEHICLE TYPE (NO ID)
        cv2.putText(
            frame,
            f"{vehicle_type}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    

    cv2.imshow("Vehicle Detection, Tracking & Analytics", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
conn.close()
cv2.destroyAllWindows()
