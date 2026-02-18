# 🚗 Vehicle Detection, Tracking & Analytics

A real-time vehicle detection and counting system built with **YOLOv8**, **SORT tracker**, and **OpenCV**. Detects and tracks cars, motorcycles, buses, and trucks in video footage, logs every unique vehicle to a SQLite database, and displays live annotated output.

---

## 📁 Project Structure

```
├── main.py              # Main application script
├── sort.py              # SORT tracker implementation (Kalman Filter + Hungarian Algorithm)
├── yolov8n.pt           # YOLOv8 nano pretrained model weights
├── vehicle_data.db      # SQLite database (auto-created on first run)
├── requirements.txt     # Python dependencies
└── video.mp4            # Input video file
```

---

## ✨ Features

- **Real-time object detection** using YOLOv8 (COCO-pretrained)
- **Multi-object tracking** with the SORT algorithm (Kalman Filter + IoU-based Hungarian matching)
- **Vehicle counting** — each unique vehicle is counted exactly once
- **SQLite logging** — vehicle type and timestamp stored for every detected vehicle
- **Live display** — bounding boxes and vehicle type labels rendered on video frames

---

## 🚦 Supported Vehicle Types

| COCO Class ID | Vehicle Type |
|---------------|--------------|
| 2             | Car          |
| 3             | Motorcycle   |
| 5             | Bus          |
| 7             | Truck        |

---

## 🛠️ Installation

### 1. Clone or download the project

```bash
git clone <your-repo-url>
cd vehicle-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
ultralytics
opencv-python
numpy
scipy
filterpy
```

---

## ⚙️ Configuration

Before running, update the video path in `main.py`:

```python
# Line ~40 in main.py
video_path = r"C:\path\to\your\video.mp4"
```

Replace with the actual path to your input video file.

---

## ▶️ Usage

```bash
python main.py
```

- A window titled **"Vehicle Detection, Tracking & Analytics"** will open showing the annotated video.
- Press **`Q`** to quit at any time.
- Detection results are saved automatically to `vehicle_data.db`.

---

## 🗄️ Database Schema

Results are stored in a local SQLite database (`vehicle_data.db`):

```sql
CREATE TABLE vehicles (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id   INTEGER,   -- Unique tracker-assigned ID
    vehicle_type TEXT,      -- Car / Motorcycle / Bus / Truck
    timestamp    TEXT       -- Format: YYYY-MM-DD HH:MM:SS
);
```

You can query the data with any SQLite client or Python:

```python
import sqlite3
conn = sqlite3.connect("vehicle_data.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM vehicles")
print(cursor.fetchall())
```

---

## 🧠 How It Works

1. **Detection** — Each video frame is passed through YOLOv8. Detections with confidence > 0.4 and belonging to vehicle classes are extracted as bounding boxes.
2. **Tracking** — Bounding boxes are fed into the SORT tracker, which uses a Kalman Filter to predict object positions and the Hungarian algorithm to match detections across frames. Each tracked object receives a unique `track_id`.
3. **Counting** — When a `track_id` is seen for the first time, the vehicle is counted and its record is inserted into the database.
4. **Display** — Bounding boxes and vehicle type labels are drawn on each frame and shown in real time.

---

## ⚠️ Known Limitations

- Vehicle type assignment uses the first detection in each frame, which may cause occasional misclassification when multiple vehicle types are present simultaneously. A more robust approach would match each track to its closest detection by bounding box overlap.
- The system requires a display environment (OpenCV `imshow`); it will not run headlessly without modification.
- Performance depends on hardware — a GPU is recommended for smooth real-time inference.

---

## 📦 Dependencies

| Package       | Purpose                          |
|---------------|----------------------------------|
| ultralytics   | YOLOv8 model inference           |
| opencv-python | Video I/O and frame rendering    |
| numpy         | Array operations                 |
| scipy         | Hungarian algorithm (SORT)       |
| filterpy      | Kalman Filter (SORT)             |

---

## 📄 License

This project is for educational and research purposes. YOLOv8 weights are subject to [Ultralytics licensing terms](https://github.com/ultralytics/ultralytics).
