import cv2
import torch
import numpy as np
from numpy.linalg import norm
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# Initialize DeepSORT tracker for each camera
trackers = {}

# Store attributes of selected person
selected_person_features = None
selected_track_id = None

# Automatically detect available cameras
def get_available_cameras(max_cams=10):
    available_cams = []
    for i in range(max_cams):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cams.append(i)
            cap.release()
    return available_cams

# Get all available cameras
camera_sources = get_available_cameras()
print(f"✅ Detected Cameras: {camera_sources}")

if not camera_sources:
    print("❌ No cameras detected! Exiting...")
    exit()

# Open video streams and create trackers
caps = {}
for cam_id in camera_sources:
    caps[cam_id] = cv2.VideoCapture(cam_id)
    trackers[cam_id] = DeepSort(max_age=1, embedder="mobilenet",embedder_gpu=True)  # Use appearance embeddings

# Mouse click event to select a person
def select_person(event, x, y, flags, param):
    global selected_track_id, selected_person_features
    if event == cv2.EVENT_LBUTTONDOWN:
        for cam_id, tracker in trackers.items():
            for track in tracker.tracker.tracks:
                if track.is_confirmed():
                    x1, y1, w, h = map(int, track.to_tlwh())
                    if x1 <= x <= x1 + w and y1 <= y <= y1 + h:
                        selected_track_id = track.track_id
                        selected_person_features = np.mean(track.features, axis=0)  # Average features
                        print(f"🎯 Selected Person ID: {selected_track_id}")

# Set mouse callback for each camera window
for cam_id in camera_sources:
    cv2.namedWindow(f"Camera {cam_id}")
    cv2.setMouseCallback(f"Camera {cam_id}", select_person)

while True:
    for cam_id, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            print(f"⚠ Warning: No frame from Camera {cam_id}")
            continue

        # Run YOLO detection on each frame
        results = yolo_model(frame)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Only detect people (Class ID = 0)
            if cls == 0 and conf > 0.5:
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls, None))

        # Update the tracker for this specific camera
        tracked_objects = trackers[cam_id].update_tracks(detections, frame=frame)

        # Keep tracking the selected person even across different cameras
        for track in tracked_objects:
            if not track.is_confirmed():
                continue
            x1, y1, w, h = map(int, track.to_tlwh())
            track_id = track.track_id

            # Ensure selected_person_features is valid
            if selected_person_features is not None and track.features is not None:
                current_features = np.mean(track.features, axis=0)  # Get feature vector

                # Compute cosine similarity
                similarity = np.dot(selected_person_features, current_features) / (norm(selected_person_features) * norm(current_features) + 1e-6)

                # If similarity is high, continue tracking across cameras
                if similarity > 0.7:  # Adjust threshold for better accuracy
                    selected_track_id = track_id

            # Only draw box for the selected person
            if track_id == selected_track_id:
                cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Person under surveillance", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display each camera feed
        cv2.imshow(f"Camera {cam_id}", frame)

    # Reset selection when 'r' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        selected_track_id = None
        selected_person_features = None
        print("🔄 Reset tracking. Click on a new person to track.")

# Release resources
for cap in caps.values():
    cap.release()
cv2.destroyAllWindows()