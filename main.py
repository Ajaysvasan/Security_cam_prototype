import cv2
import torch
import numpy as np
import argparse
from numpy.linalg import norm
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Parse command line arguments
parser = argparse.ArgumentParser(description='Multi-camera tracking system with IP camera support')
parser.add_argument('--ip-cameras', nargs='+', type=str, default=[],
                    help='IP camera URLs (e.g., rtsp://username:password@192.168.1.64:554/Streaming/Channels/1)')
args = parser.parse_args()

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# Initialize DeepSORT tracker for each camera
trackers = {}

# Store attributes of selected person
selected_person_features = None
selected_track_id = None

# Automatically detect available local cameras
def get_available_cameras(max_cams=10):
    available_cams = []
    for i in range(max_cams):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cams.append(i)
            cap.release()
    return available_cams

# Get all available local cameras
camera_sources = get_available_cameras()
print(f"✅ Detected Local Cameras: {camera_sources}")

# Add IP cameras from command line arguments
ip_cameras = args.ip_cameras
print(f"✅ Added IP Cameras: {len(ip_cameras)}")

# Create a mapping of all camera sources
all_sources = {}
# Add local cameras
for i, cam_id in enumerate(camera_sources):
    all_sources[f"local_{cam_id}"] = cam_id

# Add IP cameras
for i, ip_cam in enumerate(ip_cameras):
    all_sources[f"ip_{i}"] = ip_cam

if not all_sources:
    print("❌ No cameras detected! Exiting...")
    exit()

# Function to test if an IP camera stream is accessible
def test_ip_camera(url):
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        return ret
    return False

# Open video streams and create trackers
caps = {}
for source_id, source in all_sources.items():
    # For local cameras, use the camera ID directly
    if source_id.startswith("local_"):
        caps[source_id] = cv2.VideoCapture(source)
        if caps[source_id].isOpened():
            trackers[source_id] = DeepSort(max_age=1, embedder="mobilenet", embedder_gpu=True)
        else:
            print(f"❌ Failed to open local camera {source}")
            del caps[source_id]
    # For IP cameras, use the URL
    elif source_id.startswith("ip_"):
        if test_ip_camera(source):
            caps[source_id] = cv2.VideoCapture(source)
            trackers[source_id] = DeepSort(max_age=1, embedder="mobilenet", embedder_gpu=True)
            print(f"✅ Successfully connected to IP camera: {source}")
        else:
            print(f"❌ Failed to connect to IP camera: {source}")

if not caps:
    print("❌ No cameras could be opened! Exiting...")
    exit()

# Mouse click event to select a person
def select_person(event, x, y, flags, param):
    global selected_track_id, selected_person_features
    source_id = param['source_id']
    if event == cv2.EVENT_LBUTTONDOWN:
        tracker = trackers.get(source_id)
        if tracker:
            for track in tracker.tracker.tracks:
                if track.is_confirmed():
                    x1, y1, w, h = map(int, track.to_tlwh())
                    if x1 <= x <= x1 + w and y1 <= y <= y1 + h:
                        selected_track_id = track.track_id
                        selected_person_features = np.mean(track.features, axis=0)  # Average features
                        print(f"🎯 Selected Person ID: {selected_track_id} from {source_id}")

# Set mouse callback for each camera window
for source_id in caps.keys():
    window_name = f"Camera {source_id}"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, select_person, {'source_id': source_id})

# Function to add a new IP camera during runtime
def add_ip_camera(url):
    if test_ip_camera(url):
        source_id = f"ip_{len([k for k in all_sources.keys() if k.startswith('ip_')])}"
        all_sources[source_id] = url
        caps[source_id] = cv2.VideoCapture(url)
        trackers[source_id] = DeepSort(max_age=1, embedder="mobilenet", embedder_gpu=True)
        
        # Create window and set mouse callback
        window_name = f"Camera {source_id}"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, select_person, {'source_id': source_id})
        
        print(f"✅ Added new IP camera: {url} as {source_id}")
        return True
    else:
        print(f"❌ Failed to connect to IP camera: {url}")
        return False

print("\n🔍 Controls:")
print("- Press 'q' to quit")
print("- Press 'r' to reset person tracking")
print("- Press 'a' to add a new IP camera (enter URL in console)")
print("- Click on a person to track them\n")

while True:
    for source_id, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            print(f"⚠ Warning: No frame from Camera {source_id}")
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
        tracked_objects = trackers[source_id].update_tracks(detections, frame=frame)

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

            # Draw boxes for all detected people, with highlight for selected person
            if track_id == selected_track_id:
                cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Person under surveillance", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 1)

        # Display camera info on frame
        cv2.putText(frame, f"Source: {source_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display each camera feed
        cv2.imshow(f"Camera {source_id}", frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        selected_track_id = None
        selected_person_features = None
        print("🔄 Reset tracking. Click on a new person to track.")
    elif key == ord('a'):
        # Add a new IP camera
        print("\n📷 Enter IP camera URL (e.g., rtsp://user:pass@192.168.1.64:554/1):")
        url = input().strip()
        add_ip_camera(url)

# Release resources
for cap in caps.values():
    cap.release()
cv2.destroyAllWindows()