import cv2
import torch
import numpy as np
import argparse
import threading
import queue
import time
from numpy.linalg import norm
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def iou(box1, box2):
    """Calculate Intersection over Union (IoU) for two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    
    return inter_area / (box1_area + box2_area - inter_area + 1e-6)

parser = argparse.ArgumentParser(description='Multi-camera tracking system')
parser.add_argument('--enable-gpu', action='store_true', help='Enable GPU acceleration')
parser.add_argument('--camera-sources', nargs='+', type=str, default=[], help='List of camera sources (USB, IP, RTSP)')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() and args.enable_gpu else 'cpu')
print(f"🖥️ Using device: {device}")

yolo_model = YOLO("yolov8n.pt").to(device)
tracker = DeepSort(max_age=15, n_init=2, nn_budget=100, embedder="mobilenet", 
                   embedder_gpu=True if device.type == 'cuda' else False)

selected_person_features = None
selected_track_id = None
tracking_enabled = True
last_seen_features = {}

frame_queues = {}
result_queues = {}

caps = {}
if not args.camera_sources:
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            caps[f"local_{i}"] = cap
            frame_queues[f"local_{i}"] = queue.Queue(maxsize=1)
            result_queues[f"local_{i}"] = queue.Queue(maxsize=1)
        else:
            cap.release()
else:
    for i, source in enumerate(args.camera_sources):
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            caps[f"camera_{i}"] = cap
            frame_queues[f"camera_{i}"] = queue.Queue(maxsize=1)
            result_queues[f"camera_{i}"] = queue.Queue(maxsize=1)
        else:
            print(f"❌ Failed to open camera source: {source}")

def process_frames(source_id):
    global selected_person_features, selected_track_id, tracking_enabled
    
    while True:
        frame = frame_queues[source_id].get()
        results = yolo_model(frame)[0]
        detections = []
        
        for box in results.boxes:
            xyxy = box.xyxy.cpu().numpy()[0]
            conf = float(box.conf.cpu().numpy()[0])
            cls = int(box.cls.cpu().numpy()[0])
            
            if cls == 0 and conf > 0.35:
                detections.append((xyxy.tolist(), conf, cls, None))
        
        tracked_objects = tracker.update_tracks(detections, frame=frame)
        
        processed_tracks = []
        for track in tracked_objects:
            if not track.is_confirmed():
                continue
            
            bbox = track.to_tlwh()
            track_data = {
                'bbox': bbox,
                'track_id': track.track_id,
                'features': track.features[-1] if track.features else None,
                'is_match': False,
                'similarity': 0.0
            }
            
            if selected_person_features is not None and track_data['features'] is not None:
                similarity = np.dot(selected_person_features, track_data['features']) / (
                    norm(selected_person_features) * norm(track_data['features']) + 1e-6)
                track_data['similarity'] = similarity
                
                if similarity > 0.75:
                    track_data['is_match'] = True
                    selected_track_id = track_data['track_id']
                    selected_person_features = track_data['features']
            
            processed_tracks.append(track_data)
        
        result_queues[source_id].put(processed_tracks)

def select_person(event, x, y, flags, param):
    global selected_track_id, selected_person_features, tracking_enabled
    
    if event == cv2.EVENT_LBUTTONDOWN:
        tracks = param['tracks']
        for track in tracks:
            x1, y1, w, h = map(int, track['bbox'])
            if x1 <= x <= x1 + w and y1 <= y <= y1 + h:
                selected_track_id = track['track_id']
                selected_person_features = track.get('features')
                tracking_enabled = True
                
                if selected_person_features is not None:
                    selected_person_features /= (norm(selected_person_features) + 1e-6)
                print(f"🎯 Tracking Person ID: {selected_track_id}")
                break

for source_id in caps.keys():
    threading.Thread(target=process_frames, args=(source_id,), daemon=True).start()
    cv2.namedWindow(source_id)
    cv2.setMouseCallback(source_id, select_person, {'tracks': []})

while True:
    for source_id, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            continue
        if not frame_queues[source_id].full():
            frame_queues[source_id].put(frame)
        
        if not result_queues[source_id].empty():
            tracks = result_queues[source_id].get()
            for track in tracks:
                if selected_track_id is not None and track['track_id'] != selected_track_id:
                    continue
                
                x, y, w, h = map(int, track['bbox'])
                color = (0, 255, 0)
                thickness = 3
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
                cv2.putText(frame, f"ID: {track['track_id']} ({track.get('similarity', 0):.2f})", 
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.imshow(source_id, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps.values():
    cap.release()
cv2.destroyAllWindows()
