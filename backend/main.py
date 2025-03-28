import cv2
import torch
import numpy as np
from numpy.linalg import norm
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json

class MultiCameraTracker:
    def __init__(self):
        # Load YOLOv8 model
        self.yolo_model = YOLO("yolov8n.pt")
        
        # Trackers for multiple cameras
        self.trackers = {}
        
        # Selected person tracking
        self.selected_person_features = None
        self.selected_track_id = None

    def initialize_tracker(self, camera_url):
        """Initialize tracker for a specific camera URL"""
        tracker = DeepSort(max_age=1, embedder="mobilenet", embedder_gpu=True)
        self.trackers[camera_url] = {
            'tracker': tracker,
            'capture': cv2.VideoCapture(camera_url)
        }

    def select_person(self, camera_url, track_id, features):
        """Select a specific person to track across cameras"""
        self.selected_track_id = track_id
        self.selected_person_features = features
        print(f"🎯 Selected Person ID: {track_id} from {camera_url}")

    def process_frame(self, camera_url):
        """Process a single frame from a camera"""
        camera_info = self.trackers.get(camera_url)
        if not camera_info:
            return None

        cap = camera_info['capture']
        tracker = camera_info['tracker']

        ret, frame = cap.read()
        if not ret:
            return None

        # Run YOLO detection
        results = self.yolo_model(frame)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Only detect people
            if cls == 0 and conf > 0.5:
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls, None))

        # Update tracker
        tracked_objects = tracker.update_tracks(detections, frame=frame)

        # Prepare tracking results
        track_results = []
        for track in tracked_objects:
            if not track.is_confirmed():
                continue

            x1, y1, w, h = map(int, track.to_tlwh())
            track_id = track.track_id

            # Check similarity to selected person
            if (self.selected_person_features is not None and 
                track.features is not None):
                current_features = np.mean(track.features, axis=0)
                similarity = np.dot(self.selected_person_features, current_features) / (
                    norm(self.selected_person_features) * norm(current_features) + 1e-6
                )

                # High similarity tracking
                if similarity > 0.7:
                    self.selected_track_id = track_id

            track_info = {
                'id': track_id,
                'bbox': [x1, y1, w, h],
                'is_selected': track_id == self.selected_track_id
            }
            track_results.append(track_info)

        return track_results

# FastAPI Application
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global tracker instance
multi_tracker = MultiCameraTracker()

@app.websocket("/track/{camera_url}")
async def websocket_tracking(websocket: WebSocket, camera_url: str):
    await websocket.accept()
    
    # Initialize tracker for this camera
    multi_tracker.initialize_tracker(camera_url)

    try:
        while True:
            # Process frame
            track_results = multi_tracker.process_frame(camera_url)
            
            if track_results:
                await websocket.send_json({
                    "camera": camera_url,
                    "tracks": track_results
                })
            
            await asyncio.sleep(0.1)  # Adjust for performance
    except WebSocketDisconnect:
        print(f"Disconnected from camera: {camera_url}")
    finally:
        # Clean up resources
        if camera_url in multi_tracker.trackers:
            multi_tracker.trackers[camera_url]['capture'].release()
            del multi_tracker.trackers[camera_url]

@app.post("/select-person")
async def select_person(data: dict):
    """Endpoint to select a person to track"""
    camera_url = data.get('camera_url')
    track_id = data.get('track_id')
    features = data.get('features')

    if camera_url and track_id and features:
        multi_tracker.select_person(camera_url, track_id, np.array(features))
        return {"status": "success", "message": f"Tracking person {track_id}"}
    
    return {"status": "error", "message": "Invalid selection parameters"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)