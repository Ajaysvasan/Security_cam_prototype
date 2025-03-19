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

# Parse command line arguments
parser = argparse.ArgumentParser(description='Multi-camera face tracking system with IP camera support')
parser.add_argument('--ip-cameras', nargs='+', type=str, default=[],
                    help='IP camera URLs (e.g., rtsp://username:password@192.168.1.64:554/Streaming/Channels/1)')
parser.add_argument('--buffer-size', type=int, default=4,
                    help='Buffer size for frame preprocessing (higher values reduce latency)')
parser.add_argument('--similarity-threshold', type=float, default=0.7,
                    help='Similarity threshold for face matching (0.0-1.0)')
parser.add_argument('--window-width', type=int, default=480,
                    help='Width of display window')
parser.add_argument('--window-height', type=int, default=360,
                    help='Height of display window')
args = parser.parse_args()

# Load YOLOv8 model - use a smaller model for faster processing
yolo_model = YOLO("yolov8n.pt")

# Frame processing queues and threads
frame_queues = {}
result_queues = {}
processing_threads = {}

# Store features of selected person
selected_person_features = None
selected_track_id = None
tracking_enabled = False

# Configuration
DETECTION_CONFIDENCE = 0.5  # Confidence threshold for detections
MAX_FRAME_WIDTH = 640  # Resize frames for faster processing
DETECTION_INTERVAL = 0.1  # Time in seconds between detections
FACE_DETECTION_INTERVAL = 0.1  # Time in seconds between face detections

# Load pretrained face detection model - using a more reliable model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Check if face cascade loaded successfully and download if needed
if face_cascade.empty():
    print("⚠️ Face cascade not found. Downloading...")
    import urllib.request
    import os
    
    # Create directory if it doesn't exist
    cascade_dir = os.path.dirname(cv2.data.haarcascades)
    if not os.path.exists(cascade_dir):
        os.makedirs(cascade_dir)
    
    # Download the face cascade file
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    cascade_path = os.path.join(cascade_dir, 'haarcascade_frontalface_default.xml')
    urllib.request.urlretrieve(url, cascade_path)
    
    # Load again
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("❌ Failed to load face cascade. Using regular person detection.")
        USE_FACE_DETECTION = False
    else:
        print("✅ Face cascade successfully downloaded and loaded.")
        USE_FACE_DETECTION = True
else:
    print("✅ Face cascade loaded successfully.")
    USE_FACE_DETECTION = True

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

# Function to test if an IP camera stream is accessible with optimized settings
def test_ip_camera(url):
    try:
        # Try with RTSP-specific options for lower latency
        cap = cv2.VideoCapture(url)
        # Set buffer size to minimize latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            return ret
        return False
    except Exception as e:
        print(f"Error testing camera {url}: {e}")
        return False

# Process frames in a separate thread to reduce latency
def process_frames(source_id, input_queue, output_queue):
    tracker = DeepSort(max_age=30, n_init=1, embedder="mobilenet", embedder_gpu=True)
    frame_count = 0
    last_detection_time = time.time()
    last_face_detection_time = time.time()
    
    while True:
        try:
            # Get frame from queue with timeout
            try:
                frame = input_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            if frame is None:  # Signal to exit
                break
                
            frame_count += 1
            current_time = time.time()
            
            # Only run detection every DETECTION_INTERVAL seconds to improve performance
            if current_time - last_detection_time >= DETECTION_INTERVAL:
                # Resize frame for faster processing
                orig_h, orig_w = frame.shape[:2]
                scale_factor = min(1.0, MAX_FRAME_WIDTH / orig_w)
                if scale_factor < 1.0:
                    width = int(orig_w * scale_factor)
                    height = int(orig_h * scale_factor)
                    frame_resized = cv2.resize(frame, (width, height))
                else:
                    frame_resized = frame
                    
                detections = []
                
                # Run face detection if enabled and it's time
                if USE_FACE_DETECTION and current_time - last_face_detection_time >= FACE_DETECTION_INTERVAL:
                    # Use OpenCV's built-in face detector
                    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                    # Improve face detection by adjusting parameters
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(20, 20),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    
                    for (x, y, w, h) in faces:
                        # Scale back to original frame size if resized
                        if scale_factor < 1.0:
                            x = int(x / scale_factor)
                            y = int(y / scale_factor)
                            w = int(w / scale_factor)
                            h = int(h / scale_factor)
                        
                        # Add face detection as a person detection
                        # Convert to XYXY format for compatibility with DeepSORT
                        detections.append(([x, y, x + w, y + h], 1.0, 0, None))
                    
                    last_face_detection_time = current_time
                
                # Always run person detection as a fallback
                results = yolo_model(frame_resized)[0]
                for box in results.boxes:
                    if len(box.xyxy) > 0:  # Ensure box has coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        
                        # Only detect people (Class ID = 0)
                        if cls == 0 and conf > DETECTION_CONFIDENCE:
                            # Scale back to original frame size if resized
                            if scale_factor < 1.0:
                                x1 = int(x1 / scale_factor)
                                y1 = int(y1 / scale_factor)
                                x2 = int(x2 / scale_factor)
                                y2 = int(y2 / scale_factor)
                            
                            # Add person detection in XYXY format
                            detections.append(([x1, y1, x2, y2], conf, cls, None))
                
                # Update the tracker with bounding boxes in the correct format
                tracked_objects = tracker.update_tracks(detections, frame=frame)
                last_detection_time = current_time
                
                # Process results and check for face matches
                result_with_tracks = []
                for track in tracked_objects:
                    if not track.is_confirmed():
                        continue
                    
                    track_data = {
                        'bbox': track.to_tlwh(),  # Convert to x,y,w,h format
                        'track_id': track.track_id,
                        'features': track.features[-1] if track.features is not None and len(track.features) > 0 else None,
                        'is_match': False,
                        'similarity': 0.0
                    }
                    
                    # If tracking is enabled and we have a selected person, check for matches
                    if selected_person_features is not None and tracking_enabled and track_data['features'] is not None:
                        # Compute cosine similarity with selected person features
                        similarity = np.dot(selected_person_features, track_data['features']) / (
                            norm(selected_person_features) * norm(track_data['features']) + 1e-6)
                        
                        # Only mark as match if similarity exceeds threshold
                        if similarity > args.similarity_threshold:
                            track_data['is_match'] = True
                            track_data['similarity'] = similarity
                    
                    result_with_tracks.append(track_data)
                
                output_queue.put((frame.copy(), result_with_tracks))
            else:
                # If skipping detection, still return frame without tracks
                output_queue.put((frame.copy(), []))
                
            input_queue.task_done()
        except Exception as e:
            print(f"Error in processing thread for {source_id}: {e}")
            continue

# Open video streams and create processing threads
caps = {}
for source_id, source in all_sources.items():
    # Create queues for frame processing
    frame_queues[source_id] = queue.Queue(maxsize=args.buffer_size)
    result_queues[source_id] = queue.Queue(maxsize=args.buffer_size)
    
    # For local cameras, use the camera ID directly
    if source_id.startswith("local_"):
        caps[source_id] = cv2.VideoCapture(source)
        if caps[source_id].isOpened():
            # Set camera to low resolution for faster processing
            caps[source_id].set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            caps[source_id].set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            caps[source_id].set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for lower latency
            
            # Start processing thread
            processing_threads[source_id] = threading.Thread(
                target=process_frames,
                args=(source_id, frame_queues[source_id], result_queues[source_id]),
                daemon=True
            )
            processing_threads[source_id].start()
        else:
            print(f"❌ Failed to open local camera {source}")
            del caps[source_id]
    # For IP cameras, use the URL
    elif source_id.startswith("ip_"):
        if test_ip_camera(source):
            caps[source_id] = cv2.VideoCapture(source)
            
            # Optimize RTSP connection for real-time processing
            if source.startswith('rtsp'):
                caps[source_id].set(cv2.CAP_PROP_BUFFERSIZE, 1)
                caps[source_id].set(cv2.CAP_PROP_RTSP_TRANSPORT, cv2.CAP_RTSP_TCP)
                
            # Start processing thread
            processing_threads[source_id] = threading.Thread(
                target=process_frames,
                args=(source_id, frame_queues[source_id], result_queues[source_id]),
                daemon=True
            )
            processing_threads[source_id].start()
            print(f"✅ Successfully connected to IP camera: {source}")
        else:
            print(f"❌ Failed to connect to IP camera: {source}")

if not caps:
    print("❌ No cameras could be opened! Exiting...")
    exit()

# Mouse click event to select a person
def select_person(event, x, y, flags, param):
    global selected_track_id, selected_person_features, tracking_enabled
    
    source_id = param['source_id']
    frame = param.get('frame')
    tracks = param.get('tracks', [])
    
    if event == cv2.EVENT_LBUTTONDOWN and frame is not None and tracks:
        # Find if clicked on any tracked person
        for track in tracks:
            x1, y1, w, h = map(int, track['bbox'])
            if x1 <= x <= x1 + w and y1 <= y <= y1 + h:
                selected_track_id = track['track_id']
                
                # Store features for tracking
                if track.get('features') is not None:
                    selected_person_features = track.get('features')
                    tracking_enabled = True
                    print(f"🎯 Selected Person ID: {selected_track_id} from {source_id}")
                    break
                else:
                    print(f"⚠️ No features available for this person. Try selecting again.")

# Function to add a new IP camera during runtime
def add_ip_camera(url):
    if test_ip_camera(url):
        source_id = f"ip_{len([k for k in all_sources.keys() if k.startswith('ip_')])}"
        all_sources[source_id] = url
        caps[source_id] = cv2.VideoCapture(url)
        
        # Optimize RTSP settings
        if url.startswith('rtsp'):
            caps[source_id].set(cv2.CAP_PROP_BUFFERSIZE, 1)
            caps[source_id].set(cv2.CAP_PROP_RTSP_TRANSPORT, cv2.CAP_RTSP_TCP)
        
        # Create frame processing queues
        frame_queues[source_id] = queue.Queue(maxsize=args.buffer_size)
        result_queues[source_id] = queue.Queue(maxsize=args.buffer_size)
        
        # Start processing thread
        processing_threads[source_id] = threading.Thread(
            target=process_frames,
            args=(source_id, frame_queues[source_id], result_queues[source_id]),
            daemon=True
        )
        processing_threads[source_id].start()
        
        # Create window and set mouse callback
        window_name = f"Camera {source_id}"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, select_person, {'source_id': source_id, 'frame': None, 'tracks': []})
        
        print(f"✅ Added new IP camera: {url} as {source_id}")
        return True
    else:
        print(f"❌ Failed to connect to IP camera: {url}")
        return False

# Set mouse callback for each camera window with shared frame data
mouse_callback_data = {source_id: {'source_id': source_id, 'frame': None, 'tracks': []} for source_id in caps.keys()}
for source_id in caps.keys():
    window_name = f"Camera {source_id}"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, select_person, mouse_callback_data[source_id])

# Create trackbars for window size control
control_window = "Camera Controls"
cv2.namedWindow(control_window)
cv2.createTrackbar("Window Width", control_window, args.window_width, 1280, lambda x: None)
cv2.createTrackbar("Window Height", control_window, args.window_height, 720, lambda x: None)
cv2.createTrackbar("Match Threshold", control_window, int(args.similarity_threshold * 100), 100, lambda x: None)

print("\n🔍 Controls:")
print("- Press 'q' to quit")
print("- Press 'r' to reset person tracking")
print("- Press 'a' to add a new IP camera (enter URL in console)")
print("- Press 't' to toggle tracking on/off")
print("- Use trackbars to adjust window size and matching threshold")
print("- Click on a person to track them\n")

last_fps_time = time.time()
frame_counts = {source_id: 0 for source_id in caps.keys()}

try:
    while True:
        # Get current window size from trackbars
        display_width = cv2.getTrackbarPos("Window Width", control_window)
        display_height = cv2.getTrackbarPos("Window Height", control_window)
        # Update similarity threshold from trackbar
        args.similarity_threshold = cv2.getTrackbarPos("Match Threshold", control_window) / 100.0
        
        for source_id, cap in list(caps.items()):
            # Read frame and add to processing queue
            ret, frame = cap.read()
            if not ret:
                print(f"⚠ Warning: No frame from Camera {source_id}")
                continue
            
            # If queue is full, remove oldest item
            if frame_queues[source_id].full():
                try:
                    frame_queues[source_id].get_nowait()
                except queue.Empty:
                    pass
                
            # Add frame to queue for processing
            try:
                frame_queues[source_id].put_nowait(frame.copy())
            except queue.Full:
                pass
                
            # Try to get processed results
            try:
                processed_frame, tracks = result_queues[source_id].get_nowait()
                
                # Update mouse callback data with current frame and tracks
                mouse_callback_data[source_id]['frame'] = processed_frame
                mouse_callback_data[source_id]['tracks'] = tracks
                
                # Draw tracking results - only highlight the selected person
                for track in tracks:
                    x1, y1, w, h = map(int, track['bbox'])
                    track_id = track['track_id']
                    is_match = track.get('is_match', False)
                    
                    if is_match and tracking_enabled:
                        # Draw bold green box for the selected person
                        cv2.rectangle(processed_frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                        similarity = track.get('similarity', 0) * 100
                        cv2.putText(processed_frame, f"Selected Person ({similarity:.1f}%)", 
                                   (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    elif not tracking_enabled or selected_person_features is None:
                        # Draw all boxes when tracking is disabled
                        cv2.rectangle(processed_frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 1)
                        cv2.putText(processed_frame, f"ID: {track_id}", 
                                   (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Calculate and display FPS
                frame_counts[source_id] += 1
                if time.time() - last_fps_time >= 1.0:
                    fps = {src: count for src, count in frame_counts.items()}
                    frame_counts = {src: 0 for src in frame_counts.keys()}
                    last_fps_time = time.time()
                    
                    # Display FPS on frame
                    current_fps = fps.get(source_id, 0)
                    cv2.putText(processed_frame, f"FPS: {current_fps}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Display tracking status
                tracking_status = "ON" if tracking_enabled else "OFF"
                cv2.putText(processed_frame, f"Tracking: {tracking_status}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display threshold
                cv2.putText(processed_frame, f"Threshold: {args.similarity_threshold:.2f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display camera info
                cv2.putText(processed_frame, f"Source: {source_id}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Resize the display frame to user-defined size
                if display_width > 0 and display_height > 0:
                    display_frame = cv2.resize(processed_frame, (display_width, display_height))
                else:
                    display_frame = processed_frame
                
                # Display each camera feed
                cv2.imshow(f"Camera {source_id}", display_frame)
                
                result_queues[source_id].task_done()
            except queue.Empty:
                pass
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            selected_track_id = None
            selected_person_features = None
            tracking_enabled = False
            print("🔄 Reset tracking. Click on a new person to track.")
        elif key == ord('t'):
            tracking_enabled = not tracking_enabled
            status = "enabled" if tracking_enabled else "disabled"
            print(f"🔄 Tracking {status}")
        elif key == ord('a'):
            # Add a new IP camera
            print("\n📷 Enter IP camera URL (e.g., rtsp://user:pass@192.168.1.64:554/1):")
            url = input().strip()
            add_ip_camera(url)

except KeyboardInterrupt:
    print("\n👋 Shutting down gracefully...")

finally:
    # Stop all processing threads
    for source_id in processing_threads:
        try:
            frame_queues[source_id].put(None)  # Signal thread to exit
            processing_threads[source_id].join(timeout=1.0)
        except:
            pass
        
    # Release resources
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()