import './index.css';  // Add this line
import React, { useState, useEffect, useRef } from 'react';

import React, { useState, useEffect, useRef } from 'react';
import { Camera, Target } from 'lucide-react';

const CameraTracker = () => {
  const [cameras, setCameras] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [trackingActive, setTrackingActive] = useState(false);
  const websocketRefs = useRef({});

  const addCamera = () => {
    const newCameraUrl = prompt("Enter Camera IP/URL:");
    if (newCameraUrl) {
      setCameras(prev => [...prev, newCameraUrl]);
    }
  };

  const startTracking = (cameraUrl) => {
    setSelectedCamera(cameraUrl);
    setTrackingActive(true);

    // Close existing websocket if open
    if (websocketRefs.current[cameraUrl]) {
      websocketRefs.current[cameraUrl].close();
    }

    // Create new websocket connection
    const ws = new WebSocket(`ws://localhost:8000/track/${encodeURIComponent(cameraUrl)}`);
    
    ws.onopen = () => console.log(`Connected to camera: ${cameraUrl}`);
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      // Process tracking data here
      console.log("Tracking data:", data);
    };

    ws.onerror = (error) => {
      console.error(`WebSocket error for ${cameraUrl}:`, error);
    };

    websocketRefs.current[cameraUrl] = ws;
  };

  const stopTracking = () => {
    setTrackingActive(false);
    setSelectedCamera(null);
    
    // Close all websocket connections
    Object.values(websocketRefs.current).forEach(ws => ws.close());
    websocketRefs.current = {};
  };

  const selectPersonToTrack = async (cameraUrl, trackId, features) => {
    try {
      const response = await fetch('http://localhost:8000/select-person', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ camera_url: cameraUrl, track_id: trackId, features })
      });
      
      const result = await response.json();
      console.log(result);
    } catch (error) {
      console.error('Error selecting person:', error);
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Multi-Camera Person Tracking</h1>
      
      <div className="flex gap-4 mb-4">
        <button 
          onClick={addCamera}
          className="bg-blue-500 text-white px-4 py-2 rounded flex items-center"
        >
          <Camera className="mr-2" /> Add Camera
        </button>
        
        {selectedCamera ? (
          <button 
            onClick={stopTracking}
            className="bg-red-500 text-white px-4 py-2 rounded flex items-center"
          >
            Stop Tracking
          </button>
        ) : null}
      </div>

      <div className="grid grid-cols-2 gap-4">
        {cameras.map((cameraUrl, index) => (
          <div 
            key={index} 
            className="border p-4 rounded shadow-md"
          >
            <h2 className="text-xl font-semibold mb-2">Camera {index + 1}</h2>
            <p className="mb-2 text-gray-600">{cameraUrl}</p>
            
            <div className="flex gap-2">
              <button 
                onClick={() => startTracking(cameraUrl)}
                className="bg-green-500 text-white px-3 py-1 rounded flex items-center"
                disabled={trackingActive}
              >
                <Target className="mr-2" /> Start Tracking
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default CameraTracker;