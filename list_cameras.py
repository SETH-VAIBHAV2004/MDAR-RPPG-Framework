#!/usr/bin/env python3
"""
Quick utility to list available cameras.
"""

import cv2


def list_cameras():
    """List all available cameras."""
    print("Scanning for available cameras...\n")
    
    available = []
    
    # Test up to 10 camera indices
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to get camera properties
            backend = cap.getBackendName()
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Test if we can actually read a frame
            ret, frame = cap.read()
            if ret:
                available.append({
                    'id': i,
                    'backend': backend,
                    'resolution': f"{width}x{height}",
                    'fps': fps,
                    'working': True
                })
            
            cap.release()
    
    if not available:
        print("No cameras found!")
        return
    
    print(f"Found {len(available)} camera(s):\n")
    
    for cam in available:
        print(f"Camera {cam['id']}:")
        print(f"  Backend: {cam['backend']}")
        print(f"  Resolution: {cam['resolution']}")
        print(f"  FPS: {cam['fps']:.1f}")
        print()
    
    print("Usage examples:")
    print(f"  python live_heartbeat.py --camera {available[0]['id']}  # Use first camera")
    if len(available) > 1:
        print(f"  python live_heartbeat.py --camera {available[-1]['id']}  # Use external camera")
    print("  python live_heartbeat.py  # Interactive selection")


if __name__ == '__main__':
    list_cameras()
