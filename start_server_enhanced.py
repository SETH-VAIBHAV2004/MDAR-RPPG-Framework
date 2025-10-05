#!/usr/bin/env python3
"""
Enhanced server startup script with support for large file uploads (>2GB)
This script configures uvicorn to handle large files properly.
"""

import sys
import os
import uvicorn
from pathlib import Path

def main():
    """Start the enhanced RPPG server with optimized settings for large file uploads."""
    
    # Add the current directory to the Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # Create required directories
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True) 
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("static/js", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    
    print("🚀 Starting Enhanced RPPG Server...")
    print("📁 File upload limit: 5GB")
    print("📹 Webcam integration: Enabled")
    print("🔗 Server will be available at: http://127.0.0.1:8000")
    print("\nPages:")
    print("  - Main Interface: http://127.0.0.1:8000")
    print("  - Native Webcam: http://127.0.0.1:8000/webcam") 
    print("  - Live Stream: http://127.0.0.1:8000/live")
    print("  - Health Check: http://127.0.0.1:8000/health")
    print("\nAPI Endpoints:")
    print("  - Video Upload: POST /api/video/upload")
    print("  - Camera List: GET /api/cameras/list")
    print("  - Live WebSocket: ws://127.0.0.1:8000/ws/live")
    print("\nPress Ctrl+C to stop the server\n")
    
    # Enhanced uvicorn configuration for large file uploads
    config = uvicorn.Config(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info",
        # Increase limits for large file uploads
        limit_concurrency=1000,
        limit_max_requests=1000,
        timeout_keep_alive=30,  # Increased keep-alive timeout
        # WebSocket settings
        ws_ping_interval=20,
        ws_ping_timeout=20,
        # Additional server settings
        access_log=True,
        use_colors=True,
    )
    
    # Custom configuration for large file handling
    # Note: FastAPI/Starlette handles request body size limits internally
    # The actual limit is set in the middleware we added to server.py
    
    server = uvicorn.Server(config)
    
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
