#!/usr/bin/env python3
"""
Enhanced MDAR rPPG Server - Main Launcher
==========================================

This is the main entry point for the integrated rPPG heart rate detection server.
Features:
- File upload and processing (NPZ chunks, raw videos)
- Live webcam heart rate detection
- Model-based predictions using MDAR
- Traditional POS/CHROM methods for comparison
- Real-time web interface with WebSocket streaming
- Native OpenCV webcam integration

Usage:
    python main.py [--port PORT] [--host HOST] [--debug]

Author: rPPG Project
"""

import os
import sys
import argparse
import asyncio
import threading
import subprocess
import time
from pathlib import Path

# Add current directory to Python path for imports
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import fastapi
        print(f"✅ FastAPI: {fastapi.__version__}")
    except ImportError:
        missing_deps.append("fastapi")
    
    try:
        import uvicorn
        print(f"✅ Uvicorn available")
    except ImportError:
        missing_deps.append("uvicorn")
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError:
        missing_deps.append("numpy")
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Please install them using: pip install " + " ".join(missing_deps))
        return False
    
    return True


def check_model_files():
    """Check if model files are available."""
    outputs_dir = current_dir / "outputs"
    
    # Check for enhanced model
    enhanced_model = outputs_dir / "mdar_enhanced" / "mdar_best.pth"
    original_model = outputs_dir / "mdar" / "mdar_best.pth"
    
    if enhanced_model.exists():
        print(f"✅ Enhanced MDAR model found: {enhanced_model}")
        return str(enhanced_model)
    elif original_model.exists():
        print(f"✅ Original MDAR model found: {original_model}")
        return str(original_model)
    else:
        print(f"⚠️ No trained models found in {outputs_dir}")
        print("The server will still work with POS/CHROM methods, but MDAR predictions won't be available.")
        print("To train a model, run: python train_mdar.py")
        return None


def setup_directories():
    """Create necessary directories."""
    dirs_to_create = [
        "outputs/server_logs",
        "static/css",
        "static/js",
        "templates"
    ]
    
    for dir_path in dirs_to_create:
        full_path = current_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Directory ready: {dir_path}")


def start_server(host: str = "127.0.0.1", port: int = 8000, debug: bool = False):
    """Start the FastAPI server."""
    try:
        import uvicorn
        from server import app
        
        print(f"\n🚀 Starting Enhanced MDAR rPPG Server...")
        print(f"📍 Server will be available at: http://{host}:{port}")
        print(f"🔗 Web Interface: http://{host}:{port}")
        print(f"🔗 Live Webcam: http://{host}:{port}/live")
        print(f"🔗 Native Webcam: http://{host}:{port}/webcam")
        print(f"🔗 API Health: http://{host}:{port}/health")
        print(f"\n💡 Press Ctrl+C to stop the server")
        print("-" * 60)
        
        # Configure uvicorn
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="info" if debug else "warning",
            access_log=debug,
            reload=debug,  # Auto-reload in debug mode
            reload_dirs=[str(current_dir)] if debug else None
        )
        
        server = uvicorn.Server(config)
        server.run()
        
    except KeyboardInterrupt:
        print(f"\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False
    
    return True


def launch_native_webcam():
    """Launch native webcam processing (live_heartbeat.py)."""
    try:
        live_heartbeat_path = current_dir / "live_heartbeat.py"
        if not live_heartbeat_path.exists():
            print(f"❌ live_heartbeat.py not found at {live_heartbeat_path}")
            return False
        
        print(f"🎥 Launching native webcam heart rate detection...")
        print(f"💡 This will open an OpenCV window with real-time processing")
        
        # Launch live_heartbeat.py as subprocess
        result = subprocess.run([
            sys.executable, 
            str(live_heartbeat_path),
            "--window", "8.0",  # 8 second window
            "--history-length", "16"  # Keep 16 HR estimates
        ], cwd=str(current_dir))
        
        return result.returncode == 0
        
    except KeyboardInterrupt:
        print(f"\n🛑 Native webcam stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error launching native webcam: {e}")
        return False


def print_banner():
    """Print application banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                 Enhanced MDAR rPPG Server                   ║
    ║            Real-time Heart Rate Detection System            ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  🫀 Multiple Detection Methods: MDAR + POS + CHROM          ║
    ║  📹 Live Webcam Processing with WebSocket Streaming        ║
    ║  📁 File Upload Support (Videos, NPZ chunks)               ║
    ║  🎯 Real-time Face Detection & ROI Tracking                ║
    ║  📊 Professional Web Interface with Metrics                ║
    ║  🔧 Native OpenCV Integration                               ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced MDAR rPPG Server - Real-time Heart Rate Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Start server on localhost:8000
  python main.py --port 8080              # Start on port 8080
  python main.py --host 0.0.0.0 --port 80 # Start on all interfaces, port 80
  python main.py --debug                   # Start in debug mode with auto-reload
  python main.py --native-webcam           # Launch native OpenCV webcam only
        """
    )
    
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Host to bind the server to (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to bind the server to (default: 8000)')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode with auto-reload')
    parser.add_argument('--native-webcam', action='store_true',
                       help='Launch native OpenCV webcam only (no web server)')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check dependencies and setup, don\'t start server')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        return 1
    
    print(f"\n🔍 Checking model files...")
    model_path = check_model_files()
    
    print(f"\n📁 Setting up directories...")
    setup_directories()
    
    if args.check_only:
        print(f"\n✅ System check completed successfully!")
        print(f"   Model available: {'Yes' if model_path else 'No (POS/CHROM only)'}")
        print(f"   Ready to launch server!")
        return 0
    
    # Handle native webcam mode
    if args.native_webcam:
        print(f"\n🎥 Native Webcam Mode Selected")
        print("-" * 40)
        success = launch_native_webcam()
        return 0 if success else 1
    
    # Start web server
    print(f"\n🌐 Web Server Mode Selected")
    print("-" * 40)
    success = start_server(args.host, args.port, args.debug)
    return 0 if success else 1


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n🛑 Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
