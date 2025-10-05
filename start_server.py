#!/usr/bin/env python3
"""
Start the Enhanced MDAR rPPG Server with Live Webcam
"""
import subprocess
import sys
import os

def main():
    print("🚀 Starting Enhanced MDAR rPPG Server...")
    print("📍 Server will be available at: http://127.0.0.1:8000")
    print("📱 Live webcam at: http://127.0.0.1:8000/live")
    print("📊 File upload at: http://127.0.0.1:8000/")
    print("💚 Health check at: http://127.0.0.1:8000/health")
    print("")
    print("🔴 Press Ctrl+C to stop the server")
    print("")
    
    try:
        # Run the server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "server:app", 
            "--host", "127.0.0.1", 
            "--port", "8000", 
            "--reload"
        ], cwd=os.path.dirname(__file__))
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        print("\nTry running manually with:")
        print("python -m uvicorn server:app --host 127.0.0.1 --port 8000")

if __name__ == "__main__":
    main()
