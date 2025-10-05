# 🫀 Enhanced MDAR rPPG Server

**Real-Time Heart Rate Detection System** powered by advanced remote photoplethysmography (rPPG) techniques.  
Built with **FastAPI, PyTorch, and OpenCV**, this system integrates state-of-the-art methods to provide accurate, real-time heart rate monitoring directly through your webcam or uploaded videos.

![Python](https://img.shields.io/badge/Python-3.8+-blue)  
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)  
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)  
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-red)

---

## 🌟 Key Features
- **Multiple Detection Methods**:  
  - MDAR Neural Network (transformer-based rPPG)  
  - POS (Plane-Orthogonal-to-Skin)  
  - CHROM (Chrominance-based signal)  
  - Ensemble Fusion for boosted accuracy  

- **User Interfaces**:  
  - 📁 File Upload (video & NPZ files)  
  - 📹 Live Webcam Streaming (browser-based)  
  - 🎥 Native OpenCV Window  
  - 🌐 Modern Web Dashboard with real-time updates  

- **Advanced Capabilities**:  
  - Real-time WebSocket streaming  
  - Multi-camera support with auto-switching  
  - ROI tracking with motion artifact detection  
  - Confidence scoring & temporal smoothing  
  - Professional logging & metrics tracking  

---

## 🚀 Quick Start
1. **Launch the Server**  
   ```bash
   # Default start (http://localhost:8000)
   python main.py  

   # Start on custom port
   python main.py --port 8080  

   # Debug mode with auto-reload
   python main.py --debug  

   # Native webcam only
   python main.py --native-webcam  

   # Check dependencies
   python main.py --check-only  
   ```

2. **Open the Web Interface**  
   - Main Interface → `http://localhost:8000`  
   - Live Webcam → `http://localhost:8000/live`  
   - Native Webcam Control → `http://localhost:8000/webcam`  
   - Health Check → `http://localhost:8000/health`  

---

## 📖 Usage Guide
### 🎥 Video Processing
- Upload MP4, AVI, or MOV files  
- Get: HR estimate (BPM), confidence score, frame stats, and reliability assessment  

### 🔬 NPZ File Processing
- Upload preprocessed `.npz` chunks  
- Get instant waveform-based HR predictions  

### 📹 Live Webcam
- Start browser-based or native webcam detection  
- Monitor live BPM, confidence, and method comparison  

---

## 🛠 Installation & Setup
**Requirements**:  
- Python 3.8+  
- Webcam (for live mode)  
- GPU (optional for MDAR acceleration)  

**Dependencies (auto-checked)**:  
PyTorch · OpenCV · FastAPI · Uvicorn · NumPy · SciPy  

**Model Files** (if available):  
- `outputs/mdar_enhanced/mdar_best.pth`  
- `outputs/mdar/mdar_best.pth`  

> Without models, the system still works with POS/CHROM.  

---

## 🎯 Technical Overview
1. Face Detection → ROI extraction (cheek region)  
2. Illumination Normalization (CLAHE)  
3. Signal Extraction (MDAR, POS, CHROM)  
4. Bandpass Filtering (0.7–4.0 Hz)  
5. FFT-based Heart Rate Estimation  
6. Ensemble Fusion with Confidence Weighting  
7. Temporal Smoothing for stability  

**MDAR Neural Network**:  
Transformer-based, 4-channel input (RGB + temporal diff), outputs waveform & BPM with multi-task learning.  

---

## 📊 Performance
- **Accuracy**: >90% in optimal conditions  
- **Speed**: 15–30 FPS on modern hardware  
- **Latency**: <500ms  
- **Range**: 40–200 BPM  

**Best Conditions**: good lighting · 30–100 cm distance · minimal movement  

---

## 🔧 Configuration
**Command-line Options**  
```bash
python main.py --port 8080 --debug --native-webcam
```
**Env Vars**  
- `CUDA_VISIBLE_DEVICES` → control GPU usage  
- `PYTHONPATH` → include project directory  

---

## 🔍 Troubleshooting
- **No Camera Found** → check permissions / close other apps  
- **Model Failed to Load** → check `outputs/` folder & PyTorch install  
- **WebSocket Error** → verify firewall & correct port  
- **Low Confidence** → improve lighting, reduce motion, clean camera  

---

## 🤝 API Endpoints
- `GET /health` → server status  
- `POST /predict_video` → process video file  
- `POST /predict_npz` → process NPZ file  
- `POST /api/webcam/start` / `stop` / `status`  
- `WS /ws/live` → real-time webcam streaming  

---

## 🔮 Roadmap
- Multi-person detection  
- Kalman filtering for smoother HR curves  
- Export results (CSV/JSON)  
- Long-term historical tracking  
- Mobile optimization  
- Docker-based cloud deployment  

---

## 📄 License & Citation
If used in academic work, please cite relevant rPPG methods (MDAR, POS, CHROM).  

---

## 🙋 Support
1. Check docs & troubleshooting  
2. Run diagnostics:  
   ```bash
   python main.py --check-only
   ```  
3. Enable debug logs:  
   ```bash
   python main.py --debug
   ```  
4. Check logs: `outputs/server_logs/`  

---

**Built with ❤️ using FastAPI, PyTorch, and OpenCV**  
*Real-time heart rate detection — accessible, reliable, and powerful.*  
