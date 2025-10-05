# 🫀 Enhanced MDAR rPPG Server

## Real-time Heart Rate Detection System

A comprehensive web-based heart rate detection system using remote photoplethysmography (rPPG) techniques. This system integrates multiple detection methods including the MDAR neural network, POS (Plane-Orthogonal-to-Skin), and CHROM (Chrominance) algorithms.

![System Overview](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-red)

## 🌟 Features

### Multiple Detection Methods
- **MDAR Neural Network**: Deep learning-based rPPG signal extraction
- **POS Algorithm**: Plane-Orthogonal-to-Skin method for robust HR detection
- **CHROM Algorithm**: Chrominance-based signal processing
- **Ensemble Fusion**: Combines all methods for improved accuracy

### User Interfaces
- **📁 File Upload Interface**: Process videos and NPZ files
- **📹 Live Webcam Streaming**: Real-time browser-based detection
- **🎥 Native OpenCV Window**: Direct camera access with live processing
- **🌐 Professional Web UI**: Modern, responsive design with real-time updates

### Advanced Features
- **Real-time WebSocket streaming** for live updates
- **Multi-camera support** with automatic switching
- **Face detection and ROI tracking**
- **Motion artifact detection** and rejection
- **Confidence scoring** for reliability assessment
- **Temporal smoothing** for stable readings
- **Professional logging** and metrics tracking

## 📖 Database

### Link.
- **https://drive.google.com/drive/folders/1q4vWuF2GJvKP5xyeX8dxaJ2fmq97-4ai


## 🚀 Quick Start

### 1. Launch the Server

The easiest way to start the system is using the main launcher:

```bash
# Start the web server (default: http://localhost:8000)
python main.py

# Start on a different port
python main.py --port 8080

# Start in debug mode with auto-reload
python main.py --debug

# Launch native OpenCV webcam only
python main.py --native-webcam

# Check system dependencies
python main.py --check-only
```

### 2. Access the Web Interface

Once the server is running, open your browser and navigate to:
- **Main Interface**: http://localhost:8000
- **Live Webcam**: http://localhost:8000/live
- **Native Webcam Control**: http://localhost:8000/webcam
- **Health Check**: http://localhost:8000/health

## 📖 Usage Guide

### File Upload Processing

#### Video Files
1. Navigate to the main page (http://localhost:8000)
2. Select "🎥 Predict from Raw Video"
3. Choose a video file (MP4, AVI, MOV, etc.)
4. Click "🎬 Process Video"
5. View results including:
   - Heart rate estimate (BPM)
   - Confidence score
   - Processing details (FPS, frame count, motion rejects)
   - Reliability assessment

#### NPZ Files (Preprocessed)
1. Select "🔬 Predict from Processed Chunk (.npz)"
2. Upload an NPZ file containing preprocessed features
3. Click "🚀 Predict Heart Rate"
4. Get instant results with waveform analysis

### Live Webcam Detection

#### Browser-Based Streaming
1. Go to http://localhost:8000/live
2. Click "🚀 Start Camera"
3. Grant camera permissions when prompted
4. Position your face in the video feed
5. Wait 3-5 seconds for initial readings
6. Monitor real-time heart rate with:
   - Live BPM display
   - Confidence indicators
   - Method comparison (MDAR vs POS vs CHROM)
   - Buffer status

#### Native OpenCV Window
1. Visit http://localhost:8000/webcam
2. Click "🚀 Start Native Webcam"
3. An OpenCV window will open showing:
   - Real-time face detection (green rectangle)
   - Live heart rate overlay
   - Processing status
4. Press 'q' in the OpenCV window to quit

## 🛠 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Webcam (for live detection)
- GPU (optional, for faster MDAR processing)

### Dependencies
The system will automatically check for required packages:
- **PyTorch** (deep learning framework)
- **OpenCV** (computer vision)
- **FastAPI** (web framework)
- **Uvicorn** (ASGI server)
- **NumPy** (numerical computing)
- **SciPy** (scientific computing)

### Model Files
The system looks for trained MDAR models in:
- `outputs/mdar_enhanced/mdar_best.pth` (enhanced model)
- `outputs/mdar/mdar_best.pth` (original model)

If no models are found, the system will still work using POS/CHROM methods.

## 🎯 Technical Details

### Signal Processing Pipeline
1. **Face Detection**: Haar cascade classifiers locate facial regions
2. **ROI Extraction**: Focus on cheek areas for optimal signal quality
3. **Illumination Normalization**: CLAHE enhancement for lighting robustness
4. **Signal Extraction**: Multiple methods extract pulse signals
5. **Bandpass Filtering**: Remove noise outside 0.7-4.0 Hz range
6. **Heart Rate Estimation**: FFT-based frequency analysis
7. **Ensemble Fusion**: Combine estimates with confidence weighting
8. **Temporal Smoothing**: Reduce jitter for stable readings

### MDAR Neural Network
- **Architecture**: Transformer-based with attention mechanisms
- **Input**: 4-channel features (R, G, B, temporal difference)
- **Output**: Heart rate waveform + direct BPM estimate
- **Training**: Multi-task learning with waveform and HR supervision

### Quality Assessment
- **Signal-to-Noise Ratio**: Power in physiological vs. noise bands
- **Spectral Concentration**: Energy focus in expected frequency range
- **Motion Detection**: Frame-to-frame brightness variation analysis
- **Confidence Scoring**: Peak strength and method agreement
- **Reliability Flags**: Overall assessment of measurement quality

## 🔧 Configuration Options

### Command Line Arguments
```bash
python main.py [OPTIONS]

Options:
  --host HOST            Host to bind server (default: 127.0.0.1)
  --port PORT            Port number (default: 8000)  
  --debug                Enable debug mode with auto-reload
  --native-webcam        Launch OpenCV webcam only
  --check-only           Check dependencies without starting server
```

### Environment Variables
- `CUDA_VISIBLE_DEVICES`: Control GPU usage
- `PYTHONPATH`: Include project directory

## 📊 Performance & Accuracy

### Expected Performance
- **Accuracy**: 90%+ for stationary subjects with good lighting
- **Processing Speed**: 15-30 FPS on modern hardware
- **Latency**: <500ms for live detection
- **Heart Rate Range**: 40-200 BPM (physiological limits)

### Optimal Conditions
- **Lighting**: Stable, natural light preferred
- **Distance**: 30-100cm from camera
- **Movement**: Minimal head motion
- **Camera**: 30 FPS, 640x480 or higher resolution

## 🔍 Troubleshooting

### Common Issues

#### "No Camera Found"
- Check camera permissions in browser/system
- Ensure camera is not used by other applications
- Try different camera indices (0, 1, 2...)

#### "Model Failed to Load" 
- Check if model files exist in `outputs/` directory
- Verify PyTorch installation
- Run `python main.py --check-only` for diagnostics

#### "WebSocket Connection Failed"
- Check firewall settings
- Ensure server is running on correct port
- Try refreshing the browser page

#### "Low Confidence Readings"
- Improve lighting conditions
- Reduce head movement
- Check camera focus and positioning
- Clean camera lens

### Debug Mode
Run with `--debug` flag for detailed logging:
```bash
python main.py --debug
```

## 🤝 API Reference

### REST Endpoints

#### `GET /health`
Check server status
```json
{"status": "ok"}
```

#### `POST /predict_video`
Process uploaded video file
- **Input**: Video file (multipart/form-data)
- **Output**: Heart rate estimate with metrics

#### `POST /predict_npz`
Process preprocessed NPZ file
- **Input**: NPZ file with features
- **Output**: Heart rate and confidence

#### `POST /api/webcam/start`
Start native webcam processing
- **Output**: Status message

#### `POST /api/webcam/stop`
Stop native webcam processing
- **Output**: Status message

#### `GET /api/webcam/status`
Get webcam status
- **Output**: Running state and message

### WebSocket Endpoints

#### `WS /ws/live`
Real-time webcam streaming
- **Input**: Base64 encoded frames
- **Output**: Live heart rate predictions

## 🔮 Future Enhancements

### Planned Features
- **Multi-person detection**: Track multiple faces simultaneously
- **Advanced filtering**: Kalman filtering for smoother estimates  
- **Export capabilities**: Save results to CSV/JSON
- **Historical tracking**: Long-term heart rate monitoring
- **Mobile optimization**: Responsive design for smartphones
- **Cloud deployment**: Docker containerization

### Research Directions
- **Attention mechanisms**: Improve MDAR architecture
- **Domain adaptation**: Better generalization across populations
- **Multi-modal fusion**: Combine with other physiological signals
- **Federated learning**: Privacy-preserving model updates

## 📄 License & Citation

This project builds upon state-of-the-art research in remote photoplethysmography. If you use this system in academic work, please cite relevant papers on rPPG methods (MDAR, POS, CHROM).

## 🙋‍♂️ Support

For questions, issues, or contributions:
1. Check existing documentation and troubleshooting guide
2. Run system diagnostics: `python main.py --check-only`
3. Enable debug mode for detailed logs: `python main.py --debug`
4. Review server logs in `outputs/server_logs/`

---

**Built with ❤️ using FastAPI, PyTorch, and OpenCV**

*Real-time heart rate detection made accessible for everyone!*
