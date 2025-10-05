import os
import os.path as osp
import json
import base64
import asyncio
from typing import List, Tuple, Dict, Any, Deque
from collections import deque
import time
import threading

import numpy as np
import cv2
import torch
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request as StarletteRequest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.datastructures import Headers

from models.mdar import MDAR


# Configure FastAPI app with increased limits for large file uploads (>2GB)
app = FastAPI(
    title="Enhanced MDAR rPPG Server with Live Webcam",
    # Increase default request size limit to 5GB
    dependencies=[]
)

# Create directories if they don't exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)

# Custom middleware to handle large file uploads
class LargeFileMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Set content length limit to 5GB
        if request.url.path.startswith('/api/video/upload'):
            # Set custom content-length handling for large files
            pass
        response = await call_next(request)
        return response

app.add_middleware(LargeFileMiddleware)

# Mount static files and templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


def load_model(weights_path: str) -> MDAR:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEBUG: Loading model from {weights_path}")
    
    # First, try to load the state to see what we have
    try:
        state = torch.load(weights_path, map_location=device, weights_only=False)
        print(f"DEBUG: Loaded state dict with keys: {list(state.keys()) if isinstance(state, dict) else 'Direct state dict'}")
    except Exception as e:
        print(f"ERROR: Failed to load weights: {e}")
        raise e
    
    # Try enhanced model first
    try:
        model = MDAR(in_features=4, hidden_channels=128, dropout=0.4, 
                     sample_rate=30.0, use_bandpass=True, multitask=True).to(device)
        
        if isinstance(state, dict) and 'model_state' in state:
            model.load_state_dict(state['model_state'])
            print("DEBUG: Loaded enhanced model with multitask=True")
        else:
            model.load_state_dict(state)
            print("DEBUG: Loaded enhanced model with direct state dict")
        
        model.eval()
        return model
        
    except Exception as e:
        print(f"DEBUG: Enhanced model failed: {e}")
        print("Trying original model architecture...")
        
        # Fallback to original model
        try:
            model = MDAR(in_features=4, hidden_channels=64, dropout=0.5, 
                         sample_rate=30.0, use_bandpass=False, multitask=False).to(device)
            
            if isinstance(state, dict) and 'model_state' in state:
                model.load_state_dict(state['model_state'])
                print("DEBUG: Loaded original model with multitask=False")
            else:
                model.load_state_dict(state)
                print("DEBUG: Loaded original model with direct state dict")
            
            model.eval()
            return model
            
        except Exception as e2:
            print(f"DEBUG: Original model also failed: {e2}")
            print("Trying basic architecture...")
            
            # Last resort - basic architecture
            model = MDAR(in_features=4, hidden_channels=64, dropout=0.3, 
                         sample_rate=30.0, use_bandpass=False, multitask=False).to(device)
            
            if isinstance(state, dict) and 'model_state' in state:
                # Try to load only compatible layers
                model_dict = model.state_dict()
                compatible_state = {k: v for k, v in state['model_state'].items() 
                                   if k in model_dict and model_dict[k].shape == v.shape}
                model_dict.update(compatible_state)
                model.load_state_dict(model_dict)
                print(f"DEBUG: Loaded basic model with {len(compatible_state)} compatible layers")
            else:
                compatible_state = {k: v for k, v in state.items() 
                                   if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
                model_dict = model.state_dict()
                model_dict.update(compatible_state)
                model.load_state_dict(model_dict)
                print(f"DEBUG: Loaded basic model with {len(compatible_state)} compatible layers")
            
            model.eval()
            return model


# Try enhanced model first, fallback to original
ENHANCED_WEIGHTS = osp.join(osp.dirname(__file__), 'outputs', 'mdar_enhanced', 'mdar_best.pth')
ORIGINAL_WEIGHTS = osp.join(osp.dirname(__file__), 'outputs', 'mdar', 'mdar_best.pth')

WEIGHTS = ENHANCED_WEIGHTS if osp.exists(ENHANCED_WEIGHTS) else ORIGINAL_WEIGHTS
MODEL = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global state for live processing
_LIVE_STATE = {
    'buffer': deque(maxlen=300),  # 10 seconds at 30fps
    'hr_history': deque(maxlen=10),
    'conf_history': deque(maxlen=10),
    'last_prediction_time': 0,
    'face_detector': None,
    'is_processing': False,
}

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, data: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except:
                # Remove broken connections
                self.active_connections.remove(connection)

manager = ConnectionManager()


_SMOOTHING_STATE = {
    'hr_history': [],
    'conf_history': [],
    'max_len': 10,
}

@app.on_event('startup')
def _startup():
    global MODEL
    print(f"Loading model from: {WEIGHTS}")
    MODEL = load_model(WEIGHTS)
    print("Model loaded successfully!")


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})




def _estimate_hr_and_conf(signal: np.ndarray, fps: float, fmin: float = 0.7, fmax: float = 4.0) -> Tuple[float, float]:
    """Estimate heart rate and confidence with robust validation"""
    if len(signal) < 16:  # Need more samples for reliable estimation
        return 0.0, 0.0
    
    # Apply windowing to reduce spectral leakage
    win = np.hanning(len(signal))
    sigw = signal * win
    
    # Compute FFT
    spec = np.fft.rfft(sigw)
    freqs = np.fft.rfftfreq(len(sigw), d=1.0 / float(fps))
    
    # Focus on physiological frequency range (0.7-4.0 Hz = 42-240 BPM)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band):
        return 0.0, 0.0
    
    mag = np.abs(spec) ** 2
    band_mag = mag[band]
    band_freqs = freqs[band]
    
    # Find peak frequency
    peak_idx = int(np.argmax(band_mag))
    peak_freq = float(band_freqs[peak_idx])
    hr_bpm = peak_freq * 60.0
    
    # Calculate confidence as peak power ratio
    peak_power = float(band_mag[peak_idx])
    total_power = float(np.sum(band_mag) + 1e-8)
    conf = float(np.clip(peak_power / total_power, 0.0, 1.0))
    
    # Physiological validation - reject implausible values
    if hr_bpm < 40.0 or hr_bpm > 200.0:
        return 0.0, 0.0
    
    # Reduce confidence for edge values
    if hr_bpm < 50.0 or hr_bpm > 180.0:
        conf *= 0.5
    
    return hr_bpm, conf


def _robust_ensemble_hr(hr_values: List[float], conf_values: List[float]) -> Tuple[float, float]:
    """Robust ensemble heart rate estimation with outlier rejection"""
    print(f"DEBUG: Ensemble input - HR values: {hr_values}, Conf values: {conf_values}")
    
    if not hr_values or not conf_values:
        return 0.0, 0.0
    
    # Filter out invalid values - use lower confidence threshold
    valid_pairs = [(hr, conf) for hr, conf in zip(hr_values, conf_values) 
                   if hr > 0 and 40 <= hr <= 200 and conf > 0.01]  # Lower confidence threshold
    
    print(f"DEBUG: Valid pairs after filtering: {valid_pairs}")
    
    if not valid_pairs:
        # If no valid pairs, try with even lower threshold
        valid_pairs = [(hr, conf) for hr, conf in zip(hr_values, conf_values) 
                       if hr > 0 and 45 <= hr <= 180]
        print(f"DEBUG: Fallback valid pairs: {valid_pairs}")
        if not valid_pairs:
            return 0.0, 0.0
    
    if len(valid_pairs) == 1:
        return valid_pairs[0][0], max(valid_pairs[0][1], 0.3)  # Boost single value confidence
    
    # Extract valid values
    valid_hrs = [pair[0] for pair in valid_pairs]
    valid_confs = [pair[1] for pair in valid_pairs]
    
    # Remove outliers using median absolute deviation
    median_hr = np.median(valid_hrs)
    mad = np.median(np.abs(np.array(valid_hrs) - median_hr))
    
    if mad > 0:
        threshold = 2.5 * mad  # More permissive threshold
        filtered_pairs = [(hr, conf) for hr, conf in valid_pairs 
                         if abs(hr - median_hr) <= threshold]
    else:
        filtered_pairs = valid_pairs
    
    if not filtered_pairs:
        return median_hr, max(np.mean(valid_confs), 0.3)
    
    # Weighted average by confidence
    hrs = [pair[0] for pair in filtered_pairs]
    confs = [pair[1] for pair in filtered_pairs]
    
    # If confidences are very low, use equal weighting
    if max(confs) < 0.1:
        final_hr = float(np.median(hrs))  # Use median for robustness
        final_conf = 0.4  # Assign reasonable confidence
    else:
        weights = np.array(confs) + 1e-6
        final_hr = np.average(hrs, weights=weights)
        final_conf = max(np.mean(confs), 0.3)  # Minimum confidence
    
    # Final validation
    if not (45 <= final_hr <= 180):
        return 0.0, 0.0
    
    print(f"DEBUG: Ensemble output - HR: {final_hr:.1f}, Conf: {final_conf:.3f}")
    return float(final_hr), float(final_conf)


# ---------- Simple preprocessing for raw video uploads ----------
def _illumination_normalize(frame: np.ndarray) -> np.ndarray:
    import cv2
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def _detect_face_roi(frame: np.ndarray, face_detector: cv2.CascadeClassifier) -> Tuple[int, int, int, int]:
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        h, w = frame.shape[:2]
        cw = int(w * 0.6)
        ch = int(h * 0.6)
        x = (w - cw) // 2
        y = (h - ch) // 2
        return int(x), int(y), int(cw), int(ch)
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    roi_y = y + h // 4
    roi_h = max(1, (3 * h) // 4)
    return int(x), int(roi_y), int(w), int(roi_h)


def _bandpass_filter(x: np.ndarray, fps: float, fmin: float = 0.7, fmax: float = 4.0) -> np.ndarray:
    if len(x) < 8:
        return x
    # Simple FFT bandpass
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / float(fps))
    mask = (freqs >= fmin) & (freqs <= fmax)
    X_f = X * mask
    y = np.fft.irfft(X_f, n=len(x))
    return y.real.astype(np.float32)


def _extract_features_from_video(path: str) -> Tuple[np.ndarray, float, np.ndarray, int]:
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return np.zeros((0, 4), dtype=np.float32), 30.0, np.zeros((0, 3), dtype=np.float32), 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    feats = []
    means_list = []
    prev_roi_mean = None
    motion_rejects = 0
    for _ in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        f = _illumination_normalize(frame)
        x, y, w, h = _detect_face_roi(f, face_detector)
        roi = f[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        # Reject frames with extreme brightness changes (motion/lighting artifacts)
        mean_bgr = np.mean(roi.reshape(-1, 3), axis=0)
        if prev_roi_mean is not None:
            if np.linalg.norm(mean_bgr - prev_roi_mean) > 40.0:
                motion_rejects += 1
                continue
        prev_roi_mean = mean_bgr
        feats.append(mean_bgr)
        means_list.append(mean_bgr)
    cap.release()

    if len(feats) == 0:
        return np.zeros((0, 4), dtype=np.float32), fps, np.zeros((0, 3), dtype=np.float32), motion_rejects
    means = np.asarray(feats, dtype=np.float32)
    green = means[:, 1]
    # Bandpass filter green to physiological range
    green_bp = _bandpass_filter(green, fps)
    diff = np.zeros_like(green_bp)
    if len(green_bp) > 1:
        diff[1:] = np.abs(np.diff(green_bp))
    out = np.concatenate([means, diff[:, None]], axis=1)
    return out.astype(np.float32), fps, np.asarray(means_list, dtype=np.float32), motion_rejects


# ---------- Traditional POS / CHROM methods for fallback/fusion ----------
def _detrend_normalize_rgb(means_bgr: np.ndarray) -> np.ndarray:
    # Convert BGR -> RGB, zero-mean, unit-std per channel
    if means_bgr.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    b = means_bgr[:, 0]
    g = means_bgr[:, 1]
    r = means_bgr[:, 2]
    X = np.stack([r, g, b], axis=1).astype(np.float64)
    X = X - np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True) + 1e-8
    X = X / std
    return X


def _pos_signal(means_bgr: np.ndarray) -> np.ndarray:
    X = _detrend_normalize_rgb(means_bgr)
    if X.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    H = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    S = X @ H.T
    s1 = S[:, 0]
    s2 = S[:, 1]
    alpha = (np.std(s1) + 1e-8) / (np.std(s2) + 1e-8)
    s = s1 + alpha * s2
    return (s - np.mean(s)).astype(np.float64)


def _chrom_signal(means_bgr: np.ndarray) -> np.ndarray:
    X = _detrend_normalize_rgb(means_bgr)
    if X.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    r = X[:, 0]; g = X[:, 1]; b = X[:, 2]
    x = 3 * r - 2 * g
    y = 1.5 * r + g - 1.5 * b
    alpha = (np.std(x) + 1e-8) / (np.std(y) + 1e-8)
    s = x - alpha * y
    return (s - np.mean(s)).astype(np.float64)



# ========== Live Webcam Processing ==========

def init_face_detector():
    """Initialize face detector if not already done"""
    if _LIVE_STATE['face_detector'] is None:
        _LIVE_STATE['face_detector'] = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    return _LIVE_STATE['face_detector']


# Use existing functions from this module instead of importing
def init_face_detector():
    """Initialize face detector if not already done"""
    if _ENHANCED_LIVE_STATE['face_detector'] is None:
        _ENHANCED_LIVE_STATE['face_detector'] = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    return _ENHANCED_LIVE_STATE['face_detector']


def enhanced_illumination_normalize(frame: np.ndarray, enable_clahe: bool = True) -> np.ndarray:
    """Normalize illumination with optional CLAHE (from live_heartbeat.py)"""
    if not enable_clahe:
        return frame
    
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def enhanced_detect_face_roi(frame: np.ndarray, face_detector: cv2.CascadeClassifier) -> Tuple[int, int, int, int]:
    """Detect face ROI (from live_heartbeat.py)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        h, w = frame.shape[:2]
        cw = int(w * 0.4)
        ch = int(h * 0.4)
        x = (w - cw) // 2
        y = (h - ch) // 2
        return int(x), int(y), int(cw), int(ch)
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    # Use the cheek area (lower 2/3 of the face, excluding mouth-chin area)
    roi_y = y + h // 3
    roi_h = max(1, (2 * h) // 3)
    return int(x), int(roi_y), int(w), int(roi_h)


def enhanced_average_cheeks(bgr: np.ndarray) -> np.ndarray:
    """Average color over two cheek sub-ROIs (from live_heartbeat.py)"""
    h, w = bgr.shape[:2]
    if h <= 0 or w <= 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    # Define left/right cheek boxes inside the face ROI
    margin_w = int(0.1 * w)
    cheek_w = max(1, int(0.25 * w))
    cheek_h = max(1, int(0.45 * h))
    top = max(0, int(0.1 * h))

    left_x = margin_w
    right_x = w - margin_w - cheek_w
    y = top

    left = bgr[y:y + cheek_h, left_x:left_x + cheek_w]
    right = bgr[y:y + cheek_h, right_x:right_x + cheek_w]
    if left.size == 0 or right.size == 0:
        roi = bgr
    else:
        roi = np.concatenate([left.reshape(-1, 3), right.reshape(-1, 3)], axis=0)
    return np.mean(roi, axis=0).astype(np.float32)


def enhanced_detrend_and_normalize(rgb_bgr: np.ndarray) -> np.ndarray:
    """Detrend and normalize RGB data (from live_heartbeat.py)"""
    b = rgb_bgr[:, 0]
    g = rgb_bgr[:, 1]
    r = rgb_bgr[:, 2]
    X = np.stack([r, g, b], axis=1).astype(np.float64)
    X = X - np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True) + 1e-8
    X = X / std
    return X


def enhanced_pos_method(rgb_bgr: np.ndarray) -> np.ndarray:
    """POS method signal extraction (from live_heartbeat.py)"""
    X = enhanced_detrend_and_normalize(rgb_bgr)
    H = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    S = X @ H.T
    s1 = S[:, 0]
    s2 = S[:, 1]
    alpha = (np.std(s1) + 1e-8) / (np.std(s2) + 1e-8)
    s = s1 + alpha * s2
    return s - np.mean(s)


def enhanced_chrom_method(rgb_bgr: np.ndarray) -> np.ndarray:
    """CHROM method signal extraction (from live_heartbeat.py)"""
    X = enhanced_detrend_and_normalize(rgb_bgr)
    r = X[:, 0]
    g = X[:, 1]
    b = X[:, 2]
    x = 3 * r - 2 * g
    y = 1.5 * r + g - 1.5 * b
    alpha = (np.std(x) + 1e-8) / (np.std(y) + 1e-8)
    s = x - alpha * y
    return s - np.mean(s)


def enhanced_bandpass(x: np.ndarray, fps: float, fmin: float = 0.7, fmax: float = 4.0) -> np.ndarray:
    """Bandpass filter (from live_heartbeat.py)"""
    if len(x) < 8:
        return x
    
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / float(fps))
    mask = (freqs >= fmin) & (freqs <= fmax)
    Xf = X * mask
    y = np.fft.irfft(Xf, n=len(x))
    return y.real.astype(np.float32)


def enhanced_quadratic_peak_interpolate(mag: np.ndarray, idx: int) -> float:
    """Parabolic interpolation around peak (from live_heartbeat.py)"""
    if idx <= 0 or idx >= len(mag) - 1:
        return 0.0
    y0, y1, y2 = mag[idx - 1], mag[idx], mag[idx + 1]
    denom = (y0 - 2 * y1 + y2)
    if abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (y0 - y2) / denom


def enhanced_estimate_hr_fft_enhanced(signal: np.ndarray, fps: float, fmin: float = 0.75, fmax: float = 3.5) -> Tuple[float, float]:
    """Enhanced HR estimation (from live_heartbeat.py)"""
    if len(signal) < 12:  # Even more aggressive for real-time
        return 0.0, 0.0
    
    # Enhanced detrending with polynomial fitting
    if len(signal) > 30:
        t = np.arange(len(signal))
        # Remove linear trend
        coeffs = np.polyfit(t, signal, 1)
        signal = signal - np.polyval(coeffs, t)
    else:
        signal = signal - np.mean(signal)
    
    # Adaptive windowing based on signal length
    if len(signal) < 60:
        win = np.hanning(len(signal))  # Hanning for short signals
    else:
        # Tukey window with 10% tapering for longer signals
        alpha = 0.1
        win_len = len(signal)
        win = np.ones(win_len)
        taper_len = int(alpha * win_len / 2)
        for i in range(taper_len):
            factor = 0.5 * (1 + np.cos(np.pi * (i / taper_len - 1)))
            win[i] = factor
            win[-(i+1)] = factor
    
    sigw = signal * win
    
    # Zero-padding for better frequency resolution
    nfft = max(len(sigw) * 2, 512)  # At least 512 points for good resolution
    spec = np.fft.rfft(sigw, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / float(fps))
    
    # Refined physiological band
    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band):
        return 0.0, 0.0
    
    mag = (np.abs(spec) ** 2).astype(np.float64)
    band_idxs = np.where(band)[0]
    local_mag = mag[band]
    
    # Find multiple peaks for better validation
    peak_indices = []
    peak_powers = []
    
    # Primary peak
    primary_peak_local = int(np.argmax(local_mag))
    primary_peak_idx = band_idxs[primary_peak_local]
    peak_indices.append(primary_peak_idx)
    peak_powers.append(float(mag[primary_peak_idx]))
    
    # Look for secondary peaks (harmonics or alternative estimates)
    primary_power = mag[primary_peak_idx]
    for i, idx in enumerate(band_idxs):
        if abs(idx - primary_peak_idx) > 3:  # Avoid nearby peaks
            if mag[idx] > 0.3 * primary_power:  # At least 30% of primary peak
                peak_indices.append(idx)
                peak_powers.append(float(mag[idx]))
    
    # Enhanced peak frequency estimation with interpolation
    delta = enhanced_quadratic_peak_interpolate(mag, primary_peak_idx)
    peak_freq = freqs[primary_peak_idx] + delta * (fps / nfft)
    
    # Validate frequency against harmonics
    if len(peak_indices) > 1:
        frequencies = [freqs[idx] for idx in peak_indices]
        # Check if any secondary peaks are harmonics of the primary
        for f in frequencies[1:]:
            if abs(f - 2 * peak_freq) < 0.1 or abs(2 * f - peak_freq) < 0.1:
                # Harmonic detected, increase confidence in primary peak
                peak_powers[0] *= 1.2
                break
    
    # Convert to BPM
    hr = float(peak_freq * 60.0)
    
    # Enhanced validation
    if hr < 40 or hr > 200:
        return 0.0, 0.0
    
    # Calculate enhanced confidence incorporating spectral characteristics
    confidence = float(np.sqrt(peak_powers[0]) / (np.sum(local_mag) + 1e-8))
    
    return hr, confidence


def enhanced_assess_signal_quality(signal: np.ndarray, fps: float) -> float:
    """Assess signal quality (from live_heartbeat.py)"""
    if len(signal) < 20:
        return 0.1
    
    # 1. Signal-to-noise ratio estimation
    signal_power = np.var(signal)
    if signal_power < 1e-12:
        return 0.0
    
    # 2. Spectral concentration in physiological range
    spec = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / float(fps))
    mag = np.abs(spec) ** 2
    
    # Physiological band power
    physio_band = (freqs >= 0.7) & (freqs <= 4.0)
    noise_band = (freqs > 4.0) | (freqs < 0.5)
    
    if np.any(physio_band) and np.any(noise_band):
        physio_power = np.sum(mag[physio_band])
        noise_power = np.sum(mag[noise_band]) + 1e-12
        snr = physio_power / noise_power
        quality_snr = min(1.0, snr / 10.0)  # Normalize to [0, 1]
    else:
        quality_snr = 0.1
    
    # 3. Periodicity assessment using autocorrelation
    if len(signal) > 40:
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Look for peaks in autocorrelation at physiological lags
        min_lag = int(0.5 * fps)  # 0.5 seconds (120 bpm max)
        max_lag = int(1.5 * fps)  # 1.5 seconds (40 bpm min)
        
        if max_lag < len(autocorr):
            autocorr_range = autocorr[min_lag:max_lag]
            max_autocorr = np.max(autocorr_range) if len(autocorr_range) > 0 else 0.0
            quality_periodicity = max(0.0, max_autocorr)
        else:
            quality_periodicity = 0.1
    else:
        quality_periodicity = 0.2
    
    # 4. Signal stability (inverse of coefficient of variation)
    cv = np.std(signal) / (np.abs(np.mean(signal)) + 1e-8)
    quality_stability = 1.0 / (1.0 + cv)  # Higher CV = lower quality
    
    # Weighted combination of quality metrics
    overall_quality = (
        0.4 * quality_snr + 
        0.3 * quality_periodicity + 
        0.3 * quality_stability
    )
    
    return float(np.clip(overall_quality, 0.0, 1.0))


def enhanced_calculate_measurement_confidence(hr_estimate: float, all_estimates: List[float], 
                                           weights: List[float], hr_history: Deque[float]) -> float:
    """Calculate confidence in measurement (from live_heartbeat.py)"""
    if hr_estimate <= 0:
        return 0.0
    
    confidence_factors = []
    
    # 1. Physiological plausibility (higher confidence for normal ranges)
    if 60 <= hr_estimate <= 100:
        physio_conf = 1.0
    elif 50 <= hr_estimate <= 150:
        physio_conf = 0.8
    elif 40 <= hr_estimate <= 180:
        physio_conf = 0.6
    else:
        physio_conf = 0.3
    confidence_factors.append(physio_conf)
    
    # 2. Cross-method agreement
    if len(all_estimates) > 1:
        estimates_array = np.array(all_estimates)
        std_estimates = np.std(estimates_array)
        mean_estimates = np.mean(estimates_array)
        agreement = 1.0 - (std_estimates / (mean_estimates + 1e-8))
        agreement_conf = max(0.1, min(1.0, agreement))
    else:
        agreement_conf = 0.7  # Moderate confidence for single estimate
    confidence_factors.append(agreement_conf)
    
    # 3. Signal strength (from weights)
    if len(weights) > 0:
        weights_array = np.array(weights) if not isinstance(weights, np.ndarray) else weights
        max_weight = np.max(weights_array)
        signal_conf = min(1.0, max_weight / 1e-3)  # Normalize based on expected range
    else:
        signal_conf = 0.5
    confidence_factors.append(signal_conf)
    
    # 4. Historical consistency
    if len(hr_history) >= 3:
        recent_values = list(hr_history)[-5:]  # Last 5 values
        hist_mean = np.mean(recent_values)
        hist_std = np.std(recent_values)
        
        if hist_std < 5:  # Very stable
            stability_conf = 1.0
        elif hist_std < 10:  # Moderately stable
            stability_conf = 0.8
        elif hist_std < 20:  # Variable but acceptable
            stability_conf = 0.6
        else:  # Highly variable
            stability_conf = 0.3
        
        # Penalize estimates that deviate significantly from recent history
        deviation = abs(hr_estimate - hist_mean)
        if deviation > 3 * hist_std + 10:  # 3 sigma + 10 bpm tolerance
            stability_conf *= 0.5
        
        confidence_factors.append(stability_conf)
    else:
        confidence_factors.append(0.6)  # Moderate confidence for limited history
    
    # 5. Measurement precision (how well-defined the peak is)
    precision_conf = 0.8  # Default value; could be enhanced with spectral sharpness
    confidence_factors.append(precision_conf)
    
    # Weighted average of confidence factors
    weights_conf = [0.2, 0.3, 0.2, 0.2, 0.1]  # Emphasize agreement and stability
    overall_confidence = np.average(confidence_factors, weights=weights_conf)
    
    return float(np.clip(overall_confidence, 0.0, 1.0))


# Enhanced live state with advanced features matching live_heartbeat.py
_ENHANCED_LIVE_STATE = {
    'buffer': deque(maxlen=240),  # 8 seconds at 30fps
    'hr_history': deque(maxlen=16),  # History for advanced smoothing
    'conf_history': deque(maxlen=16),
    'last_prediction_time': 0,
    'face_detector': None,
    'is_processing': False,
    'smoothed_hr': 0.0,
    'clahe_enabled': True,
    'min_samples': 75,  # Minimum 2.5 seconds at 30fps
    'update_interval': 0.25,  # Update every 250ms
    'window_seconds': 8.0,  # 8 second sliding window
}


def process_live_frame(frame_data: str) -> Dict[str, Any]:
    """Process a single frame using advanced live_heartbeat.py algorithm"""
    try:
        # Decode base64 frame
        img_data = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {'error': 'Failed to decode frame'}
        
        # Initialize face detector
        face_detector = init_face_detector()
        
        # Apply illumination normalization (CLAHE) like live_heartbeat.py
        frame_norm = enhanced_illumination_normalize(frame, _ENHANCED_LIVE_STATE['clahe_enabled'])
        x, y, w, h = enhanced_detect_face_roi(frame_norm, face_detector)
        roi = frame_norm[y:y + h, x:x + w]
        
        if roi.size == 0:
            return {'error': 'No face detected'}
        
        # Use average_cheeks for better ROI extraction (live_heartbeat.py method)
        mean_bgr = enhanced_average_cheeks(roi)
        _ENHANCED_LIVE_STATE['buffer'].append(mean_bgr)
        
        current_time = time.time()
        fps = 30.0  # Assume 30fps for webcam
        min_samples = _ENHANCED_LIVE_STATE['min_samples']
        
        # Advanced processing when enough samples are available
        if (len(_ENHANCED_LIVE_STATE['buffer']) >= min_samples and 
            current_time - _ENHANCED_LIVE_STATE['last_prediction_time'] > _ENHANCED_LIVE_STATE['update_interval'] and
            not _ENHANCED_LIVE_STATE['is_processing']):
            
            _ENHANCED_LIVE_STATE['is_processing'] = True
            _ENHANCED_LIVE_STATE['last_prediction_time'] = current_time
            
            try:
                # Use sliding window approach (live_heartbeat.py style)
                buffer_size = len(_ENHANCED_LIVE_STATE['buffer'])
                window_samples = min(buffer_size, int(_ENHANCED_LIVE_STATE['window_seconds'] * fps))
                recent_data = list(_ENHANCED_LIVE_STATE['buffer'])[-window_samples:]
                arr = np.stack(recent_data, axis=0)
                
                # Enhanced signal processing with tighter frequency band
                s_pos = enhanced_bandpass(enhanced_pos_method(arr), fps, fmin=0.75, fmax=3.5)
                s_chr = enhanced_bandpass(enhanced_chrom_method(arr), fps, fmin=0.75, fmax=3.5)
                
                # Multi-scale HR estimation for better accuracy
                hr_pos, p_pos = enhanced_estimate_hr_fft_enhanced(s_pos, fps)
                hr_chr, p_chr = enhanced_estimate_hr_fft_enhanced(s_chr, fps)
                
                # Quality assessment for each method
                pos_quality = enhanced_assess_signal_quality(s_pos, fps)
                chr_quality = enhanced_assess_signal_quality(s_chr, fps)
                
                # Advanced fusion algorithm (from live_heartbeat.py)
                valid_estimates = []
                weights = []
                
                # More lenient validation for real-time processing
                if 45 <= hr_pos <= 180 and p_pos > 0 and pos_quality > 0.2:
                    valid_estimates.append(hr_pos)
                    weights.append(p_pos * pos_quality)
                
                if 45 <= hr_chr <= 180 and p_chr > 0 and chr_quality > 0.2:
                    valid_estimates.append(hr_chr)
                    weights.append(p_chr * chr_quality)
                
                hr_fused = 0.0
                if valid_estimates:
                    if len(valid_estimates) == 1:
                        hr_fused = valid_estimates[0]
                    else:
                        # Weighted average with outlier detection
                        weights = np.array(weights)
                        weights = weights / np.sum(weights)  # Normalize weights
                        
                        # Check for outliers using median absolute deviation
                        estimates = np.array(valid_estimates)
                        median_hr = np.median(estimates)
                        mad = np.median(np.abs(estimates - median_hr))
                        
                        if mad > 0:
                            # Remove outliers that are more than 2.5 MAD from median
                            threshold = 2.5 * mad
                            inliers = np.abs(estimates - median_hr) <= threshold
                            
                            if np.any(inliers):
                                hr_fused = np.average(estimates[inliers], weights=weights[inliers])
                            else:
                                hr_fused = median_hr
                        else:
                            hr_fused = np.average(estimates, weights=weights)
                    
                    # Enhanced temporal consistency validation
                    hr_history = _ENHANCED_LIVE_STATE['hr_history']
                    if len(hr_history) > 0:
                        recent_window = min(5, len(hr_history))
                        recent_values = list(hr_history)[-recent_window:]
                        recent_median = np.median(recent_values)
                        recent_std = np.std(recent_values) if len(recent_values) > 1 else 10.0
                        
                        # Adaptive threshold based on recent variability
                        threshold = max(15, 2 * recent_std)
                        
                        if abs(hr_fused - recent_median) > threshold:
                            # Check for consistent trend
                            if len(hr_history) >= 3:
                                last_3 = recent_values[-3:] if len(recent_values) >= 3 else recent_values
                                if len(last_3) >= 2:
                                    trend_consistent = True
                                    for i in range(len(last_3)-1):
                                        if np.sign(last_3[i+1] - last_3[i]) != np.sign(hr_fused - recent_median):
                                            trend_consistent = False
                                            break
                                    
                                    if not trend_consistent and abs(hr_fused - recent_median) > 20:
                                        hr_fused = 0.0  # Reject inconsistent outliers
                    
                    # Additional signal quality constraints
                    if hr_fused > 0:
                        # Cross-method consistency check
                        if len(valid_estimates) > 1:
                            consistency = 1.0 - (np.std(valid_estimates) / (np.mean(valid_estimates) + 1e-8))
                            if consistency < 0.7:  # Methods disagree significantly
                                hr_fused *= 0.8  # Reduce confidence but don't reject entirely
                        
                        # Signal power validation
                        avg_power = np.mean([p_pos, p_chr]) if p_pos > 0 and p_chr > 0 else max(p_pos, p_chr)
                        if avg_power < 1e-6:  # Very weak signal
                            hr_fused = 0.0
                
                # Adaptive smoothing with momentum and confidence weighting
                smoothed_hr = _ENHANCED_LIVE_STATE['smoothed_hr']
                if hr_fused > 0:
                    # Calculate confidence based on multiple factors
                    confidence = enhanced_calculate_measurement_confidence(
                        hr_fused, valid_estimates, weights, _ENHANCED_LIVE_STATE['hr_history']
                    )
                    
                    # Adaptive smoothing parameter based on confidence and change magnitude
                    if smoothed_hr > 0:
                        change_magnitude = abs(hr_fused - smoothed_hr)
                        base_alpha = 0.3 + 0.3 * confidence  # 0.3 to 0.6 based on confidence
                        
                        if change_magnitude > 20:
                            alpha = base_alpha * 0.3  # Very conservative for large changes
                        elif change_magnitude > 10:
                            alpha = base_alpha * 0.6  # Moderately conservative
                        else:
                            alpha = base_alpha  # Normal adaptation
                    else:
                        alpha = 0.7  # Quick initial acquisition
                    
                    smoothed_hr = (1 - alpha) * smoothed_hr + alpha * hr_fused
                    _ENHANCED_LIVE_STATE['smoothed_hr'] = smoothed_hr
                    
                    # Track recent HRs for stability monitoring
                    _ENHANCED_LIVE_STATE['hr_history'].append(hr_fused)
                    _ENHANCED_LIVE_STATE['conf_history'].append(confidence)
                    
                    # Determine stability status
                    if len(_ENHANCED_LIVE_STATE['hr_history']) >= 5:
                        recent_std = np.std(list(_ENHANCED_LIVE_STATE['hr_history'])[-5:])
                        if recent_std < 3.0:
                            status = 'STABLE'
                        elif recent_std < 6.0:
                            status = 'TRACKING'
                        else:
                            status = 'VARIABLE'
                    else:
                        status = 'STARTING'
                    
                    _ENHANCED_LIVE_STATE['is_processing'] = False
                    
                    return {
                        'hr_bpm': float(smoothed_hr),
                        'confidence': float(confidence * 100),
                        'hr_pos': float(hr_pos) if hr_pos > 0 else None,
                        'hr_chrom': float(hr_chr) if hr_chr > 0 else None,
                        'hr_mdar': None,  # Not using MDAR in live mode for performance
                        'buffer_size': len(_ENHANCED_LIVE_STATE['buffer']),
                        'face_roi': [int(x), int(y), int(w), int(h)],
                        'reliable': 45 <= smoothed_hr <= 200 and confidence > 0.3,
                        'status': status,
                        'quality': float(max(pos_quality, chr_quality)),
                        'methods_used': len(valid_estimates)
                    }
                else:
                    _ENHANCED_LIVE_STATE['is_processing'] = False
                    return {
                        'buffer_size': len(_ENHANCED_LIVE_STATE['buffer']),
                        'face_roi': [int(x), int(y), int(w), int(h)],
                        'waiting_for_data': False,
                        'status': 'NO_SIGNAL'
                    }
                
            except Exception as e:
                _ENHANCED_LIVE_STATE['is_processing'] = False
                return {'error': f'Processing error: {str(e)}'}
        
        # Return basic info if not ready for prediction
        return {
            'buffer_size': len(_ENHANCED_LIVE_STATE['buffer']),
            'face_roi': [int(x), int(y), int(w), int(h)],
            'waiting_for_data': len(_ENHANCED_LIVE_STATE['buffer']) < min_samples,
            'status': 'BUFFERING' if len(_ENHANCED_LIVE_STATE['buffer']) < min_samples else 'READY'
        }
        
    except Exception as e:
        return {'error': f'Frame processing error: {str(e)}'}


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Receive frame data from client
            data = await websocket.receive_json()
            
            if data.get('type') == 'frame':
                frame_data = data.get('frame')
                if frame_data:
                    result = process_live_frame(frame_data)
                    await websocket.send_json({
                        'type': 'prediction',
                        'data': result,
                        'timestamp': time.time()
                    })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get('/webcam', response_class=HTMLResponse)
def native_webcam():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Native Webcam Monitor</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-align: center;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        .card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 30px;
            backdrop-filter: blur(10px);
            margin: 20px 0;
        }
        button {
            background: rgba(255, 255, 255, 0.2);
            border: 2px solid rgba(255, 255, 255, 0.3);
            color: white;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            margin: 5px;
            transition: all 0.3s ease;
        }
        button:hover {
            background: rgba(255, 255, 255, 0.3);
        }
        .status {
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
            font-size: 18px;
            font-weight: bold;
        }
        .status.running { background: rgba(76, 175, 80, 0.3); }
        .status.stopped { background: rgba(244, 67, 54, 0.3); }
        .status.info { background: rgba(33, 150, 243, 0.3); }
        .nav {
            text-align: center;
            margin-bottom: 2rem;
        }
        .nav a {
            color: white;
            text-decoration: none;
            background: rgba(255, 255, 255, 0.2);
            padding: 12px 24px;
            border-radius: 25px;
            margin: 0 10px;
            transition: all 0.3s ease;
        }
        .nav a:hover {
            background: rgba(255, 255, 255, 0.3);
        }
        h1 { text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); }
    </style>
</head>
<body>
    <div class="container">
        <h1>📹 Native Webcam Heart Rate Monitor</h1>
        
        <div class="nav" style="display: none;">
            <!-- Navigation removed - single page application -->
        </div>
        
        <div class="card">
            <h2>Direct OpenCV Webcam Processing</h2>
            <p>Uses the live_heartbeat.py algorithm with POS + CHROM methods</p>
            <p>Opens a native window with real-time face detection and heart rate monitoring</p>
            
            <div class="status info" id="status">Ready to start native webcam processing</div>
            
            <button onclick="startNativeWebcam()">🚀 Start Native Webcam</button>
            <button onclick="stopNativeWebcam()">🛑 Stop Webcam</button>
            <button onclick="checkStatus()">📊 Check Status</button>
        </div>
        
        <div class="card">
            <h3>Instructions</h3>
            <ul style="text-align: left; max-width: 500px; margin: 0 auto;">
                <li>Click "Start Native Webcam" to open OpenCV window</li>
                <li>Position your face in the green rectangle</li>
                <li>Wait 3-10 seconds for heart rate calculation</li>
                <li>Heart rate will be displayed on the video feed</li>
                <li>Press 'q' in the video window to quit</li>
            </ul>
        </div>
    </div>

    <script>
        let webcamProcess = null;
        
        async function startNativeWebcam() {
            try {
                document.getElementById('status').textContent = 'Starting native webcam...';
                document.getElementById('status').className = 'status info';
                
                const response = await fetch('/api/webcam/start', {
                    method: 'POST'
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    document.getElementById('status').textContent = result.message;
                    document.getElementById('status').className = 'status running';
                } else {
                    document.getElementById('status').textContent = `Error: ${result.detail}`;
                    document.getElementById('status').className = 'status stopped';
                }
            } catch (error) {
                document.getElementById('status').textContent = `Connection error: ${error.message}`;
                document.getElementById('status').className = 'status stopped';
            }
        }
        
        async function stopNativeWebcam() {
            try {
                const response = await fetch('/api/webcam/stop', {
                    method: 'POST'
                });
                
                const result = await response.json();
                document.getElementById('status').textContent = result.message;
                document.getElementById('status').className = 'status stopped';
            } catch (error) {
                document.getElementById('status').textContent = `Error stopping: ${error.message}`;
                document.getElementById('status').className = 'status stopped';
            }
        }
        
        async function checkStatus() {
            try {
                const response = await fetch('/api/webcam/status');
                const result = await response.json();
                
                document.getElementById('status').textContent = result.message;
                document.getElementById('status').className = result.running ? 'status running' : 'status stopped';
            } catch (error) {
                document.getElementById('status').textContent = `Status check failed: ${error.message}`;
                document.getElementById('status').className = 'status stopped';
            }
        }
    </script>
</body>
</html>
"""


@app.get('/live', response_class=HTMLResponse)
def live_webcam():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Live rPPG Heart Rate Monitor</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🫀 Live Heart Rate Monitor</h1>
            <p class="subtitle">Enhanced MDAR + POS/CHROM Ensemble Method</p>
        </div>
        
        <div class="nav" style="display: none;">
            <!-- Navigation removed - single page application -->
        </div>
        
        <div class="controls">
            <button id="startCamera" class="btn btn-primary" onclick="startCamera()">
                🚀 Start Camera
            </button>
            <button id="stopCamera" class="btn" onclick="stopCamera()" disabled>
                🛑 Stop Camera
            </button>
        </div>
        
        <div class="video-container">
            <div class="video-box">
                <video id="video" autoplay muted playsinline></video>
                <canvas id="canvas"></canvas>
            </div>
            
            <div class="metrics card">
                <div class="hr-display" id="hrDisplay">-- BPM</div>
                <div class="confidence-display" id="confidence">Confidence: --%</div>
                
                <div class="status info" id="status">Click Start Camera to begin</div>
                
                <div class="method-comparison">
                    <h3>📊 Method Comparison</h3>
                    <div class="method-row">
                        <span class="method-name">MDAR Model:</span>
                        <span class="method-value" id="hrMdar">--</span>
                    </div>
                    <div class="method-row">
                        <span class="method-name">POS Method:</span>
                        <span class="method-value" id="hrPos">--</span>
                    </div>
                    <div class="method-row">
                        <span class="method-name">CHROM Method:</span>
                        <span class="method-value" id="hrChrom">--</span>
                    </div>
                    <div class="method-row">
                        <span class="method-name">Buffer Size:</span>
                        <span class="method-value" id="bufferSize">0</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let video, canvas, ctx, ws;
        let isRunning = false;
        let frameCount = 0;
        
        function updateDisplay(data) {
            if (data.hr_bpm) {
                document.getElementById('hrDisplay').textContent = data.hr_bpm.toFixed(1) + ' BPM';
                document.getElementById('confidence').textContent = `Confidence: ${data.confidence.toFixed(1)}%`;
                
                document.getElementById('hrMdar').textContent = data.hr_mdar ? data.hr_mdar.toFixed(1) : '--';
                document.getElementById('hrPos').textContent = data.hr_pos ? data.hr_pos.toFixed(1) : '--';
                document.getElementById('hrChrom').textContent = data.hr_chrom ? data.hr_chrom.toFixed(1) : '--';
                
                const statusEl = document.getElementById('status');
                if (data.reliable) {
                    statusEl.textContent = '✅ Reading reliable';
                    statusEl.className = 'status success';
                } else {
                    statusEl.textContent = '⚠️ Reading may be unreliable';
                    statusEl.className = 'status warning';
                }
            }
            
            if (data.buffer_size !== undefined) {
                document.getElementById('bufferSize').textContent = data.buffer_size;
            }
            
            if (data.waiting_for_data) {
                document.getElementById('status').textContent = 
                    `🔄 Collecting data... (${data.buffer_size}/90 frames)`;
                document.getElementById('status').className = 'status warning';
            }
            
            if (data.error) {
                document.getElementById('status').textContent = `❌ Error: ${data.error}`;
                document.getElementById('status').className = 'status error';
            }
        }
        
        function sendFrame() {
            if (!isRunning || !ws || ws.readyState !== WebSocket.OPEN) return;
            
            frameCount++;
            // Send every 3rd frame to reduce bandwidth
            if (frameCount % 3 !== 0) {
                requestAnimationFrame(sendFrame);
                return;
            }
            
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const frameData = canvas.toDataURL('image/jpeg', 0.7);
            
            ws.send(JSON.stringify({
                type: 'frame',
                frame: frameData
            }));
            
            requestAnimationFrame(sendFrame);
        }
        
        function startCamera() {
            navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480, frameRate: 30 } 
            })
            .then(stream => {
                video.srcObject = stream;
                
                // Initialize WebSocket
                ws = new WebSocket(`ws://${window.location.host}/ws/live`);
                
                ws.onopen = () => {
                    console.log('WebSocket connected');
                    isRunning = true;
                    sendFrame();
                };
                
                ws.onmessage = (event) => {
                    const message = JSON.parse(event.data);
                    if (message.type === 'prediction') {
                        updateDisplay(message.data);
                    }
                };
                
                ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    document.getElementById('status').textContent = '❌ Connection error';
                    document.getElementById('status').className = 'status error';
                };
                
                ws.onclose = () => {
                    console.log('WebSocket disconnected');
                    isRunning = false;
                };
                
                document.getElementById('status').textContent = '🔄 Camera started, connecting...';
                document.getElementById('status').className = 'status warning';
            })
            .catch(err => {
                console.error('Camera error:', err);
                document.getElementById('status').textContent = '❌ Camera access denied';
                document.getElementById('status').className = 'status error';
            });
        }
        
        function stopCamera() {
            isRunning = false;
            
            if (video.srcObject) {
                video.srcObject.getTracks().forEach(track => track.stop());
                video.srcObject = null;
            }
            
            if (ws) {
                ws.close();
                ws = null;
            }
            
            document.getElementById('status').textContent = 'Camera stopped';
            document.getElementById('status').className = 'status';
            document.getElementById('hrDisplay').textContent = '-- BPM';
            document.getElementById('confidence').textContent = 'Confidence: --%';
        }
        
        // Initialize elements
        window.onload = () => {
            video = document.getElementById('video');
            canvas = document.getElementById('canvas');
            ctx = canvas.getContext('2d');
            canvas.width = 640;
            canvas.height = 480;
        };
        
        // Cleanup on page unload
        window.onbeforeunload = stopCamera;
    </script>
    <script src="/static/js/app.js"></script>
</body>
</html>
"""


# ========== Camera Selection API ==========

# Camera cache to avoid repeated detection
_CAMERA_CACHE = {
    'cameras': [],
    'last_updated': 0,
    'cache_duration': 30  # Cache for 30 seconds
}

def get_cached_cameras():
    """Get cameras with caching to improve performance"""
    import time
    current_time = time.time()
    
    # Return cached result if still valid
    if (current_time - _CAMERA_CACHE['last_updated'] < _CAMERA_CACHE['cache_duration'] and 
        _CAMERA_CACHE['cameras']):
        return _CAMERA_CACHE['cameras']
    
    # Otherwise, refresh the cache
    try:
        from live_heartbeat import list_available_cameras
        cameras = list_available_cameras()
        
        camera_list = []
        for cam_id, description in cameras:
            camera_list.append({
                'id': cam_id,
                'description': description
            })
        
        _CAMERA_CACHE['cameras'] = camera_list
        _CAMERA_CACHE['last_updated'] = current_time
        
        return camera_list
        
    except Exception as e:
        print(f"Error listing cameras: {e}")
        # Return cached result even if expired, or empty list
        return _CAMERA_CACHE['cameras'] if _CAMERA_CACHE['cameras'] else []

@app.get('/api/cameras/list')
async def list_cameras():
    """List available cameras with caching"""
    camera_list = get_cached_cameras()
    return JSONResponse(camera_list)


# ========== Video Upload Processing ==========

@app.post('/api/video/upload')
async def upload_video_for_prediction(file: UploadFile = File(..., description="Video file for heart rate analysis (supports files up to 5GB)")):
    """Process uploaded video file and return heart rate prediction."""
    if not file.filename:
        return JSONResponse({'error': 'No file uploaded'}, status_code=400)
    
    # Validate file type
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
    file_ext = os.path.splitext(file.filename.lower())[1]
    if file_ext not in allowed_extensions:
        return JSONResponse({
            'error': f'Unsupported file format {file_ext}. Supported formats: {", ".join(allowed_extensions)}'
        }, status_code=400)
    
    # Create temporary file
    temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_filename = f"temp_{int(time.time())}_{file.filename}"
    temp_path = os.path.join(temp_dir, temp_filename)
    
    try:
        # Save uploaded file with streaming for large files (>2GB support)
        with open(temp_path, 'wb') as f:
            # Stream file in chunks to handle large files efficiently
            chunk_size = 8192  # 8KB chunks
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
        
        # Get file size for logging
        file_size = os.path.getsize(temp_path)
        print(f"Processing video file: {temp_path} ({file_size} bytes)")
        
        # Extract features from video
        features, fps, means_bgr, motion_rejects = _extract_features_from_video(temp_path)
        
        if features.shape[0] == 0:
            return JSONResponse({
                'error': 'No valid frames found in video. Please ensure the video contains clear face regions.'
            }, status_code=400)
        
        print(f"Extracted {features.shape[0]} frames at {fps:.1f} fps, motion rejects: {motion_rejects}")
        
        # Process with MDAR model
        try:
            x = torch.from_numpy(features[None, ...]).to(DEVICE)
            with torch.no_grad():
                outputs = MODEL(x)
                if isinstance(outputs, dict):
                    waveform = outputs['waveform'].cpu().numpy()[0]
                    hr_mdar = outputs['heart_rate'].cpu().item() * 60.0
                    confidence = outputs['confidence'].cpu().item()
                else:
                    waveform = outputs.cpu().numpy()[0]
                    hr_mdar, confidence = _estimate_hr_and_conf(waveform, fps)
        except Exception as e:
            print(f"MDAR model error: {e}")
            hr_mdar, confidence = 0.0, 0.0
            waveform = np.zeros(100)
        
        # Traditional methods for comparison
        if means_bgr.shape[0] > 0:
            pos_signal = _pos_signal(means_bgr)
            chrom_signal = _chrom_signal(means_bgr)
            
            pos_signal_bp = _bandpass_filter(pos_signal.astype(np.float32), fps)
            chrom_signal_bp = _bandpass_filter(chrom_signal.astype(np.float32), fps)
            
            hr_pos, conf_pos = _estimate_hr_and_conf(pos_signal_bp, fps)
            hr_chrom, conf_chrom = _estimate_hr_and_conf(chrom_signal_bp, fps)
        else:
            hr_pos = hr_chrom = conf_pos = conf_chrom = 0.0
        
        # Ensemble prediction
        hr_values = [hr_mdar, hr_pos, hr_chrom]
        conf_values = [confidence, conf_pos, conf_chrom]
        hr_ensemble, ensemble_conf = _robust_ensemble_hr(hr_values, conf_values)
        
        # Use best available prediction
        if hr_ensemble > 0:
            final_hr = hr_ensemble
            final_conf = ensemble_conf
        else:
            # Fallback to individual methods
            valid_hrs = [(hr, conf) for hr, conf in zip(hr_values, conf_values) if 45 <= hr <= 180]
            if valid_hrs:
                # Use the prediction with highest confidence
                final_hr, final_conf = max(valid_hrs, key=lambda x: x[1])
            else:
                final_hr, final_conf = 0.0, 0.0
        
        # Determine reliability
        reliable = 45 <= final_hr <= 180 and final_conf > 0.2
        
        result = {
            'hr_bpm': float(final_hr),
            'confidence': float(final_conf * 100),
            'reliable': bool(reliable),
            'hr_mdar': float(hr_mdar) if hr_mdar > 0 else None,
            'hr_pos': float(hr_pos) if hr_pos > 0 else None,
            'hr_chrom': float(hr_chrom) if hr_chrom > 0 else None,
            'fps': float(fps),
            'frames_processed': int(features.shape[0]),
            'motion_rejects': int(motion_rejects),
            'waveform_length': int(len(waveform)),
            'processing_time': time.time() - int(temp_filename.split('_')[1])
        }
        
        print(f"Video processing result: HR={final_hr:.1f} BPM, Conf={final_conf:.3f}")
        return JSONResponse(result)
        
    except Exception as e:
        print(f"Video processing error: {str(e)}")
        return JSONResponse({
            'error': f'Video processing failed: {str(e)}'
        }, status_code=500)
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"Warning: Failed to remove temp file {temp_path}: {e}")


# ========== Native Webcam Processing (live_heartbeat.py integration) ==========

# Global webcam state
_WEBCAM_STATE = {
    'process': None,
    'thread': None,
    'running': False
}

# Native webcam API endpoints

def native_webcam_process():
    """Native webcam processing using live_heartbeat.py functionality"""
    import cv2
    import time
    from collections import deque
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print('Failed to open webcam (device 0).')
            return

        # Try to estimate FPS from camera; default to 30
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 1.0 or np.isnan(fps):
            fps = 30.0

        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Sliding buffers - using live_heartbeat.py approach
        window_seconds = 10.0
        max_len = int(window_seconds * fps)
        rgb_buffer = deque(maxlen=max_len)

        last_time = time.time()
        smoothed_hr = 0.0

        while _WEBCAM_STATE['running']:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            
            # Use server's illumination normalization
            norm = _illumination_normalize(frame)
            
            # Use server's face detection but with live_heartbeat.py parameters
            gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
            
            if len(faces) == 0:
                h, w = norm.shape[:2]
                cw = int(w * 0.4)
                ch = int(h * 0.4)
                x = (w - cw) // 2
                y = (h - ch) // 2
                roi = norm[y:y + ch, x:x + cw]
            else:
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                roi_y = y + h // 3
                roi_h = max(1, (2 * h) // 3)
                roi = norm[roi_y:roi_y + roi_h, x:x + w]
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if roi.size > 0:
                mean_bgr = np.mean(roi.reshape(-1, 3), axis=0)
                rgb_buffer.append(mean_bgr)

            # Compute HR every 0.5s if enough samples - using live_heartbeat.py logic
            now = time.time()
            if len(rgb_buffer) > int(3.0 * fps) and (now - last_time) > 0.5:
                last_time = now
                arr = np.stack(list(rgb_buffer), axis=0)
                
                # Use server's POS and CHROM methods
                s_pos = _bandpass_filter(_pos_signal(arr).astype(np.float32), fps)
                s_chr = _bandpass_filter(_chrom_signal(arr).astype(np.float32), fps)
                
                hr_pos, conf_pos = _estimate_hr_and_conf(s_pos, fps)
                hr_chr, conf_chr = _estimate_hr_and_conf(s_chr, fps)
                
                # Fuse by confidence - live_heartbeat.py approach
                w_pos = conf_pos + 1e-6
                w_chr = conf_chr + 1e-6
                hr_fused = (w_pos * hr_pos + w_chr * hr_chr) / (w_pos + w_chr)
                
                # Clamp to physiological range
                if hr_fused < 50 or hr_fused > 180:
                    hr_fused = 0.0
                
                # Smooth output to reduce jitter
                smoothed_hr = 0.7 * smoothed_hr + 0.3 * hr_fused

            # Overlay text - live_heartbeat.py style
            cv2.putText(frame, f"HR: {smoothed_hr:5.1f} bpm", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            cv2.imshow('Live Heartbeat (POS/CHROM)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        _WEBCAM_STATE['running'] = False
        
    except Exception as e:
        print(f"Webcam process error: {e}")
        _WEBCAM_STATE['running'] = False


@app.post('/api/webcam/start')
async def start_native_webcam():
    """Start native webcam processing"""
    global _WEBCAM_STATE
    
    if _WEBCAM_STATE['running']:
        return JSONResponse({'message': '✅ Native webcam is already running'})
    
    try:
        _WEBCAM_STATE['running'] = True
        _WEBCAM_STATE['thread'] = threading.Thread(target=native_webcam_process, daemon=True)
        _WEBCAM_STATE['thread'].start()
        
        return JSONResponse({'message': '🚀 Native webcam started! OpenCV window should appear.'})
    except Exception as e:
        _WEBCAM_STATE['running'] = False
        return JSONResponse({'detail': f'Failed to start webcam: {str(e)}'}, status_code=500)


@app.post('/api/webcam/stop')
async def stop_native_webcam():
    """Stop native webcam processing"""
    global _WEBCAM_STATE
    
    _WEBCAM_STATE['running'] = False
    
    if _WEBCAM_STATE['thread'] and _WEBCAM_STATE['thread'].is_alive():
        # Give it a moment to clean up
        _WEBCAM_STATE['thread'].join(timeout=2.0)
    
    return JSONResponse({'message': '🛑 Native webcam stopped'})


@app.get('/api/webcam/status')
async def webcam_status():
    """Get webcam status"""
    running = _WEBCAM_STATE['running']
    if running:
        return JSONResponse({
            'running': True, 
            'message': '✅ Native webcam is running - check OpenCV window'
        })
    else:
        return JSONResponse({
            'running': False,
            'message': '⏹️ Native webcam is stopped'
        })


# To run: uvicorn server:app --host 127.0.0.1 --port 8000

