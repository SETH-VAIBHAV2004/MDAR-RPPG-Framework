import time
import os.path as osp
import argparse
from collections import deque
from typing import Tuple, Deque, Optional, List

import cv2
import numpy as np

# Import enhanced tracking modules
try:
    from enhanced_tracking import EnhancedFaceTracker, MotionArtifactDetector
except ImportError:
    print("Warning: Enhanced tracking module not found. Using basic tracking.")
    EnhancedFaceTracker = None
    MotionArtifactDetector = None

# Global camera switching state
_CAMERA_STATE = {
    'available_cameras': [],
    'current_camera_idx': 0,
    'current_camera_id': None,
    'camera_info': ''
}


def list_available_cameras() -> List[Tuple[int, str]]:
    """List all available cameras with their indices and descriptions."""
    available_cameras = []
    
    # Test up to 10 camera indices
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to get camera name/description
            backend = cap.getBackendName()
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Try to read a frame to ensure camera is working
            ret, frame = cap.read()
            if ret:
                desc = f"{backend} - {width}x{height}@{fps:.1f}fps"
                available_cameras.append((i, desc))
            
            cap.release()
    
    return available_cameras


def init_camera_state() -> None:
    """Initialize camera state with available cameras."""
    global _CAMERA_STATE
    _CAMERA_STATE['available_cameras'] = list_available_cameras()
    if _CAMERA_STATE['available_cameras']:
        _CAMERA_STATE['current_camera_idx'] = 0
        _CAMERA_STATE['current_camera_id'] = _CAMERA_STATE['available_cameras'][0][0]
        _CAMERA_STATE['camera_info'] = f"Cam {_CAMERA_STATE['current_camera_id']}: {_CAMERA_STATE['available_cameras'][0][1]}"


def switch_camera(cap: cv2.VideoCapture, direction: int = 1) -> cv2.VideoCapture:
    """Switch to next/previous camera with improved error handling and timeout.
    
    Args:
        cap: Current camera capture object
        direction: 1 for next camera, -1 for previous camera
    
    Returns:
        New camera capture object or the same if switching fails
    """
    import threading
    import time
    
    global _CAMERA_STATE
    
    if not _CAMERA_STATE['available_cameras']:
        print("No cameras available for switching")
        return cap
    
    # Calculate next camera index
    num_cameras = len(_CAMERA_STATE['available_cameras'])
    new_idx = (_CAMERA_STATE['current_camera_idx'] + direction) % num_cameras
    new_camera_id = _CAMERA_STATE['available_cameras'][new_idx][0]
    
    print(f"\nSwitching from camera {_CAMERA_STATE['current_camera_id']} to camera {new_camera_id}...")
    
    # Store current state for fallback
    fallback_idx = _CAMERA_STATE['current_camera_idx']
    fallback_id = _CAMERA_STATE['current_camera_id']
    
    # Release current camera with timeout
    def release_camera_safe(camera_obj):
        """Safe camera release with timeout"""
        if camera_obj is not None:
            try:
                camera_obj.release()
            except Exception as e:
                print(f"Warning: Error releasing camera: {e}")
    
    # Release current camera in separate thread with timeout
    if cap is not None:
        release_thread = threading.Thread(target=release_camera_safe, args=(cap,), daemon=True)
        release_thread.start()
        release_thread.join(timeout=1.0)  # 1 second timeout for release
    
    # Function to try opening a camera with timeout
    def try_open_camera(camera_id, timeout=3.0):
        """Try to open camera with timeout"""
        result = {'cap': None, 'success': False}
        
        def open_camera():
            try:
                new_cap = cv2.VideoCapture(camera_id)
                if new_cap.isOpened():
                    # Test if we can actually read a frame
                    ret, frame = new_cap.read()
                    if ret and frame is not None:
                        result['cap'] = new_cap
                        result['success'] = True
                        return
                # If we get here, camera didn't work properly
                if new_cap.isOpened():
                    new_cap.release()
            except Exception as e:
                print(f"Error opening camera {camera_id}: {e}")
        
        thread = threading.Thread(target=open_camera, daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        
        return result['cap'], result['success']
    
    # Try to open new camera
    new_cap, success = try_open_camera(new_camera_id, timeout=3.0)
    
    if success and new_cap is not None:
        # Success! Update camera state
        _CAMERA_STATE['current_camera_idx'] = new_idx
        _CAMERA_STATE['current_camera_id'] = new_camera_id
        _CAMERA_STATE['camera_info'] = f"Cam {new_camera_id}: {_CAMERA_STATE['available_cameras'][new_idx][1]}"
        print(f"Successfully switched to {_CAMERA_STATE['camera_info']}")
        return new_cap
    
    # Failed to open new camera, try fallback strategies
    print(f"Failed to open camera {new_camera_id}. Trying fallback options...")
    
    # Strategy 1: Try to reopen the previous camera
    fallback_cap, fallback_success = try_open_camera(fallback_id, timeout=2.0)
    if fallback_success and fallback_cap is not None:
        print(f"Reopened previous camera {fallback_id}")
        return fallback_cap
    
    # Strategy 2: Try other available cameras
    for attempt_idx, (attempt_id, attempt_desc) in enumerate(_CAMERA_STATE['available_cameras']):
        if attempt_id == new_camera_id or attempt_id == fallback_id:
            continue  # Skip cameras we already tried
        
        print(f"Trying camera {attempt_id} as fallback...")
        attempt_cap, attempt_success = try_open_camera(attempt_id, timeout=2.0)
        if attempt_success and attempt_cap is not None:
            _CAMERA_STATE['current_camera_idx'] = attempt_idx
            _CAMERA_STATE['current_camera_id'] = attempt_id
            _CAMERA_STATE['camera_info'] = f"Cam {attempt_id}: {attempt_desc}"
            print(f"Switched to fallback camera {_CAMERA_STATE['camera_info']}")
            return attempt_cap
    
    # Strategy 3: Re-scan for available cameras
    print("Rescanning for available cameras...")
    _CAMERA_STATE['available_cameras'] = list_available_cameras()
    
    if _CAMERA_STATE['available_cameras']:
        first_id = _CAMERA_STATE['available_cameras'][0][0]
        first_cap, first_success = try_open_camera(first_id, timeout=2.0)
        if first_success and first_cap is not None:
            _CAMERA_STATE['current_camera_idx'] = 0
            _CAMERA_STATE['current_camera_id'] = first_id
            _CAMERA_STATE['camera_info'] = f"Cam {first_id}: {_CAMERA_STATE['available_cameras'][0][1]}"
            print(f"Switched to rescanned camera {_CAMERA_STATE['camera_info']}")
            return first_cap
    
    # Last resort: return a dummy camera object
    print("All camera switching strategies failed. Creating dummy camera.")
    dummy_cap = cv2.VideoCapture()  # This will not be opened
    return dummy_cap


def switch_to_camera_index(cap: cv2.VideoCapture, camera_idx: int) -> cv2.VideoCapture:
    """Switch to a specific camera by index in available cameras list.
    
    Args:
        cap: Current camera capture object
        camera_idx: Index in the available cameras list (0-based)
    
    Returns:
        New camera capture object or the same if switching fails
    """
    global _CAMERA_STATE
    
    if not _CAMERA_STATE['available_cameras']:
        print("No cameras available for switching")
        return cap
    
    if camera_idx < 0 or camera_idx >= len(_CAMERA_STATE['available_cameras']):
        print(f"Invalid camera index {camera_idx}. Available cameras: 0-{len(_CAMERA_STATE['available_cameras'])-1}")
        return cap
    
    if camera_idx == _CAMERA_STATE['current_camera_idx']:
        print(f"Already using camera {camera_idx}")
        return cap
    
    new_camera_id = _CAMERA_STATE['available_cameras'][camera_idx][0]
    
    print(f"\nSwitching from camera {_CAMERA_STATE['current_camera_id']} to camera {new_camera_id}...")
    
    # Release current camera
    if cap is not None:
        cap.release()
    
    # Try to open new camera
    new_cap = cv2.VideoCapture(new_camera_id)
    if not new_cap.isOpened():
        print(f"Failed to open camera {new_camera_id}. Trying to reopen previous camera...")
        # Try to reopen the previous camera
        fallback_cap = cv2.VideoCapture(_CAMERA_STATE['current_camera_id'])
        if fallback_cap.isOpened():
            print(f"Reopened camera {_CAMERA_STATE['current_camera_id']}")
            return fallback_cap
        else:
            print("Failed to reopen previous camera. Continuing with failed camera.")
            return new_cap  # Return even if not opened, let main loop handle it
    
    # Update camera state
    _CAMERA_STATE['current_camera_idx'] = camera_idx
    _CAMERA_STATE['current_camera_id'] = new_camera_id
    _CAMERA_STATE['camera_info'] = f"Cam {new_camera_id}: {_CAMERA_STATE['available_cameras'][camera_idx][1]}"
    
    print(f"Successfully switched to {_CAMERA_STATE['camera_info']}")
    return new_cap


def print_camera_list() -> None:
    """Print list of available cameras."""
    global _CAMERA_STATE
    
    if not _CAMERA_STATE['available_cameras']:
        print("No cameras available")
        return
    
    print("\nAvailable cameras:")
    for idx, (cam_id, desc) in enumerate(_CAMERA_STATE['available_cameras']):
        marker = " <- CURRENT" if idx == _CAMERA_STATE['current_camera_idx'] else ""
        print(f"  {idx}: Camera {cam_id} - {desc}{marker}")
    print()


def select_camera() -> int:
    """Interactive camera selection."""
    cameras = list_available_cameras()
    
    if not cameras:
        print("No cameras found!")
        return -1
    
    print("\nAvailable cameras:")
    for idx, (cam_id, desc) in enumerate(cameras):
        print(f"  {idx}: Camera {cam_id} - {desc}")
    
    while True:
        try:
            choice = input(f"\nSelect camera (0-{len(cameras)-1}): ").strip()
            choice_idx = int(choice)
            if 0 <= choice_idx < len(cameras):
                return cameras[choice_idx][0]
            else:
                print(f"Please enter a number between 0 and {len(cameras)-1}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            return -1


def illumination_normalize(frame: np.ndarray, enable_clahe: bool = True) -> np.ndarray:
    """Normalize illumination with optional CLAHE."""
    if not enable_clahe:
        return frame
    
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def detect_face_roi(frame: np.ndarray, face_detector: cv2.CascadeClassifier) -> Tuple[int, int, int, int]:
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


def average_cheeks(bgr: np.ndarray) -> np.ndarray:
    """Average color over two cheek sub-ROIs to reduce motion sensitivity.
    Returns mean BGR vector of concatenated cheeks.
    """
    h, w = bgr.shape[:2]
    if h <= 0 or w <= 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    # Define left/right cheek boxes inside the face ROI
    # Exclude center (nose/mouth) region, focus on lateral cheeks
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


def detrend_and_normalize(rgb_bgr: np.ndarray) -> np.ndarray:
    b = rgb_bgr[:, 0]
    g = rgb_bgr[:, 1]
    r = rgb_bgr[:, 2]
    X = np.stack([r, g, b], axis=1).astype(np.float64)
    X = X - np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True) + 1e-8
    X = X / std
    return X


def pos_method(rgb_bgr: np.ndarray) -> np.ndarray:
    X = detrend_and_normalize(rgb_bgr)
    H = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    S = X @ H.T
    s1 = S[:, 0]
    s2 = S[:, 1]
    alpha = (np.std(s1) + 1e-8) / (np.std(s2) + 1e-8)
    s = s1 + alpha * s2
    return s - np.mean(s)


def chrom_method(rgb_bgr: np.ndarray) -> np.ndarray:
    X = detrend_and_normalize(rgb_bgr)
    r = X[:, 0]
    g = X[:, 1]
    b = X[:, 2]
    x = 3 * r - 2 * g
    y = 1.5 * r + g - 1.5 * b
    alpha = (np.std(x) + 1e-8) / (np.std(y) + 1e-8)
    s = x - alpha * y
    return s - np.mean(s)


def bandpass(x: np.ndarray, fps: float, fmin: float = 0.7, fmax: float = 4.0) -> np.ndarray:
    # Wider physiological band for better real-time response: 42-240 bpm (0.7-4.0 Hz)
    if len(x) < 8:
        return x
    
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / float(fps))
    mask = (freqs >= fmin) & (freqs <= fmax)
    Xf = X * mask
    y = np.fft.irfft(Xf, n=len(x))
    return y.real.astype(np.float32)


def quadratic_peak_interpolate(mag: np.ndarray, idx: int) -> float:
    """Parabolic interpolation around peak idx -> sub-bin offset in [-0.5, 0.5]."""
    if idx <= 0 or idx >= len(mag) - 1:
        return 0.0
    y0, y1, y2 = mag[idx - 1], mag[idx], mag[idx + 1]
    denom = (y0 - 2 * y1 + y2)
    if abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (y0 - y2) / denom


def estimate_hr_fft(signal: np.ndarray, fps: float, fmin: float = 0.7, fmax: float = 4.0) -> Tuple[float, float]:
    """Estimate heart rate with balanced accuracy and responsiveness."""
    if len(signal) < 16:  # Reduced minimum samples for faster response
        return 0.0, 0.0
    
    # Basic detrending
    signal = signal - np.mean(signal)
    
    # Light windowing to reduce spectral leakage
    win = np.hanning(len(signal))
    sigw = signal * win
    
    # FFT with moderate zero-padding
    spec = np.fft.rfft(sigw)
    freqs = np.fft.rfftfreq(len(sigw), d=1.0 / float(fps))
    
    # Physiological band: 42-240 bpm (0.7-4.0 Hz)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band):
        return 0.0, 0.0
    
    mag = (np.abs(spec) ** 2).astype(np.float64)
    band_idxs = np.where(band)[0]
    local_mag = mag[band]
    
    # Find the strongest peak
    peak_local = int(np.argmax(local_mag))
    peak_idx = band_idxs[peak_local]
    
    # Quadratic interpolation for sub-bin frequency
    delta = quadratic_peak_interpolate(mag, peak_idx)
    peak_freq = (freqs[peak_idx] + delta * (fps / len(sigw)))
    peak_power = float(mag[peak_idx])
    
    # Convert to BPM
    hr = float(peak_freq * 60.0)
    
    # Only reject truly implausible values
    if hr < 30 or hr > 250:
        return 0.0, 0.0
    
    return hr, peak_power


def compute_confidence(mag: np.ndarray, band: np.ndarray) -> float:
    # Confidence = peak power / total band power (SNR-like)
    if not np.any(band):
        return 0.0
    band_mag = np.abs(mag[band]) ** 2 if np.iscomplexobj(mag) else mag[band]
    total = float(np.sum(band_mag) + 1e-8)
    peak = float(np.max(band_mag))
    return float(peak / total)


def estimate_hr_fft_enhanced(signal: np.ndarray, fps: float, fmin: float = 0.75, fmax: float = 3.5) -> Tuple[float, float]:
    """Enhanced HR estimation with improved peak detection and validation."""
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
    delta = quadratic_peak_interpolate(mag, primary_peak_idx)
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


def assess_signal_quality(signal: np.ndarray, fps: float) -> float:
    """Assess signal quality based on multiple metrics."""
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


def calculate_measurement_confidence(hr_estimate: float, all_estimates: List[float], 
                                   weights: List[float], hr_history: Deque[float]) -> float:
    """Calculate confidence in a heart rate measurement based on multiple factors."""
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


def main(camera_id: int = None, enable_clahe: bool = True, 
         window_seconds: float = 6.0, history_length: int = 12) -> None:
    """Main function with enhanced real-time HR detection."""
    
    # Initialize camera state
    init_camera_state()
    
    # Camera selection
    if camera_id is None:
        camera_id = select_camera()
        if camera_id == -1:
            print("No camera selected or available.")
            return
    
    # Update camera state with selected camera
    global _CAMERA_STATE
    for idx, (cam_id, desc) in enumerate(_CAMERA_STATE['available_cameras']):
        if cam_id == camera_id:
            _CAMERA_STATE['current_camera_idx'] = idx
            _CAMERA_STATE['current_camera_id'] = camera_id
            _CAMERA_STATE['camera_info'] = f"Cam {camera_id}: {desc}"
            break
    
    print(f"\nOpening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f'Failed to open camera {camera_id}.')
        return

    # Optimize camera settings for performance
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for lower latency
    
    # Try to estimate FPS from camera; default to 30
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 1.0 or np.isnan(fps):
        fps = 30.0
    
    print(f"Camera FPS: {fps}")
    print(f"CLAHE normalization: {'Enabled' if enable_clahe else 'Disabled'}")
    print(f"Window size: {window_seconds}s, History length: {history_length}")
    print(f"Available cameras: {len(_CAMERA_STATE['available_cameras'])}")
    print("\nKeyboard controls:")
    print("  q - Quit")
    print("  c - Toggle CLAHE normalization")
    print("  r - Reset buffers")
    print("  n - Next camera")
    print("  p - Previous camera")
    print("  0-9 - Direct camera selection (by list index)")
    print("  l - List all available cameras")
    print()

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Sliding buffers for signal processing
    max_len = int(window_seconds * fps)
    rgb_buffer: Deque[np.ndarray] = deque(maxlen=max_len)

    last_time = time.time()
    smoothed_hr = 0.0

    # History tracking for adaptive smoothing
    hr_history: Deque[float] = deque(maxlen=history_length)
    current_clahe = enable_clahe

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        norm = illumination_normalize(frame, current_clahe)
        x, y, w, h = detect_face_roi(norm, face_detector)
        roi = norm[y:y + h, x:x + w]

        # Draw face/cheek ROIs
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 1)
        if roi.size > 0:
            h_r, w_r = roi.shape[:2]
            margin_w = int(0.1 * w_r)
            cheek_w = max(1, int(0.25 * w_r))
            cheek_h = max(1, int(0.45 * h_r))
            top = max(0, int(0.1 * h_r))
            left_x = x + margin_w
            right_x = x + w_r - margin_w - cheek_w
            top_y = y + top
            cv2.rectangle(frame, (left_x, top_y), (left_x + cheek_w, top_y + cheek_h), (0, 255, 255), 1)
            cv2.rectangle(frame, (right_x, top_y), (right_x + cheek_w, top_y + cheek_h), (0, 255, 255), 1)

        if roi.size > 0:
            mean_bgr = average_cheeks(roi)
            rgb_buffer.append(mean_bgr)

        # Adaptive real-time HR computation with reduced latency
        now = time.time()
        min_samples = max(int(2.5 * fps), 60)  # Minimum 2.5 seconds of data
        update_interval = 0.25  # Update every 250ms for better responsiveness
        
        if len(rgb_buffer) >= min_samples and (now - last_time) > update_interval:
            last_time = now
            
            # Use sliding window for real-time processing
            buffer_size = len(rgb_buffer)
            use_samples = min(buffer_size, int(window_seconds * fps))  # Don't exceed window size
            
            # Get the most recent samples
            recent_data = list(rgb_buffer)[-use_samples:]
            arr = np.stack(recent_data, axis=0)

            # Enhanced signal processing with adaptive parameters
            s_pos = bandpass(pos_method(arr), fps, fmin=0.75, fmax=3.5)  # Tighter frequency band
            s_chr = bandpass(chrom_method(arr), fps, fmin=0.75, fmax=3.5)

            # Multi-scale HR estimation for better accuracy
            hr_pos, p_pos = estimate_hr_fft_enhanced(s_pos, fps)
            hr_chr, p_chr = estimate_hr_fft_enhanced(s_chr, fps)
            
            # Quality assessment for each method
            pos_quality = assess_signal_quality(s_pos, fps)
            chr_quality = assess_signal_quality(s_chr, fps)
            
            # Adaptive weighting based on signal quality and power
            valid_estimates = []
            weights = []
            
            # More lenient validation for real-time processing
            if 45 <= hr_pos <= 180 and p_pos > 0 and pos_quality > 0.2:
                valid_estimates.append(hr_pos)
                weights.append(p_pos * pos_quality)
            
            if 45 <= hr_chr <= 180 and p_chr > 0 and chr_quality > 0.2:
                valid_estimates.append(hr_chr)
                weights.append(p_chr * chr_quality)
            
            # Advanced fusion algorithm
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
                
                # Additional physiological and signal quality constraints
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
            else:
                hr_fused = 0.0

            # Adaptive smoothing with momentum and confidence weighting
            if hr_fused > 0:
                # Calculate confidence based on multiple factors
                confidence = calculate_measurement_confidence(hr_fused, valid_estimates, weights, hr_history)
                
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
            
            # Track recent HRs for stability monitoring with confidence weighting
            if hr_fused > 0:
                hr_history.append(hr_fused)

        # Display smoothed HR with stability indicator
        display_hr = smoothed_hr
        
        # Determine status based on recent stability
        if len(hr_history) >= 5:
            recent_std = np.std(list(hr_history)[-5:])
            if recent_std < 3.0:
                status = 'STABLE'
            elif recent_std < 6.0:
                status = 'TRACKING'
            else:
                status = 'VARIABLE'
        else:
            status = 'STARTING'

        # Enhanced overlay text with more info
        cv2.putText(frame, f"HR: {display_hr:5.1f} bpm ({status})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, f"{_CAMERA_STATE.get('camera_info','')}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)
        cv2.putText(frame, f"CLAHE: {'ON' if current_clahe else 'OFF'} | Buffer: {len(rgb_buffer)}/{max_len}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"History: {len(hr_history)}/{history_length}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "Keys: q quit | c CLAHE | r reset | n/p next/prev cam | 0-9 select | l list", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Live Heartbeat (POS/CHROM)', frame)
        
        # Enhanced keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            current_clahe = not current_clahe
            print(f"\nCLAHE toggled: {'ON' if current_clahe else 'OFF'}")
        elif key == ord('r'):
            hr_history.clear()
            rgb_buffer.clear()
            smoothed_hr = 0.0
            print("\nReset: Buffers cleared")
        elif key == ord('l'):
            print_camera_list()
        elif key == ord('n'):
            # Next camera
            cap = switch_camera(cap, direction=1)
            # Re-init FPS and buffers
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 1.0 or np.isnan(fps):
                fps = 30.0
            max_len = int(window_seconds * fps)
            rgb_buffer = deque(maxlen=max_len)
            hr_history.clear()
            smoothed_hr = 0.0
        elif key == ord('p'):
            # Previous camera
            cap = switch_camera(cap, direction=-1)
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 1.0 or np.isnan(fps):
                fps = 30.0
            max_len = int(window_seconds * fps)
            rgb_buffer = deque(maxlen=max_len)
            hr_history.clear()
            smoothed_hr = 0.0
        elif key in [ord(str(d)) for d in range(10)]:
            # Direct select by index 0-9 if available
            idx = int(chr(key))
            cap = switch_to_camera_index(cap, idx)
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 1.0 or np.isnan(fps):
                fps = 30.0
            max_len = int(window_seconds * fps)
            rgb_buffer = deque(maxlen=max_len)
            hr_history.clear()
            smoothed_hr = 0.0
            

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Live Heart Rate Detection with Camera Selection')
    parser.add_argument('--camera', type=int, default=None, help='Camera ID (if not specified, interactive selection)')
    parser.add_argument('--no-clahe', action='store_true', help='Disable CLAHE normalization')
    parser.add_argument('--window', type=float, default=8.0, help='Window size in seconds (default: 8.0)')
    parser.add_argument('--history-length', type=int, default=16, help='HR history buffer length (default: 16)')
    
    args = parser.parse_args()
    
    main(
        camera_id=args.camera,
        enable_clahe=not args.no_clahe,
        window_seconds=args.window,
        history_length=args.history_length
    )

