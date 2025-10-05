"""
Enhanced face tracking and ROI optimization for improved heart rate detection.
Implements adaptive ROI selection, motion compensation, and quality-based tracking.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
from collections import deque


class EnhancedFaceTracker:
    """Enhanced face tracker with motion prediction and quality assessment."""
    
    def __init__(self, buffer_size: int = 10):
        self.face_buffer = deque(maxlen=buffer_size)
        self.roi_quality_history = deque(maxlen=5)
        self.motion_threshold = 10.0  # pixels
        self.quality_threshold = 0.3
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Kalman filter for face position prediction
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                 [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
        self.kalman.measurementNoiseCov = 10.0 * np.eye(2, dtype=np.float32)
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)
        
        self.kalman_initialized = False
        self.last_detection = None
        
    def detect_and_track_face(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """Detect and track face with motion prediction and quality assessment."""
        h, w = frame.shape[:2]
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,  # More sensitive detection
            minNeighbors=3,    # Reduced for better sensitivity
            minSize=(60, 60),  # Smaller minimum size
            maxSize=(int(w*0.8), int(h*0.8))  # Maximum size constraint
        )
        
        current_face = None
        
        if len(faces) > 0:
            # Select best face based on size and position stability
            if self.face_buffer:
                # Choose face closest to previous position
                last_face = self.face_buffer[-1]
                last_center = (last_face[0] + last_face[2]//2, last_face[1] + last_face[3]//2)
                
                best_face = None
                min_distance = float('inf')
                
                for face in faces:
                    center = (face[0] + face[2]//2, face[1] + face[3]//2)
                    distance = np.sqrt((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)
                    if distance < min_distance:
                        min_distance = distance
                        best_face = face
                
                # Use best face if motion is reasonable
                if min_distance < self.motion_threshold * 2:
                    current_face = best_face
                else:
                    # Large motion detected, use largest face
                    current_face = max(faces, key=lambda f: f[2] * f[3])
            else:
                # No history, use largest face
                current_face = max(faces, key=lambda f: f[2] * f[3])
        
        # If no face detected or poor quality, use prediction
        if current_face is None and self.kalman_initialized:
            predicted = self.kalman.predict()
            pred_x, pred_y = int(predicted[0]), int(predicted[1])
            
            # Use last known size if available
            if self.face_buffer:
                last_face = self.face_buffer[-1]
                pred_w, pred_h = last_face[2], last_face[3]
                
                # Ensure predicted face is within frame bounds
                pred_x = max(0, min(pred_x, w - pred_w))
                pred_y = max(0, min(pred_y, h - pred_h))
                
                current_face = (pred_x, pred_y, pred_w, pred_h)
        
        # If still no face, use fallback region
        if current_face is None:
            # Center region fallback
            fallback_w = int(w * 0.4)
            fallback_h = int(h * 0.4)
            fallback_x = (w - fallback_w) // 2
            fallback_y = (h - fallback_h) // 2
            current_face = (fallback_x, fallback_y, fallback_w, fallback_h)
        
        # Update Kalman filter
        if len(faces) > 0:  # Only update with actual detections
            center_x = current_face[0] + current_face[2] // 2
            center_y = current_face[1] + current_face[3] // 2
            
            if not self.kalman_initialized:
                self.kalman.statePre = np.array([center_x, center_y, 0, 0], dtype=np.float32)
                self.kalman.statePost = np.array([center_x, center_y, 0, 0], dtype=np.float32)
                self.kalman_initialized = True
            
            measurement = np.array([[center_x], [center_y]], dtype=np.float32)
            self.kalman.correct(measurement)
        elif self.kalman_initialized:
            # Predict without measurement
            self.kalman.predict()
        
        # Store face in buffer
        self.face_buffer.append(current_face)
        self.last_detection = current_face
        
        return current_face
    
    def get_optimal_roi(self, frame: np.ndarray, face_rect: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Get optimal ROI within face for heart rate detection."""
        x, y, w, h = face_rect
        
        # Use forehead and cheek regions (avoid eyes, nose, mouth)
        # Focus on the upper cheek area which typically has good blood flow
        roi_y = y + int(h * 0.25)  # Start from 25% down the face
        roi_h = int(h * 0.4)       # Use 40% of face height
        roi_x = x + int(w * 0.15)  # Start from 15% from face edge
        roi_w = int(w * 0.7)       # Use 70% of face width
        
        # Ensure ROI is within frame bounds
        frame_h, frame_w = frame.shape[:2]
        roi_x = max(0, min(roi_x, frame_w - roi_w))
        roi_y = max(0, min(roi_y, frame_h - roi_h))
        roi_w = min(roi_w, frame_w - roi_x)
        roi_h = min(roi_h, frame_h - roi_y)
        
        return (roi_x, roi_y, roi_w, roi_h)
    
    def assess_roi_quality(self, roi: np.ndarray) -> float:
        """Assess the quality of ROI for heart rate detection."""
        if roi.size == 0:
            return 0.0
        
        # 1. Check illumination consistency
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray)
        
        # Penalize very dark or very bright regions
        if mean_intensity < 50 or mean_intensity > 200:
            illumination_score = 0.3
        else:
            illumination_score = 1.0
        
        # 2. Check for sufficient texture/detail
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        texture_score = min(1.0, laplacian_var / 500.0)  # Normalize based on expected range
        
        # 3. Check color distribution (good skin regions have balanced RGB)
        mean_bgr = np.mean(roi.reshape(-1, 3), axis=0)
        b, g, r = mean_bgr
        
        # Healthy skin typically has r > g > b
        if r > g > b and r > 80:
            color_score = 1.0
        else:
            color_score = 0.5
        
        # 4. Check for saturation (avoid over/under saturated regions)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:, :, 1])
        
        if 30 < saturation < 180:  # Good saturation range
            saturation_score = 1.0
        else:
            saturation_score = 0.6
        
        # Combined quality score
        overall_quality = (
            0.3 * illumination_score +
            0.2 * texture_score +
            0.3 * color_score +
            0.2 * saturation_score
        )
        
        return float(np.clip(overall_quality, 0.0, 1.0))
    
    def get_adaptive_cheek_regions(self, roi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get adaptive left and right cheek regions with quality assessment."""
        h, w = roi.shape[:2]
        if h <= 0 or w <= 0:
            return np.array([]), np.array([])
        
        # Define cheek regions more precisely
        margin_w = int(0.05 * w)  # Smaller margins
        cheek_w = max(1, int(0.3 * w))  # Wider cheek regions
        cheek_h = max(1, int(0.6 * h))  # Taller cheek regions
        top = max(0, int(0.1 * h))
        
        left_x = margin_w
        right_x = w - margin_w - cheek_w
        y = top
        
        # Ensure coordinates are valid
        left_x = max(0, min(left_x, w - cheek_w))
        right_x = max(0, min(right_x, w - cheek_w))
        y = max(0, min(y, h - cheek_h))
        cheek_w = min(cheek_w, w - max(left_x, right_x))
        cheek_h = min(cheek_h, h - y)
        
        if cheek_w <= 0 or cheek_h <= 0:
            return np.array([]), np.array([])
        
        left_cheek = roi[y:y + cheek_h, left_x:left_x + cheek_w]
        right_cheek = roi[y:y + cheek_h, right_x:right_x + cheek_w]
        
        # Assess quality of each cheek region
        left_quality = self.assess_roi_quality(left_cheek) if left_cheek.size > 0 else 0.0
        right_quality = self.assess_roi_quality(right_cheek) if right_cheek.size > 0 else 0.0
        
        # Store quality history
        avg_quality = (left_quality + right_quality) / 2
        self.roi_quality_history.append(avg_quality)
        
        return left_cheek, right_cheek
    
    def extract_signal_with_quality(self, roi: np.ndarray) -> Tuple[np.ndarray, float]:
        """Extract color signal from ROI with quality assessment."""
        if roi.size == 0:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.0
        
        left_cheek, right_cheek = self.get_adaptive_cheek_regions(roi)
        
        if left_cheek.size == 0 and right_cheek.size == 0:
            # Fallback to whole ROI
            mean_bgr = np.mean(roi.reshape(-1, 3), axis=0).astype(np.float32)
            quality = self.assess_roi_quality(roi)
        else:
            # Combine cheek regions with quality weighting
            regions = []
            weights = []
            
            if left_cheek.size > 0:
                left_mean = np.mean(left_cheek.reshape(-1, 3), axis=0)
                left_quality = self.assess_roi_quality(left_cheek)
                regions.append(left_mean)
                weights.append(left_quality)
            
            if right_cheek.size > 0:
                right_mean = np.mean(right_cheek.reshape(-1, 3), axis=0)
                right_quality = self.assess_roi_quality(right_cheek)
                regions.append(right_mean)
                weights.append(right_quality)
            
            if regions:
                weights = np.array(weights)
                weights = weights / (np.sum(weights) + 1e-8)  # Normalize weights
                mean_bgr = np.average(regions, weights=weights, axis=0).astype(np.float32)
                quality = np.mean([self.assess_roi_quality(left_cheek) if left_cheek.size > 0 else 0,
                                 self.assess_roi_quality(right_cheek) if right_cheek.size > 0 else 0])
            else:
                mean_bgr = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                quality = 0.0
        
        return mean_bgr, quality
    
    def get_stability_score(self) -> float:
        """Get stability score based on recent face tracking history."""
        if len(self.face_buffer) < 3:
            return 0.5  # Moderate stability for insufficient history
        
        # Calculate position variance
        positions = [(face[0] + face[2]//2, face[1] + face[3]//2) for face in self.face_buffer]
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        x_var = np.var(x_coords)
        y_var = np.var(y_coords)
        
        # Calculate size variance
        sizes = [face[2] * face[3] for face in self.face_buffer]
        size_var = np.var(sizes)
        
        # Normalize variances to [0, 1] range
        max_expected_pos_var = 100.0  # pixels^2
        max_expected_size_var = 1000.0  # pixels^2
        
        pos_stability = 1.0 - min(1.0, (x_var + y_var) / max_expected_pos_var)
        size_stability = 1.0 - min(1.0, size_var / max_expected_size_var)
        
        # Quality stability based on ROI quality history
        quality_stability = 1.0
        if len(self.roi_quality_history) > 1:
            quality_var = np.var(list(self.roi_quality_history))
            quality_stability = 1.0 - min(1.0, quality_var / 0.1)  # Max expected quality variance
        
        # Combined stability score
        overall_stability = (
            0.4 * pos_stability +
            0.3 * size_stability +
            0.3 * quality_stability
        )
        
        return float(np.clip(overall_stability, 0.0, 1.0))


class MotionArtifactDetector:
    """Detect and compensate for motion artifacts in the signal."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.signal_history = deque(maxlen=window_size)
        self.motion_history = deque(maxlen=window_size)
        
    def detect_motion_artifacts(self, current_signal: np.ndarray, 
                               face_stability: float) -> bool:
        """Detect if current signal contains motion artifacts."""
        if len(self.signal_history) < 3:
            self.signal_history.append(current_signal)
            return False
        
        # 1. Check signal magnitude changes
        recent_signals = list(self.signal_history)[-3:]
        signal_changes = []
        
        for i in range(len(recent_signals) - 1):
            change = np.linalg.norm(recent_signals[i+1] - recent_signals[i])
            signal_changes.append(change)
        
        avg_change = np.mean(signal_changes)
        current_change = np.linalg.norm(current_signal - recent_signals[-1])
        
        # Motion artifact if current change is much larger than recent average
        change_threshold = 3.0 * avg_change + 0.1  # Adaptive threshold
        magnitude_artifact = current_change > change_threshold
        
        # 2. Check face stability
        stability_artifact = face_stability < 0.3
        
        # 3. Check signal consistency across color channels
        signal_std = np.std(current_signal)
        signal_mean = np.mean(current_signal)
        cv = signal_std / (abs(signal_mean) + 1e-8)
        consistency_artifact = cv > 2.0  # High coefficient of variation
        
        # Combined artifact detection
        motion_detected = magnitude_artifact or stability_artifact or consistency_artifact
        
        self.signal_history.append(current_signal)
        self.motion_history.append(motion_detected)
        
        return motion_detected
    
    def get_clean_signal(self, current_signal: np.ndarray, 
                        motion_detected: bool) -> np.ndarray:
        """Get motion-compensated signal."""
        if not motion_detected or len(self.signal_history) < 2:
            return current_signal
        
        # Use median filter for motion compensation
        if len(self.signal_history) >= 5:
            recent_signals = np.array(list(self.signal_history)[-5:])
            median_signal = np.median(recent_signals, axis=0)
            
            # Blend current signal with median for smooth compensation
            alpha = 0.3  # Weight for current signal
            compensated_signal = alpha * current_signal + (1 - alpha) * median_signal
            
            return compensated_signal.astype(np.float32)
        
        return current_signal
    
    def get_motion_score(self) -> float:
        """Get recent motion artifact score (0 = no motion, 1 = high motion)."""
        if len(self.motion_history) == 0:
            return 0.0
        
        recent_motion = list(self.motion_history)[-5:]  # Last 5 frames
        motion_ratio = sum(recent_motion) / len(recent_motion)
        
        return float(motion_ratio)
