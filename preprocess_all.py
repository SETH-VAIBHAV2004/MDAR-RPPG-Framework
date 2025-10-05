import os
import os.path as osp
import glob
import json
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from Datasets.utils import find_pairs_by_glob, load_gt_auto, resample_to_length


def ensure_dir(path: str) -> None:
    if not osp.exists(path):
        os.makedirs(path, exist_ok=True)


def detect_face_roi(frame: np.ndarray, face_detector: cv2.CascadeClassifier) -> Optional[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    # Upper face/cheek ROI: crop lower 2/3 of the detected box to avoid eyes/forehead glare
    roi_y = y + h // 3
    roi_h = max(1, (2 * h) // 3)
    return int(x), int(roi_y), int(w), int(roi_h)


def illumination_normalize(frame: np.ndarray) -> np.ndarray:
    # Apply CLAHE on the L channel in LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def suppress_motion_artifacts(signal: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    if signal.size == 0:
        return signal
    # Median filter to suppress impulsive motion artifacts
    k = max(3, kernel_size | 1)
    med = cv2.medianBlur(signal.astype(np.float32).reshape(-1, 1), k).reshape(-1)
    # High-pass (detrend) using moving average subtraction
    win = min(51, max(3, (len(signal) // 20) | 1))
    mov_avg = cv2.blur(med.reshape(-1, 1), (1, win)).reshape(-1)
    detrended = med - mov_avg
    return detrended


def bandpass_fft(x: np.ndarray, fps: float, fmin: float = 0.7, fmax: float = 4.0) -> np.ndarray:
    if x is None or len(x) < 8:
        return x
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / float(fps))
    mask = (freqs >= fmin) & (freqs <= fmax)
    Xf = X * mask
    y = np.fft.irfft(Xf, n=len(x))
    return y.astype(np.float32)


def chunk_frames(num_frames: int, chunk_len: int, hop_len: int) -> List[Tuple[int, int]]:
    chunks: List[Tuple[int, int]] = []
    start = 0
    while start + chunk_len <= num_frames:
        chunks.append((start, start + chunk_len))
        start += hop_len
    if not chunks and num_frames > 0:
        chunks.append((0, num_frames))
    return chunks


def random_augment(frames: List[np.ndarray], rng: np.random.Generator) -> List[np.ndarray]:
    aug_frames = []
    # Random brightness and slight color jitter
    alpha = float(rng.uniform(0.9, 1.1))
    beta = float(rng.uniform(-10, 10))
    # Random horizontal flip
    do_flip = bool(rng.random() < 0.5)
    for f in frames:
        g = cv2.convertScaleAbs(f, alpha=alpha, beta=beta)
        if do_flip:
            g = cv2.flip(g, 1)
        aug_frames.append(g)
    return aug_frames


def extract_spatiotemporal_features(frames: List[np.ndarray]) -> np.ndarray:
    # Simple features: mean color over ROI per frame (BGR) + temporal differences
    if len(frames) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    means = np.array([np.mean(f.reshape(-1, 3), axis=0) for f in frames], dtype=np.float32)
    # Temporal diff magnitude of green channel as motion proxy
    green = means[:, 1]
    diff = np.zeros_like(green)
    if len(green) > 1:
        diff[1:] = np.abs(np.diff(green))
    feats = np.concatenate([means, diff[:, None]], axis=1)
    return feats


def synchronize_signal_to_frames(gt_signal: Optional[np.ndarray], num_frames: int) -> Optional[np.ndarray]:
    if gt_signal is None:
        return None
    return resample_to_length(gt_signal, num_frames)


def visualize_qc(video_path: str, frame_indices: List[int], rois: List[Tuple[int, int, int, int]], signal: Optional[np.ndarray], out_png: str) -> None:
    cols = min(5, len(frame_indices))
    rows = int(np.ceil(len(frame_indices) / max(1, cols)))
    plt.figure(figsize=(3 * cols, 2.5 * rows))
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i, fi in enumerate(frame_indices):
        if fi < 0 or fi >= total:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        x, y, w, h = rois[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        ax.set_title(f'Frame {fi}')
    cap.release()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    # If signal provided, plot it too
    if signal is not None and signal.size > 0:
        plt.figure(figsize=(8, 2.5))
        plt.plot(signal)
        plt.title('Synchronized signal (resampled)')
        plt.tight_layout()
        plt.savefig(osp.splitext(out_png)[0] + '_signal.png', dpi=150)
        plt.close()


def process_video_pair(video_path: str,
                       gt_path: Optional[str],
                       out_dir: str,
                       rng: np.random.Generator,
                       config: Dict) -> None:
    ensure_dir(out_dir)
    qc_dir = osp.join(out_dir, 'qc')
    ensure_dir(qc_dir)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

    gt_signal, _ = load_gt_auto(gt_path) if gt_path is not None else (None, None)

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    rois: List[Tuple[int, int, int, int]] = []
    roi_frames: List[np.ndarray] = []
    rgb_means_all: List[np.ndarray] = []

    for _ in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frame_norm = illumination_normalize(frame)
        bbox = detect_face_roi(frame_norm, face_detector)
        if bbox is None:
            # fallback to center crop
            h, w = frame_norm.shape[:2]
            cw = int(w * 0.4)
            ch = int(h * 0.4)
            x = (w - cw) // 2
            y = (h - ch) // 2
            bbox = (x, y, cw, ch)
        x, y, w, h = bbox
        roi = frame_norm[y:y + h, x:x + w]
        rois.append(bbox)
        roi_frames.append(roi)
        rgb_means_all.append(np.mean(roi.reshape(-1, 3), axis=0))
    cap.release()

    # Aggregate green channel mean as raw PPG-like signal
    raw_signal = np.array([m[1] for m in rgb_means_all], dtype=np.float32)
    motion_suppressed = suppress_motion_artifacts(raw_signal, kernel_size=5)
    green_bp = bandpass_fft(motion_suppressed, fps)

    # Chunk frames
    chunk_len = int(config.get('chunk_seconds', 10) * fps)
    hop_len = int(config.get('hop_seconds', 5) * fps)
    chunks = chunk_frames(len(roi_frames), chunk_len=chunk_len, hop_len=hop_len)

    features_dir = osp.join(out_dir, 'features')
    ensure_dir(features_dir)

    for ci, (s, e) in enumerate(chunks):
        frames_chunk = roi_frames[s:e]
        # Optional augmentation for training set
        if config.get('do_augment', True):
            frames_chunk = random_augment(frames_chunk, rng)

        feats = extract_spatiotemporal_features(frames_chunk)
        # Replace diff with bandpassed diff from global green_bp aligned to chunk
        if feats.shape[0] == (e - s):
            diff = np.zeros((feats.shape[0],), dtype=np.float32)
            if (e - s) > 1:
                seg = green_bp[s:e]
                diff[1:] = np.abs(np.diff(seg))
            feats[:, 3] = diff

        # Sync ground-truth to this chunk
        gt_sync = None
        if gt_signal is not None:
            gt_chunk = gt_signal
            if len(gt_signal) != len(roi_frames):
                gt_chunk = synchronize_signal_to_frames(gt_signal, len(roi_frames))
            gt_sync = gt_chunk[s:e]

        # Save features and labels
        out_npz = osp.join(features_dir, f'chunk_{ci:04d}.npz')
        np.savez_compressed(out_npz,
                            features=feats.astype(np.float32),
                            start_frame=int(s),
                            end_frame=int(e),
                            fps=float(fps),
                            gt=gt_sync.astype(np.float32) if gt_sync is not None else None,
                            rgb_means=np.asarray(rgb_means_all[s:e], dtype=np.float32))

    # QC visualization on a few evenly spaced frames
    if len(rois) > 0:
        idxs = np.linspace(0, len(rois) - 1, num=min(10, len(rois)), dtype=int).tolist()
        sel_rois = [rois[i] for i in idxs]
        out_png = osp.join(qc_dir, 'rois.png')
        visualize_qc(video_path, idxs, sel_rois, motion_suppressed, out_png)

    # Save run metadata
    meta = {
        'video_path': video_path,
        'gt_path': gt_path,
        'total_frames': len(roi_frames),
        'fps': fps,
        'chunks': chunks,
        'config': config,
    }
    with open(osp.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    root = osp.join(osp.dirname(__file__), 'Datasets')
    out_root = osp.join(root, 'processed')
    ensure_dir(out_root)

    pairs = find_pairs_by_glob(root, video_glob='vid.avi', gt_glob='ground_truth.txt')
    rng = np.random.default_rng(1234)

    config = {
        'chunk_seconds': 10.0,
        'hop_seconds': 5.0,
        'do_augment': True,
    }

    for video_path, gt_path in tqdm(pairs, desc='Preprocessing videos'):
        subject_dir = osp.dirname(video_path)
        subject_name = osp.basename(subject_dir)
        out_dir = osp.join(out_root, subject_name)
        process_video_pair(video_path, gt_path, out_dir, rng, config)


if __name__ == '__main__':
    main()


