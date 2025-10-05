import os
import os.path as osp
import csv
from typing import List, Tuple, Dict

import numpy as np

from data.ubfc_dataset import list_subjects, split_subjects


def detrend_and_normalize(rgb: np.ndarray) -> np.ndarray:
    # rgb: (T, 3) in BGR order from preprocessing; convert to RGB order
    b = rgb[:, 0]
    g = rgb[:, 1]
    r = rgb[:, 2]
    X = np.stack([r, g, b], axis=1).astype(np.float64)
    X = X - np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True) + 1e-8
    X = X / std
    return X


def pos_method(rgb: np.ndarray) -> np.ndarray:
    # Implement POS per de Haan & Jeanne (2013)
    X = detrend_and_normalize(rgb)  # (T, 3) RGB
    H = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])  # 2x3
    S = X @ H.T  # (T, 2)
    s1 = S[:, 0]
    s2 = S[:, 1]
    alpha = (np.std(s1) + 1e-8) / (np.std(s2) + 1e-8)
    s = s1 + alpha * s2
    s = s - np.mean(s)
    return s.astype(np.float64)


def chrom_method(rgb: np.ndarray) -> np.ndarray:
    # Implement CHROM per de Haan & Jeanne (2013)
    X = detrend_and_normalize(rgb)  # RGB
    r = X[:, 0]
    g = X[:, 1]
    b = X[:, 2]
    x = 3*r - 2*g
    y = 1.5*r + g - 1.5*b
    alpha = (np.std(x) + 1e-8) / (np.std(y) + 1e-8)
    s = x - alpha * y
    s = s - np.mean(s)
    return s.astype(np.float64)


def estimate_hr_fft(signal: np.ndarray, fps: float, fmin: float = 0.7, fmax: float = 4.0) -> Tuple[float, float]:
    # Returns (hr_bpm, peak_power)
    if len(signal) < 4:
        return 0.0, 0.0
    win = np.hanning(len(signal))
    sigw = signal * win
    spec = np.fft.rfft(sigw)
    freqs = np.fft.rfftfreq(len(sigw), d=1.0 / float(fps))
    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band):
        return 0.0, 0.0
    mag = np.abs(spec) ** 2
    idx = np.argmax(mag[band])
    band_freqs = freqs[band]
    peak_freq = float(band_freqs[idx])
    peak_power = float(np.max(mag[band]))
    hr_bpm = peak_freq * 60.0
    return hr_bpm, peak_power


def bland_altman_data(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    mean = 0.5 * (a + b)
    diff = a - b
    md = float(np.mean(diff))
    sd = float(np.std(diff))
    return mean, diff, md, sd


def load_chunks(processed_root: str, subjects: List[str]) -> List[Dict]:
    samples = []
    for s in subjects:
        feat_dir = osp.join(processed_root, s, 'features')
        if not osp.isdir(feat_dir):
            continue
        for npz in sorted(os.listdir(feat_dir)):
            if not npz.endswith('.npz'):
                continue
            samples.append({'subject': s, 'path': osp.join(feat_dir, npz)})
    return samples


def run_eval() -> None:
    root = osp.join(osp.dirname(__file__), 'Datasets', 'processed')
    subjects = list_subjects(root)
    train_subs, val_subs = split_subjects(subjects, train_ratio=0.8, seed=42)

    val_samples = load_chunks(root, val_subs)
    out_root = osp.join(osp.dirname(__file__), 'outputs', 'pos_chrom')
    os.makedirs(out_root, exist_ok=True)
    preds_dir = osp.join(out_root, 'preds')
    os.makedirs(preds_dir, exist_ok=True)

    all_gt = []
    all_pos = []
    all_chrom = []
    all_fused = []

    metrics_rows = []

    for sample in val_samples:
        data = np.load(sample['path'], allow_pickle=True)
        feats = data['features']  # (T, 4) B,G,R, diff
        fps = float(data['fps']) if 'fps' in data else 30.0
        gt = data['gt'] if 'gt' in data.files and data['gt'] is not None else None
        if gt is None or len(gt) != feats.shape[0]:
            gt = None

        rgb_bgr = feats[:, :3]  # (T, 3) B,G,R

        s_pos = pos_method(rgb_bgr)
        s_chrom = chrom_method(rgb_bgr)

        hr_pos, p_pos = estimate_hr_fft(s_pos, fps)
        hr_chrom, p_chrom = estimate_hr_fft(s_chrom, fps)

        # Weighted fusion by peak power (SNR proxy)
        w_pos = p_pos + 1e-6
        w_chrom = p_chrom + 1e-6
        hr_fused = float((w_pos * hr_pos + w_chrom * hr_chrom) / (w_pos + w_chrom))

        gt_hr = float(np.nan)
        if gt is not None:
            # If gt is waveform-like HR per frame, compute average HR over the window
            gt_hr = float(np.nanmean(gt))

        # accumulate for aggregate metrics when gt is available
        if not np.isnan(gt_hr):
            all_gt.append(gt_hr)
            all_pos.append(hr_pos)
            all_chrom.append(hr_chrom)
            all_fused.append(hr_fused)

        # save per-chunk predictions
        base = osp.splitext(osp.basename(sample['path']))[0]
        with open(osp.join(preds_dir, base + '.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['subject', 'chunk', 'gt_hr', 'hr_pos', 'hr_chrom', 'hr_fused'])
            w.writerow([sample['subject'], base, gt_hr, hr_pos, hr_chrom, hr_fused])

    # Compute aggregate metrics
    def safe_metrics(pred: List[float], gt: List[float]) -> Dict[str, float]:
        if len(pred) == 0:
            return {'MAE': np.nan, 'RMSE': np.nan, 'Pearson': np.nan}
        pred_a = np.asarray(pred, dtype=np.float64)
        gt_a = np.asarray(gt, dtype=np.float64)
        mae = float(np.mean(np.abs(pred_a - gt_a)))
        rmse = float(np.sqrt(np.mean((pred_a - gt_a) ** 2)))
        if np.std(pred_a) < 1e-8 or np.std(gt_a) < 1e-8:
            corr = np.nan
        else:
            corr = float(np.corrcoef(pred_a, gt_a)[0, 1])
        return {'MAE': mae, 'RMSE': rmse, 'Pearson': corr}

    m_pos = safe_metrics(all_pos, all_gt)
    m_chrom = safe_metrics(all_chrom, all_gt)
    m_fused = safe_metrics(all_fused, all_gt)

    with open(osp.join(out_root, 'metrics.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'MAE', 'RMSE', 'Pearson', 'N'])
        w.writerow(['POS', m_pos['MAE'], m_pos['RMSE'], m_pos['Pearson'], len(all_pos)])
        w.writerow(['CHROM', m_chrom['MAE'], m_chrom['RMSE'], m_chrom['Pearson'], len(all_chrom)])
        w.writerow(['Fused', m_fused['MAE'], m_fused['RMSE'], m_fused['Pearson'], len(all_fused)])

    # Optional Bland-Altman for fused
    if len(all_gt) > 1 and len(all_fused) > 1:
        import matplotlib.pyplot as plt
        mean, diff, md, sd = bland_altman_data(np.asarray(all_fused), np.asarray(all_gt))
        plt.figure(figsize=(6, 4))
        plt.scatter(mean, diff, s=12, alpha=0.7)
        plt.axhline(md, color='r', linestyle='--', label='mean diff')
        plt.axhline(md + 1.96 * sd, color='g', linestyle=':')
        plt.axhline(md - 1.96 * sd, color='g', linestyle=':')
        plt.xlabel('Mean HR (bpm)')
        plt.ylabel('Difference (Fused - GT) bpm')
        plt.title('Bland-Altman: Fused vs GT')
        plt.legend()
        plt.tight_layout()
        plt.savefig(osp.join(out_root, 'bland_altman_fused.png'), dpi=150)
        plt.close()

    print('Saved metrics to', osp.join(out_root, 'metrics.csv'))


if __name__ == '__main__':
    run_eval()


