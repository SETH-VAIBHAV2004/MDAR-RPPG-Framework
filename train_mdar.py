import os
import os.path as osp
import csv
import math
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from scipy import signal
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from data.ubfc_dataset import list_subjects, split_subjects, UBFCChunks
from models.mdar import MDAR


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def mape(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    denom = torch.clamp(target.abs(), min=eps)
    return torch.mean(torch.abs((pred - target) / denom)) * 100.0


def pearson_correlation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative Pearson correlation as loss (to maximize correlation)"""
    pred_mean = torch.mean(pred, dim=-1, keepdim=True)
    target_mean = torch.mean(target, dim=-1, keepdim=True)
    
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    
    numerator = torch.sum(pred_centered * target_centered, dim=-1)
    pred_std = torch.sqrt(torch.sum(pred_centered ** 2, dim=-1) + 1e-8)
    target_std = torch.sqrt(torch.sum(target_centered ** 2, dim=-1) + 1e-8)
    
    correlation = numerator / (pred_std * target_std + 1e-8)
    return -torch.mean(correlation)  # Negative to maximize correlation


def frequency_domain_loss(pred: torch.Tensor, target: torch.Tensor, sample_rate: float = 30.0) -> torch.Tensor:
    """Frequency domain loss comparing power spectral densities"""
    B, T = pred.shape
    
    # Compute FFT
    pred_fft = torch.fft.rfft(pred, dim=-1)
    target_fft = torch.fft.rfft(target, dim=-1)
    
    # Power spectral density
    pred_psd = torch.abs(pred_fft) ** 2
    target_psd = torch.abs(target_fft) ** 2
    
    # Focus on physiological frequency range (0.7-4 Hz)
    freqs = torch.fft.rfftfreq(T, d=1.0/sample_rate, device=pred.device)
    mask = (freqs >= 0.7) & (freqs <= 4.0)
    
    # Weighted MSE loss in frequency domain
    if torch.any(mask):
        pred_psd_masked = pred_psd[:, mask]
        target_psd_masked = target_psd[:, mask]
        return torch.mean((pred_psd_masked - target_psd_masked) ** 2)
    else:
        return torch.mean((pred_psd - target_psd) ** 2)


def compute_hr_from_signal(signal: np.ndarray, fps: float = 30.0, fmin: float = 0.7, fmax: float = 4.0) -> float:
    """Estimate heart rate from signal using FFT peak finding"""
    if len(signal) < 64:
        return 60.0  # Default fallback
    
    # Detrend and window
    signal = signal - np.mean(signal)
    window = np.hanning(len(signal))
    signal_windowed = signal * window
    
    # FFT
    fft = np.fft.rfft(signal_windowed)
    freqs = np.fft.rfftfreq(len(signal), d=1.0/fps)
    psd = np.abs(fft) ** 2
    
    # Find peak in physiological range
    mask = (freqs >= fmin) & (freqs <= fmax)
    if np.any(mask):
        psd_masked = psd[mask]
        freqs_masked = freqs[mask]
        peak_idx = np.argmax(psd_masked)
        hr_freq = freqs_masked[peak_idx]
        return hr_freq * 60.0  # Convert to BPM
    else:
        return 60.0


def pos_algorithm(rgb_means: np.ndarray) -> np.ndarray:
    """Plane-Orthogonal-to-Skin (POS) algorithm"""
    if rgb_means.shape[0] < 10 or rgb_means.shape[1] != 3:
        return np.zeros(rgb_means.shape[0])
    
    # Temporal normalization
    rgb_norm = np.zeros_like(rgb_means)
    for i in range(3):
        channel = rgb_means[:, i]
        mean_val = np.mean(channel)
        if mean_val > 0:
            rgb_norm[:, i] = channel / mean_val
    
    # POS projection
    l = len(rgb_norm)
    pos_signal = np.zeros(l)
    
    # Projection matrix for POS
    P = np.array([[0, 1, -1], [-2, 1, 1]]) / np.sqrt(6)
    
    # Apply projection
    C = P @ rgb_norm.T  # Shape: (2, T)
    alpha = np.std(C[0]) / (np.std(C[1]) + 1e-8)
    pos_signal = C[0] - alpha * C[1]
    
    return pos_signal


def chrom_algorithm(rgb_means: np.ndarray) -> np.ndarray:
    """Chrominance-based (CHROM) algorithm"""
    if rgb_means.shape[0] < 10 or rgb_means.shape[1] != 3:
        return np.zeros(rgb_means.shape[0])
    
    # Temporal normalization
    rgb_norm = np.zeros_like(rgb_means)
    for i in range(3):
        channel = rgb_means[:, i]
        mean_val = np.mean(channel)
        if mean_val > 0:
            rgb_norm[:, i] = channel / mean_val
    
    # CHROM signal
    R, G, B = rgb_norm[:, 0], rgb_norm[:, 1], rgb_norm[:, 2]
    X = 3 * R - 2 * G
    Y = 1.5 * R + G - 1.5 * B
    
    alpha = np.std(X) / (np.std(Y) + 1e-8)
    chrom_signal = X - alpha * Y
    
    return chrom_signal


def temporal_smoothing(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Apply temporal smoothing using moving average"""
    if len(signal) < window_size:
        return signal
    
    kernel = np.ones(window_size) / window_size
    return np.convolve(signal, kernel, mode='same')


def bootstrap_confidence_interval(data: np.ndarray, n_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for mean"""
    bootstrap_means = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    return lower, upper


def collate_pad(batch):
    # variable-length chunks; pad to max T in batch
    lengths = [b['x'].shape[0] for b in batch]
    max_t = max(lengths)
    feat_dim = batch[0]['x'].shape[1]
    x_pad = torch.zeros((len(batch), max_t, feat_dim), dtype=torch.float32)
    y_pad = torch.zeros((len(batch), max_t), dtype=torch.float32)
    for i, b in enumerate(batch):
        t = b['x'].shape[0]
        x_pad[i, :t] = b['x']
        y_pad[i, :t] = b['y']
    return x_pad, y_pad, lengths, batch


def save_array(path: str, arr: np.ndarray) -> None:
    os.makedirs(osp.dirname(path), exist_ok=True)
    if path.lower().endswith('.npy'):
        np.save(path, arr)
    elif path.lower().endswith('.csv'):
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['t', 'value'])
            for i, v in enumerate(arr):
                writer.writerow([i, float(v)])
    else:
        np.save(path + '.npy', arr)


def compute_ensemble_hr(model_hr: float, pos_hr: float, chrom_hr: float, confidence: float) -> float:
    """Ensemble heart rate estimation using confidence weighting"""
    # Weighted average with model confidence
    if confidence > 0.5:
        return model_hr
    else:
        # Fallback to traditional methods with equal weighting
        return (pos_hr + chrom_hr) / 2.0


class CombinedLoss(nn.Module):
    """Combined loss function with MAE, correlation, and frequency domain components"""
    def __init__(self, mae_weight: float = 1.0, corr_weight: float = 0.5, freq_weight: float = 0.3, 
                 hr_weight: float = 0.1, sample_rate: float = 30.0):
        super().__init__()
        self.mae_weight = mae_weight
        self.corr_weight = corr_weight
        self.freq_weight = freq_weight
        self.hr_weight = hr_weight
        self.sample_rate = sample_rate
        self.mse_loss = nn.MSELoss()
        
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], mask: torch.Tensor):
        waveform_pred = outputs['waveform'] * mask
        waveform_target = targets['waveform'] * mask
        
        # Waveform losses
        mae_loss = mae(waveform_pred, waveform_target)
        corr_loss = pearson_correlation_loss(waveform_pred, waveform_target)
        freq_loss = frequency_domain_loss(waveform_pred, waveform_target, self.sample_rate)
        
        waveform_loss = (self.mae_weight * mae_loss + 
                        self.corr_weight * corr_loss +
                        self.freq_weight * freq_loss)
        
        total_loss = waveform_loss
        
        # Heart rate loss (if available)
        if 'heart_rate' in outputs and 'heart_rate' in targets:
            hr_loss = self.mse_loss(outputs['heart_rate'], targets['heart_rate'])
            total_loss += self.hr_weight * hr_loss
        
        return total_loss, {
            'mae_loss': mae_loss.item(),
            'corr_loss': corr_loss.item(),
            'freq_loss': freq_loss.item(),
            'hr_loss': hr_loss.item() if 'heart_rate' in outputs else 0.0
        }


def enhanced_collate_pad(batch):
    """Enhanced collate function that also extracts RGB means for ensemble methods"""
    lengths = [b['x'].shape[0] for b in batch]
    max_t = max(lengths)
    feat_dim = batch[0]['x'].shape[1]
    x_pad = torch.zeros((len(batch), max_t, feat_dim), dtype=torch.float32)
    y_pad = torch.zeros((len(batch), max_t), dtype=torch.float32)
    
    rgb_means_batch = []
    hr_targets_batch = []
    
    for i, b in enumerate(batch):
        t = b['x'].shape[0]
        x_pad[i, :t] = b['x']
        y_pad[i, :t] = b['y']
        
        # Load RGB means for ensemble methods
        try:
            data = np.load(b['path'], allow_pickle=True)
            if 'rgb_means' in data.files:
                rgb_means = data['rgb_means']
                rgb_means_batch.append(rgb_means)
                
                # Compute HR target from ground truth signal
                if 'gt' in data.files and data['gt'] is not None:
                    gt_signal = data['gt']
                    fps = float(data['fps']) if 'fps' in data.files else 30.0
                    hr_target = compute_hr_from_signal(gt_signal, fps)
                    hr_targets_batch.append(hr_target)
                else:
                    hr_targets_batch.append(60.0)  # Default
            else:
                # Fallback - use first 3 channels as RGB
                rgb_means_batch.append(b['x'][:, :3].numpy())
                hr_targets_batch.append(60.0)
        except:
            # Fallback
            rgb_means_batch.append(b['x'][:, :3].numpy())
            hr_targets_batch.append(60.0)
    
    return x_pad, y_pad, lengths, batch, rgb_means_batch, torch.tensor(hr_targets_batch, dtype=torch.float32)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset setup
    processed_root = osp.join(osp.dirname(__file__), 'Datasets')
    subjects = list_subjects(processed_root)
    train_subs, val_subs = split_subjects(subjects, train_ratio=0.85, seed=42)  # Increased training data
    print(f"Training subjects: {len(train_subs)}, Validation subjects: {len(val_subs)}")

    train_ds = UBFCChunks(processed_root, train_subs)
    val_ds = UBFCChunks(processed_root, val_subs)

    # Improved data loaders with more workers for efficiency
    train_loader = DataLoader(train_ds, batch_size=12, shuffle=True, num_workers=2, collate_fn=enhanced_collate_pad, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=12, shuffle=False, num_workers=2, collate_fn=enhanced_collate_pad, pin_memory=True)

    # Enhanced model with better capacity and regularization
    model = MDAR(in_features=4, hidden_channels=96, dropout=0.35, 
                 sample_rate=30.0, use_bandpass=True, multitask=True).to(device)
    
    # Improved loss function with better weighting
    criterion = CombinedLoss(mae_weight=1.0, corr_weight=0.7, freq_weight=0.4, hr_weight=0.25)
    
    # Better optimizer settings for stability
    optimizer = optim.AdamW(model.parameters(), lr=0.0008, weight_decay=2e-4, betas=(0.9, 0.95))
    
    # Cosine annealing with warm restarts for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    out_root = osp.join(osp.dirname(__file__), 'outputs', 'mdar_enhanced')
    os.makedirs(out_root, exist_ok=True)

    # Enhanced logging
    log_path = osp.join(out_root, 'training_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_mae', 'train_corr', 'train_freq', 'train_hr',
                        'val_loss', 'val_mae', 'val_mape', 'val_pearson_r', 'val_hr_mae', 
                        'ensemble_hr_mae', 'lr'])

    best_val = float('inf')
    
    # Training loop
    for epoch in range(1, 31):
        model.train()
        train_losses = []
        train_loss_components = {'mae_loss': [], 'corr_loss': [], 'freq_loss': [], 'hr_loss': []}
        
        for batch_data in train_loader:
            x, y, lengths, batch, rgb_means_batch, hr_targets = batch_data
            x = x.to(device)
            y = y.to(device)
            hr_targets = hr_targets.to(device)
            
            # Forward pass
            outputs = model(x)
            
            # Create mask for padded positions
            mask = torch.zeros_like(y)
            for i, t in enumerate(lengths):
                mask[i, :t] = 1.0
            
            # Prepare targets
            targets = {'waveform': y, 'heart_rate': hr_targets}
            model_outputs = {'waveform': outputs['waveform'], 'heart_rate': outputs['heart_rate']}
            
            # Compute loss
            loss, loss_components = criterion(model_outputs, targets, mask)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Log training metrics
            train_losses.append(loss.item())
            for k, v in loss_components.items():
                train_loss_components[k].append(v)

        # Validation
        model.eval()
        val_losses = []
        val_maes = []
        val_mapes = []
        val_correlations = []
        val_hr_maes = []
        ensemble_hr_maes = []
        
        with torch.no_grad():
            for batch_data in val_loader:
                x, y, lengths, batch, rgb_means_batch, hr_targets = batch_data
                x = x.to(device)
                y = y.to(device)
                hr_targets = hr_targets.to(device)
                
                outputs = model(x)
                
                # Create mask
                mask = torch.zeros_like(y)
                for i, t in enumerate(lengths):
                    mask[i, :t] = 1.0
                
                # Compute validation loss
                targets = {'waveform': y, 'heart_rate': hr_targets}
                model_outputs = {'waveform': outputs['waveform'], 'heart_rate': outputs['heart_rate']}
                loss, _ = criterion(model_outputs, targets, mask)
                val_losses.append(loss.item())
                
                # Compute detailed metrics
                waveform_pred = outputs['waveform'] * mask
                waveform_target = y * mask
                
                val_maes.append(mae(waveform_pred, waveform_target).item())
                val_mapes.append(mape(waveform_pred, waveform_target).item())
                val_hr_maes.append(mae(outputs['heart_rate'], hr_targets).item())
                
                # Compute correlations and ensemble HR
                for i in range(len(lengths)):
                    t = lengths[i]
                    pred_seq = outputs['waveform'][i, :t].cpu().numpy()
                    target_seq = y[i, :t].cpu().numpy()
                    
                    # Temporal smoothing
                    pred_smooth = temporal_smoothing(pred_seq)
                    
                    # Pearson correlation
                    if len(pred_smooth) > 5 and np.std(pred_smooth) > 1e-6 and np.std(target_seq) > 1e-6:
                        corr, _ = pearsonr(pred_smooth, target_seq)
                        val_correlations.append(abs(corr) if not np.isnan(corr) else 0.0)
                    
                    # Ensemble HR estimation
                    if i < len(rgb_means_batch):
                        rgb_means = rgb_means_batch[i]
                        model_hr = outputs['heart_rate'][i].cpu().item()
                        confidence = outputs['confidence'][i].cpu().item()
                        
                        # Compute POS and CHROM HR estimates
                        pos_signal = pos_algorithm(rgb_means)
                        chrom_signal = chrom_algorithm(rgb_means)
                        pos_hr = compute_hr_from_signal(pos_signal, fps=30.0)
                        chrom_hr = compute_hr_from_signal(chrom_signal, fps=30.0)
                        
                        # Ensemble HR
                        ensemble_hr = compute_ensemble_hr(model_hr, pos_hr, chrom_hr, confidence)
                        target_hr = hr_targets[i].cpu().item()
                        ensemble_hr_maes.append(abs(ensemble_hr - target_hr))

        # Compute epoch metrics
        train_loss_avg = np.mean(train_losses)
        train_mae_avg = np.mean(train_loss_components['mae_loss'])
        train_corr_avg = np.mean(train_loss_components['corr_loss']) 
        train_freq_avg = np.mean(train_loss_components['freq_loss'])
        train_hr_avg = np.mean(train_loss_components['hr_loss'])
        
        val_loss_avg = np.mean(val_losses)
        val_mae_avg = np.mean(val_maes)
        val_mape_avg = np.mean(val_mapes)
        val_pearson_avg = np.mean(val_correlations) if val_correlations else 0.0
        val_hr_mae_avg = np.mean(val_hr_maes)
        ensemble_hr_mae_avg = np.mean(ensemble_hr_maes) if ensemble_hr_maes else 0.0
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to CSV
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss_avg, train_mae_avg, train_corr_avg, train_freq_avg, train_hr_avg,
                           val_loss_avg, val_mae_avg, val_mape_avg, val_pearson_avg, val_hr_mae_avg,
                           ensemble_hr_mae_avg, current_lr])
        
        # Save checkpoint
        ckpt_path = osp.join(out_root, f'mdar_epoch_{epoch:03d}.pth')
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'val_loss': val_loss_avg,
            'val_mae': val_mae_avg,
        }, ckpt_path)
        
        # Track best model
        if val_loss_avg < best_val:
            best_val = val_loss_avg
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_loss': val_loss_avg,
                'val_mae': val_mae_avg,
            }, osp.join(out_root, 'mdar_best.pth'))
        
        # Save final predictions with temporal smoothing
        if epoch == 30:
            pred_dir = osp.join(out_root, 'predictions')
            os.makedirs(pred_dir, exist_ok=True)
            
            with torch.no_grad():
                for batch_data in val_loader:
                    x, y, lengths, batch, rgb_means_batch, hr_targets = batch_data
                    x = x.to(device)
                    outputs = model(x)
                    
                    for i, sample in enumerate(batch):
                        t = lengths[i]
                        pred_raw = outputs['waveform'][i, :t].cpu().numpy()
                        pred_smooth = temporal_smoothing(pred_raw, window_size=5)
                        
                        # Save predictions
                        base = os.path.splitext(os.path.basename(sample['path']))[0]
                        save_array(osp.join(pred_dir, base + '_raw.npy'), pred_raw)
                        save_array(osp.join(pred_dir, base + '_smooth.npy'), pred_smooth)
                        save_array(osp.join(pred_dir, base + '_smooth.csv'), pred_smooth)
                        
                        # Save HR predictions
                        hr_pred = outputs['heart_rate'][i].cpu().item()
                        confidence = outputs['confidence'][i].cpu().item()
                        
                        # Save ensemble HR
                        if i < len(rgb_means_batch):
                            rgb_means = rgb_means_batch[i]
                            pos_signal = pos_algorithm(rgb_means)
                            chrom_signal = chrom_algorithm(rgb_means)
                            pos_hr = compute_hr_from_signal(pos_signal)
                            chrom_hr = compute_hr_from_signal(chrom_signal)
                            ensemble_hr = compute_ensemble_hr(hr_pred, pos_hr, chrom_hr, confidence)
                            
                            hr_data = {
                                'model_hr': hr_pred,
                                'pos_hr': pos_hr,
                                'chrom_hr': chrom_hr,
                                'ensemble_hr': ensemble_hr,
                                'confidence': confidence
                            }
                            
                            with open(osp.join(pred_dir, base + '_hr.csv'), 'w', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=hr_data.keys())
                                writer.writeheader()
                                writer.writerow(hr_data)
                    
                    break  # Only process first batch for efficiency
        
        # Compute bootstrap confidence intervals for validation MAE
        if len(val_maes) > 10:
            val_mae_ci_lower, val_mae_ci_upper = bootstrap_confidence_interval(np.array(val_maes))
            ci_str = f"MAE CI: [{val_mae_ci_lower:.4f}, {val_mae_ci_upper:.4f}]"
        else:
            ci_str = "MAE CI: N/A"
        
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss_avg:.4f} MAE: {train_mae_avg:.4f} | "
              f"Val Loss: {val_loss_avg:.4f} MAE: {val_mae_avg:.4f} MAPE: {val_mape_avg:.2f}% | "
              f"Pearson r: {val_pearson_avg:.3f} HR MAE: {val_hr_mae_avg:.2f} Ensemble HR MAE: {ensemble_hr_mae_avg:.2f} | "
              f"{ci_str} | LR: {current_lr:.6f}")
    
    print("\nTraining completed! Best model saved with validation loss: {:.4f}".format(best_val))


if __name__ == '__main__':
    train()


