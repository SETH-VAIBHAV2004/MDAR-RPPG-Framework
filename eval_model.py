import os
import os.path as osp
import argparse
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

from data.ubfc_dataset import list_subjects, split_subjects, UBFCChunks
from models.mdar import MDAR
from train_mdar import enhanced_collate_pad, compute_hr_from_signal, mae


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics."""
    mae_val = np.mean(np.abs(predictions - targets))
    mse_val = np.mean((predictions - targets) ** 2)
    rmse_val = np.sqrt(mse_val)
    
    # Pearson correlation
    correlation, _ = pearsonr(predictions.flatten(), targets.flatten())
    if np.isnan(correlation):
        correlation = 0.0
    
    return {
        'mae': float(mae_val),
        'mse': float(mse_val), 
        'rmse': float(rmse_val),
        'correlation': float(correlation)
    }


def evaluate_model(model_path: str, data_root: str, device: torch.device) -> Dict[str, float]:
    """Evaluate trained model on test set."""
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model_state = checkpoint['model_state']
    
    # Create model (assuming standard config - you may need to adjust)
    model = MDAR(in_features=4, hidden_channels=96, dropout=0.35, 
                 sample_rate=30.0, use_bandpass=True, multitask=True).to(device)
    model.load_state_dict(model_state)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Model val_loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
    
    # Load test data
    subjects = list_subjects(data_root)
    _, val_subjects = split_subjects(subjects, train_ratio=0.85, seed=42)
    
    test_ds = UBFCChunks(data_root, val_subjects)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0, 
                            collate_fn=enhanced_collate_pad)
    
    print(f"Evaluating on {len(test_ds)} test samples from {len(val_subjects)} subjects")
    
    all_predictions = []
    all_targets = []
    all_hr_predictions = []
    all_hr_targets = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            x, y, lengths, batch, rgb_means_batch, hr_targets = batch_data
            x = x.to(device)
            y = y.to(device)
            hr_targets = hr_targets.to(device)
            
            outputs = model(x)
            
            # Extract valid (non-padded) predictions
            for i in range(len(lengths)):
                t = lengths[i]
                pred_seq = outputs['waveform'][i, :t].cpu().numpy()
                target_seq = y[i, :t].cpu().numpy()
                
                all_predictions.extend(pred_seq)
                all_targets.extend(target_seq)
                
                # Heart rate predictions
                if 'heart_rate' in outputs:
                    all_hr_predictions.append(outputs['heart_rate'][i].cpu().item())
                    all_hr_targets.append(hr_targets[i].cpu().item())
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    # Compute waveform metrics
    waveform_metrics = compute_metrics(predictions, targets)
    
    # Compute HR metrics if available
    hr_metrics = {}
    if all_hr_predictions and all_hr_targets:
        hr_pred_arr = np.array(all_hr_predictions)
        hr_target_arr = np.array(all_hr_targets)
        hr_metrics = compute_metrics(hr_pred_arr, hr_target_arr)
        hr_metrics = {f'hr_{k}': v for k, v in hr_metrics.items()}
    
    # Combine metrics
    all_metrics = {**waveform_metrics, **hr_metrics}
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate MDAR model')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_root', type=str, default='Datasets', 
                       help='Path to dataset root')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Evaluate model
    try:
        metrics = evaluate_model(args.model_path, args.data_root, device)
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        print("\nWaveform Metrics:")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  Correlation: {metrics['correlation']:.4f}")
        
        if any(k.startswith('hr_') for k in metrics.keys()):
            print("\nHeart Rate Metrics:")
            print(f"  HR MAE: {metrics['hr_mae']:.2f} BPM")
            print(f"  HR RMSE: {metrics['hr_rmse']:.2f} BPM")
            print(f"  HR Correlation: {metrics['hr_correlation']:.4f}")
        
        print("\n" + "="*50)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
