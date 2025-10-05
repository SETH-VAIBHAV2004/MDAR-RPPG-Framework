#!/usr/bin/env python3
"""
Heart Rate Prediction vs Ground Truth Visualization
Generates comparison graphs for rPPG heart rate predictions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import os
import seaborn as sns

# Set the style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def generate_sample_data(duration_minutes=5, sampling_rate=30):
    """
    Generate sample heart rate data for demonstration
    
    Args:
        duration_minutes (int): Duration of the signal in minutes
        sampling_rate (int): Samples per second (Hz)
    
    Returns:
        tuple: (time_points, ground_truth, predictions)
    """
    # Time points
    total_samples = duration_minutes * 60 * sampling_rate
    time_points = np.linspace(0, duration_minutes * 60, total_samples)
    
    # Ground truth heart rate (simulate realistic variation)
    base_hr = 72  # Base heart rate
    # Add natural variation and some periodic components
    variation = 8 * np.sin(0.02 * time_points) + 4 * np.cos(0.05 * time_points)
    noise = 2 * np.random.normal(0, 1, len(time_points))
    ground_truth = base_hr + variation + noise
    
    # Predicted heart rate (simulate prediction errors)
    prediction_noise = 3 * np.random.normal(0, 1, len(time_points))
    phase_shift = 0.1  # Small phase shift to simulate prediction lag
    predictions = base_hr + 0.9 * variation + prediction_noise
    
    # Add some systematic bias in certain regions
    bias_region = (time_points > 120) & (time_points < 180)
    predictions[bias_region] += 5
    
    # Ensure realistic heart rate range
    ground_truth = np.clip(ground_truth, 50, 120)
    predictions = np.clip(predictions, 50, 120)
    
    return time_points, ground_truth, predictions

def calculate_metrics(ground_truth, predictions):
    """Calculate evaluation metrics"""
    mae = np.mean(np.abs(ground_truth - predictions))
    rmse = np.sqrt(np.mean((ground_truth - predictions) ** 2))
    correlation = np.corrcoef(ground_truth, predictions)[0, 1]
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'Correlation': correlation
    }

def create_comparison_plot(time_points, ground_truth, predictions, save_path):
    """
    Create a comprehensive comparison plot
    
    Args:
        time_points (array): Time points in seconds
        ground_truth (array): Ground truth heart rate values
        predictions (array): Predicted heart rate values
        save_path (str): Path to save the plot
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], width_ratios=[3, 1])
    
    # Main time series plot
    ax1 = fig.add_subplot(gs[0, :])
    
    # Convert time to minutes for better readability
    time_minutes = time_points / 60
    
    # Plot the signals
    line1 = ax1.plot(time_minutes, ground_truth, 'b-', linewidth=2, alpha=0.8, label='Ground Truth')
    line2 = ax1.plot(time_minutes, predictions, 'r--', linewidth=2, alpha=0.8, label='Predicted')
    
    # Fill area between curves to show error
    ax1.fill_between(time_minutes, ground_truth, predictions, alpha=0.2, color='gray', label='Prediction Error')
    
    ax1.set_xlabel('Time (minutes)', fontsize=12)
    ax1.set_ylabel('Heart Rate (BPM)', fontsize=12)
    ax1.set_title('Predicted vs Ground Truth Heart Rate Over Time', fontsize=16, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(45, 125)
    
    # Error plot
    ax2 = fig.add_subplot(gs[1, :])
    error = predictions - ground_truth
    ax2.plot(time_minutes, error, 'g-', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.fill_between(time_minutes, error, 0, alpha=0.3, color='green')
    ax2.set_xlabel('Time (minutes)', fontsize=12)
    ax2.set_ylabel('Error (BPM)', fontsize=12)
    ax2.set_title('Prediction Error Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Scatter plot
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.scatter(ground_truth, predictions, alpha=0.6, s=10, c='purple')
    
    # Perfect prediction line
    min_hr, max_hr = min(ground_truth.min(), predictions.min()), max(ground_truth.max(), predictions.max())
    ax3.plot([min_hr, max_hr], [min_hr, max_hr], 'k--', alpha=0.7, label='Perfect Prediction')
    
    ax3.set_xlabel('Ground Truth (BPM)', fontsize=12)
    ax3.set_ylabel('Predicted (BPM)', fontsize=12)
    ax3.set_title('Scatter Plot: Predicted vs Ground Truth', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Metrics box
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')
    
    metrics = calculate_metrics(ground_truth, predictions)
    
    # Create metrics text
    metrics_text = f"""
    Performance Metrics:
    
    MAE: {metrics['MAE']:.2f} BPM
    RMSE: {metrics['RMSE']:.2f} BPM
    Correlation: {metrics['Correlation']:.3f}
    
    Data Points: {len(ground_truth):,}
    Duration: {time_points[-1]/60:.1f} min
    """
    
    ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Add method information
    method_text = """
    rPPG Methods:
    • MDAR Neural Network
    • POS Algorithm
    • CHROM Method
    • Ensemble Fusion
    """
    
    ax4.text(0.1, 0.4, method_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✅ Graph saved to: {save_path}")
    
    return fig

def create_statistical_analysis_plot(ground_truth, predictions, save_path):
    """
    Create statistical analysis plots
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Error distribution
    error = predictions - ground_truth
    ax1.hist(error, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(error), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(error):.2f}')
    ax1.axvline(np.median(error), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(error):.2f}')
    ax1.set_xlabel('Prediction Error (BPM)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Error Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bland-Altman plot
    mean_values = (ground_truth + predictions) / 2
    differences = predictions - ground_truth
    
    ax2.scatter(mean_values, differences, alpha=0.6, s=15)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    
    ax2.axhline(mean_diff, color='red', linestyle='-', label=f'Mean Diff: {mean_diff:.2f}')
    ax2.axhline(mean_diff + 1.96*std_diff, color='red', linestyle='--', alpha=0.7, label=f'+1.96σ: {mean_diff + 1.96*std_diff:.2f}')
    ax2.axhline(mean_diff - 1.96*std_diff, color='red', linestyle='--', alpha=0.7, label=f'-1.96σ: {mean_diff - 1.96*std_diff:.2f}')
    
    ax2.set_xlabel('Mean of Ground Truth and Predicted (BPM)')
    ax2.set_ylabel('Difference (Predicted - Ground Truth)')
    ax2.set_title('Bland-Altman Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Box plot comparison
    data_to_plot = [ground_truth, predictions]
    ax3.boxplot(data_to_plot, labels=['Ground Truth', 'Predicted'], patch_artist=True)
    ax3.set_ylabel('Heart Rate (BPM)')
    ax3.set_title('Distribution Comparison')
    ax3.grid(True, alpha=0.3)
    
    # Regression plot
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(ground_truth, predictions)
    line = slope * ground_truth + intercept
    
    ax4.scatter(ground_truth, predictions, alpha=0.6, s=15, label='Data Points')
    ax4.plot(ground_truth, line, 'r', label=f'Regression Line (R²={r_value**2:.3f})')
    ax4.plot([ground_truth.min(), ground_truth.max()], [ground_truth.min(), ground_truth.max()], 'k--', alpha=0.7, label='Perfect Prediction')
    ax4.set_xlabel('Ground Truth (BPM)')
    ax4.set_ylabel('Predicted (BPM)')
    ax4.set_title('Regression Analysis')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✅ Statistical analysis saved to: {save_path}")
    
    return fig

def main():
    """Main function to generate all plots"""
    print("🫀 Generating Heart Rate Prediction Analysis Graphs...")
    
    # Create output directory if it doesn't exist
    output_dir = "graphs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created directory: {output_dir}")
    
    # Generate sample data
    print("📊 Generating sample data...")
    time_points, ground_truth, predictions = generate_sample_data(duration_minutes=5, sampling_rate=30)
    
    # Create main comparison plot
    print("🎨 Creating main comparison plot...")
    main_plot_path = os.path.join(output_dir, "hr_prediction_comparison.png")
    fig1 = create_comparison_plot(time_points, ground_truth, predictions, main_plot_path)
    
    # Create statistical analysis plot
    print("📈 Creating statistical analysis plot...")
    stats_plot_path = os.path.join(output_dir, "hr_statistical_analysis.png")
    fig2 = create_statistical_analysis_plot(ground_truth, predictions, stats_plot_path)
    
    # Calculate and display final metrics
    metrics = calculate_metrics(ground_truth, predictions)
    print("\n📋 Final Performance Metrics:")
    print(f"   • Mean Absolute Error (MAE): {metrics['MAE']:.2f} BPM")
    print(f"   • Root Mean Square Error (RMSE): {metrics['RMSE']:.2f} BPM")
    print(f"   • Correlation Coefficient: {metrics['Correlation']:.3f}")
    print(f"   • Data Points Analyzed: {len(ground_truth):,}")
    print(f"   • Signal Duration: {time_points[-1]/60:.1f} minutes")
    
    print(f"\n✨ All graphs successfully generated in '{output_dir}' directory!")
    print("📁 Generated files:")
    print(f"   • {main_plot_path}")
    print(f"   • {stats_plot_path}")

if __name__ == "__main__":
    main()
