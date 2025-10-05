#!/usr/bin/env python3
"""
Simple Heart Rate Graph Generator
Easy-to-customize script for generating heart rate comparison plots
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def create_simple_hr_graph(ground_truth_data, predicted_data, time_data=None, save_name="hr_comparison.png"):
    """
    Create a simple heart rate comparison graph
    
    Args:
        ground_truth_data (array): Ground truth heart rate values
        predicted_data (array): Predicted heart rate values  
        time_data (array, optional): Time points. If None, uses indices
        save_name (str): Name for the saved file
    """
    
    # Use indices as time if no time data provided
    if time_data is None:
        time_data = np.arange(len(ground_truth_data))
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Main comparison plot
    ax1.plot(time_data, ground_truth_data, 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax1.plot(time_data, predicted_data, 'r--', linewidth=2, label='Predicted', alpha=0.8)
    ax1.fill_between(time_data, ground_truth_data, predicted_data, alpha=0.2, color='gray', label='Error Region')
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Heart Rate (BPM)')
    ax1.set_title('Heart Rate: Predicted vs Ground Truth', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Error plot
    error = predicted_data - ground_truth_data
    ax2.plot(time_data, error, 'g-', linewidth=1.5, alpha=0.7, label='Prediction Error')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax2.fill_between(time_data, error, 0, alpha=0.3, color='green')
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Error (BPM)')
    ax2.set_title('Prediction Error Over Time', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Calculate and display metrics
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    correlation = np.corrcoef(ground_truth_data, predicted_data)[0, 1]
    
    # Add metrics text to the plot
    metrics_text = f'MAE: {mae:.2f} BPM | RMSE: {rmse:.2f} BPM | Correlation: {correlation:.3f}'
    fig.suptitle(f'rPPG Heart Rate Analysis - {metrics_text}', fontsize=10, y=0.02)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    # Save the plot
    if not os.path.exists('graphs'):
        os.makedirs('graphs')
    
    save_path = os.path.join('graphs', save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"✅ Simple heart rate graph saved to: {save_path}")
    print(f"📊 Metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, Correlation={correlation:.3f}")
    
    return fig

# Example usage with sample data
if __name__ == "__main__":
    # Generate sample data (you can replace this with your real data)
    print("🫀 Generating simple heart rate comparison graph...")
    
    # Sample data - replace these with your actual data arrays
    duration = 300  # 5 minutes in seconds
    time_points = np.linspace(0, duration, 300)  # 1 sample per second
    
    # Simulated ground truth heart rate
    base_hr = 75
    ground_truth = base_hr + 10 * np.sin(0.1 * time_points) + 2 * np.random.normal(0, 1, len(time_points))
    ground_truth = np.clip(ground_truth, 60, 100)
    
    # Simulated predicted heart rate with some error
    predicted = ground_truth + 3 * np.random.normal(0, 1, len(time_points)) + 2
    predicted = np.clip(predicted, 60, 100)
    
    # Create the graph
    fig = create_simple_hr_graph(ground_truth, predicted, time_points, "simple_hr_comparison.png")
    
    print("\n💡 To use with your own data:")
    print("   ground_truth = [your_ground_truth_data]")
    print("   predicted = [your_predicted_data]") 
    print("   create_simple_hr_graph(ground_truth, predicted)")
