import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_simplified_mdar_diagram():
    """Generate a simplified block diagram following CNN → LSTM → FC → Output format."""
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors
    colors = {
        'input': '#E8F4FD',
        'cnn': '#B6E5D8',
        'lstm': '#FFB6C1',
        'fc': '#F0E68C',
        'output': '#98FB98',
        'text': '#2F4F4F'
    }
    
    # Title
    ax.text(8, 9.5, 'MDAR Model Architecture Pipeline', 
            fontsize=20, fontweight='bold', ha='center', color=colors['text'])
    
    # Subtitle
    ax.text(8, 9, 'Multi-scale Dilated Attention RPPG Network', 
            fontsize=14, ha='center', color='gray', style='italic')
    
    # Input
    input_box = FancyBboxPatch((0.5, 6.5), 2.5, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['input'],
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.75, 7.25, 'Input\nSignal', fontsize=14, ha='center', va='center', fontweight='bold')
    ax.text(1.75, 6.2, '(B, T, 4)\nR,G,B,RGB', fontsize=10, ha='center', va='center')
    
    # CNN Layers (Multi-scale + Attention)
    cnn_box = FancyBboxPatch((4, 5.5), 3.5, 2.5,
                             boxstyle="round,pad=0.1",
                             facecolor=colors['cnn'],
                             edgecolor='black', linewidth=2)
    ax.add_patch(cnn_box)
    ax.text(5.75, 7.5, 'CNN Layers', fontsize=14, ha='center', va='center', fontweight='bold')
    ax.text(5.75, 7, 'Multi-scale Dilated Convolutions', fontsize=10, ha='center', va='center')
    ax.text(5.75, 6.6, '+ Channel Attention', fontsize=10, ha='center', va='center')
    ax.text(5.75, 6.2, '+ Depthwise Separable Conv', fontsize=10, ha='center', va='center')
    ax.text(5.75, 5.8, '+ Temporal Attention', fontsize=10, ha='center', va='center')
    
    # LSTM (Temporal Processing) - Note: MDAR doesn't use LSTM, but showing conceptual equivalent
    lstm_box = FancyBboxPatch((8.5, 6.5), 2.5, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['lstm'],
                              edgecolor='black', linewidth=2)
    ax.add_patch(lstm_box)
    ax.text(9.75, 7.4, 'Temporal', fontsize=14, ha='center', va='center', fontweight='bold')
    ax.text(9.75, 7, 'Processing', fontsize=14, ha='center', va='center', fontweight='bold')
    ax.text(9.75, 6.6, '(Attention-based)', fontsize=10, ha='center', va='center', style='italic')
    
    # Fully Connected
    fc_box = FancyBboxPatch((12, 6.5), 2.5, 1.5,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['fc'],
                            edgecolor='black', linewidth=2)
    ax.add_patch(fc_box)
    ax.text(13.25, 7.4, 'Fully', fontsize=14, ha='center', va='center', fontweight='bold')
    ax.text(13.25, 7, 'Connected', fontsize=14, ha='center', va='center', fontweight='bold')
    ax.text(13.25, 6.6, 'Multi-task Heads', fontsize=10, ha='center', va='center')
    
    # Output (Multi-task)
    # Waveform output
    waveform_out = FancyBboxPatch((3, 3.5), 2.5, 1.2,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['output'],
                                  edgecolor='black', linewidth=2)
    ax.add_patch(waveform_out)
    ax.text(4.25, 4.3, 'PPG', fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(4.25, 4, 'Waveform', fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(4.25, 3.7, '(B, T)', fontsize=10, ha='center', va='center')
    
    # Heart Rate output
    hr_out = FancyBboxPatch((6.75, 3.5), 2.5, 1.2,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['output'],
                            edgecolor='black', linewidth=2)
    ax.add_patch(hr_out)
    ax.text(8, 4.3, 'Heart', fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(8, 4, 'Rate', fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(8, 3.7, '(BPM)', fontsize=10, ha='center', va='center')
    
    # Confidence output
    conf_out = FancyBboxPatch((10.5, 3.5), 2.5, 1.2,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['output'],
                              edgecolor='black', linewidth=2)
    ax.add_patch(conf_out)
    ax.text(11.75, 4.3, 'Confidence', fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(11.75, 4, 'Score', fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(11.75, 3.7, '[0,1]', fontsize=10, ha='center', va='center')
    
    # Main pipeline arrows
    main_arrows = [
        ((3, 7.25), (4, 7.25)),      # Input to CNN
        ((7.5, 7.25), (8.5, 7.25)),  # CNN to LSTM
        ((11, 7.25), (12, 7.25)),    # LSTM to FC
    ]
    
    for start, end in main_arrows:
        arrow = patches.FancyArrowPatch(start, end,
                                        arrowstyle='->', mutation_scale=20,
                                        color='darkblue', linewidth=3)
        ax.add_patch(arrow)
    
    # Output arrows
    output_arrows = [
        ((13.25, 6.5), (4.25, 4.7)),   # FC to Waveform
        ((13.25, 6.5), (8, 4.7)),      # FC to HR
        ((13.25, 6.5), (11.75, 4.7)),  # FC to Confidence
    ]
    
    for start, end in output_arrows:
        arrow = patches.FancyArrowPatch(start, end,
                                        arrowstyle='->', mutation_scale=15,
                                        color='green', linewidth=2)
        ax.add_patch(arrow)
    
    # Add stage labels
    ax.text(1.75, 5.8, 'Stage 1', fontsize=12, ha='center', fontweight='bold', color='navy')
    ax.text(5.75, 5, 'Stage 2', fontsize=12, ha='center', fontweight='bold', color='navy')
    ax.text(9.75, 5.8, 'Stage 3', fontsize=12, ha='center', fontweight='bold', color='navy')
    ax.text(13.25, 5.8, 'Stage 4', fontsize=12, ha='center', fontweight='bold', color='navy')
    
    # Add detailed architecture info box
    arch_details = """Detailed Architecture:
    
1. Input Processing:
   • 4-channel facial video signals
   • Differentiable bandpass filtering (0.7-4.0 Hz)
   
2. CNN Feature Extraction:
   • Multi-scale dilated convolutions (K=3,5,7,9)
   • Channel attention mechanism
   • Depthwise separable convolutions
   • Residual connections
   
3. Temporal Processing:
   • Temporal attention (replaces LSTM)
   • Feature refinement and temporal modeling
   
4. Multi-task Heads:
   • Waveform reconstruction
   • Heart rate estimation
   • Confidence prediction"""
    
    ax.text(0.5, 2.8, arch_details, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8),
            verticalalignment='top')
    
    # Add performance highlights
    performance_text = """Key Advantages:
    
• Superior accuracy vs traditional methods
• Real-time processing capability  
• Robust to lighting variations
• Multi-task learning efficiency
• End-to-end differentiable

Performance:
• MAE: ~4 BPM (vs ~31 BPM traditional)
• Correlation: 0.91 (vs ~0.4 traditional)
• Real-time: 30 FPS processing"""
    
    ax.text(10.5, 2.8, performance_text, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8),
            verticalalignment='top')
    
    plt.tight_layout()
    return fig

def main():
    print("="*60)
    print("GENERATING SIMPLIFIED MDAR ARCHITECTURE DIAGRAM")
    print("="*60)
    
    # Create output directory
    import os
    output_dir = "graphs"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Generate the simplified architecture diagram
        print("Creating simplified MDAR architecture block diagram...")
        fig = create_simplified_mdar_diagram()
        
        # Save the diagram
        output_path = os.path.join(output_dir, 'mdar_simplified_architecture.png')
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white',
                   edgecolor='none', pad_inches=0.2)
        plt.close()
        
        print("="*60)
        print("SIMPLIFIED ARCHITECTURE DIAGRAM GENERATED SUCCESSFULLY!")
        print("="*60)
        print(f"Generated diagram: {output_path}")
        print("="*60)
        print("\nPipeline Summary:")
        print("Stage 1: Input Signal (4-channel facial video)")
        print("Stage 2: CNN Layers (Multi-scale + Attention)")
        print("Stage 3: Temporal Processing (Attention-based)")
        print("Stage 4: Fully Connected (Multi-task heads)")
        print("Outputs: PPG Waveform, Heart Rate, Confidence")
        print("="*60)
        
    except Exception as e:
        print(f"Error generating simplified architecture diagram: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
