import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_mdar_architecture_diagram():
    """Generate a comprehensive block diagram of the MDAR architecture."""
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Define colors
    colors = {
        'input': '#E8F4FD',
        'preprocessing': '#FFE4B5',
        'conv': '#B6E5D8',
        'attention': '#FFB6C1',
        'pooling': '#DDA0DD',
        'fc': '#F0E68C',
        'output': '#98FB98',
        'text': '#2F4F4F'
    }
    
    # Title
    ax.text(10, 13.5, 'MDAR (Multi-scale Dilated Attention RPPG) Architecture', 
            fontsize=22, fontweight='bold', ha='center', color=colors['text'])
    
    # Input layer
    input_box = FancyBboxPatch((0.5, 11.5), 3, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['input'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2, 12.1, 'Input Signal\n(B, T, 4)\nR, G, B, RGB_mean', 
            fontsize=12, ha='center', va='center', fontweight='bold')
    
    # Bandpass preprocessing
    bandpass_box = FancyBboxPatch((5, 11.5), 3, 1.2,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['preprocessing'],
                                  edgecolor='black', linewidth=2)
    ax.add_patch(bandpass_box)
    ax.text(6.5, 12.1, 'Differentiable\nBandpass Filter\n(0.7-4.0 Hz)', 
            fontsize=11, ha='center', va='center', fontweight='bold')
    
    # Input projection
    proj_box = FancyBboxPatch((9.5, 11.5), 3, 1.2,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['conv'],
                              edgecolor='black', linewidth=2)
    ax.add_patch(proj_box)
    ax.text(11, 12.1, 'Input Projection\nConv1d (1x1)\n4 → 64 channels', 
            fontsize=11, ha='center', va='center', fontweight='bold')
    
    # Multi-scale branches
    branch_y = 9.5
    branch_configs = [
        ('Branch 1\nK=3, D=1', 2),
        ('Branch 2\nK=5, D=2', 5),
        ('Branch 3\nK=7, D=4', 8),
        ('Branch 4\nK=9, D=8', 11)
    ]
    
    for i, (label, x_pos) in enumerate(branch_configs):
        branch_box = FancyBboxPatch((x_pos, branch_y), 2.5, 1.2,
                                    boxstyle="round,pad=0.1",
                                    facecolor=colors['conv'],
                                    edgecolor='black', linewidth=2)
        ax.add_patch(branch_box)
        ax.text(x_pos + 1.25, branch_y + 0.6, label + '\n64 → 16 ch', 
                fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Concatenation
    concat_box = FancyBboxPatch((6, 7.5), 4, 1,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['conv'],
                                edgecolor='black', linewidth=2)
    ax.add_patch(concat_box)
    ax.text(8, 8, 'Concatenate Branches\n64 channels total', 
            fontsize=11, ha='center', va='center', fontweight='bold')
    
    # Channel attention
    ch_att_box = FancyBboxPatch((14, 7.5), 3.5, 1,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['attention'],
                                edgecolor='black', linewidth=2)
    ax.add_patch(ch_att_box)
    ax.text(15.75, 8, 'Channel Attention\nGlobal Pool + FC', 
            fontsize=11, ha='center', va='center', fontweight='bold')
    
    # Depthwise separable convolutions with residual
    ds_conv1_box = FancyBboxPatch((2, 5.5), 3.5, 1,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['conv'],
                                  edgecolor='black', linewidth=2)
    ax.add_patch(ds_conv1_box)
    ax.text(3.75, 6, 'DepthSep Conv 1\nK=5, 64→64 ch', 
            fontsize=11, ha='center', va='center', fontweight='bold')
    
    ds_conv2_box = FancyBboxPatch((7, 5.5), 3.5, 1,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['conv'],
                                  edgecolor='black', linewidth=2)
    ax.add_patch(ds_conv2_box)
    ax.text(8.75, 6, 'DepthSep Conv 2\nK=5, 64→64 ch', 
            fontsize=11, ha='center', va='center', fontweight='bold')
    
    # Residual connection
    residual_arrow = patches.FancyArrowPatch((8, 7.2), (8.75, 6.5),
                                             arrowstyle='->', mutation_scale=20,
                                             color='red', linewidth=2)
    ax.add_patch(residual_arrow)
    ax.text(8.2, 6.8, 'Residual', fontsize=10, color='red', fontweight='bold')
    
    # Temporal attention
    temp_att_box = FancyBboxPatch((12.5, 5.5), 3.5, 1,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['attention'],
                                  edgecolor='black', linewidth=2)
    ax.add_patch(temp_att_box)
    ax.text(14.25, 6, 'Temporal Attention\nConv→ReLU→Conv→Sigmoid', 
            fontsize=10, ha='center', va='center', fontweight='bold')
    
    # BatchNorm
    bn_box = FancyBboxPatch((7.5, 3.5), 3, 0.8,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['conv'],
                            edgecolor='black', linewidth=2)
    ax.add_patch(bn_box)
    ax.text(9, 3.9, 'BatchNorm1d', 
            fontsize=11, ha='center', va='center', fontweight='bold')
    
    # Multi-task heads
    # Waveform head
    waveform_box = FancyBboxPatch((2, 1.5), 3, 1,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['output'],
                                  edgecolor='black', linewidth=2)
    ax.add_patch(waveform_box)
    ax.text(3.5, 2, 'Waveform Head\nConv1d (1x1)\n64→1 channel', 
            fontsize=11, ha='center', va='center', fontweight='bold')
    
    # HR head
    hr_box = FancyBboxPatch((8, 1.5), 3, 1,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['output'],
                            edgecolor='black', linewidth=2)
    ax.add_patch(hr_box)
    ax.text(9.5, 2, 'HR Head\nGlobal Pool→FC\n64→16→1', 
            fontsize=11, ha='center', va='center', fontweight='bold')
    
    # Confidence head
    conf_box = FancyBboxPatch((14, 1.5), 3, 1,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['output'],
                              edgecolor='black', linewidth=2)
    ax.add_patch(conf_box)
    ax.text(15.5, 2, 'Confidence Head\nGlobal Pool→FC→Sigmoid\n64→16→1', 
            fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Final outputs
    waveform_out = FancyBboxPatch((2, 0.2), 3, 0.6,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['output'],
                                  edgecolor='black', linewidth=2)
    ax.add_patch(waveform_out)
    ax.text(3.5, 0.5, 'PPG Waveform\n(B, T)', 
            fontsize=11, ha='center', va='center', fontweight='bold')
    
    hr_out = FancyBboxPatch((8, 0.2), 3, 0.6,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['output'],
                            edgecolor='black', linewidth=2)
    ax.add_patch(hr_out)
    ax.text(9.5, 0.5, 'Heart Rate\n(B,)', 
            fontsize=11, ha='center', va='center', fontweight='bold')
    
    conf_out = FancyBboxPatch((14, 0.2), 3, 0.6,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['output'],
                              edgecolor='black', linewidth=2)
    ax.add_patch(conf_out)
    ax.text(15.5, 0.5, 'Confidence\n(B,)', 
            fontsize=11, ha='center', va='center', fontweight='bold')
    
    # Add arrows for data flow
    arrows = [
        # Main flow
        ((3.5, 11.5), (5, 12.1)),      # Input to bandpass
        ((8, 12.1), (9.5, 12.1)),      # Bandpass to projection
        ((11, 11.5), (8, 10.7)),       # Projection to branches (center)
        
        # From branches to concat
        ((3.25, 9.5), (7, 8.5)),       # Branch 1 to concat
        ((6.25, 9.5), (7.5, 8.5)),     # Branch 2 to concat
        ((9.25, 9.5), (8.5, 8.5)),     # Branch 3 to concat
        ((12.25, 9.5), (9, 8.5)),      # Branch 4 to concat
        
        # From concat through processing
        ((10, 8), (14, 8)),            # Concat to channel attention
        ((8, 7.5), (3.75, 6.5)),       # To DS Conv 1
        ((5.5, 6), (7, 6)),            # DS Conv 1 to DS Conv 2
        ((10.5, 6), (12.5, 6)),        # DS Conv 2 to temporal attention
        ((14.25, 5.5), (9, 4.3)),      # Temporal attention to BatchNorm
        
        # To outputs
        ((7.5, 3.5), (3.5, 2.5)),      # To waveform head
        ((9, 3.5), (9.5, 2.5)),        # To HR head
        ((10.5, 3.5), (15.5, 2.5)),    # To confidence head
        
        # Final outputs
        ((3.5, 1.5), (3.5, 0.8)),      # Waveform head to output
        ((9.5, 1.5), (9.5, 0.8)),      # HR head to output
        ((15.5, 1.5), (15.5, 0.8)),    # Confidence head to output
    ]
    
    for start, end in arrows:
        arrow = patches.FancyArrowPatch(start, end,
                                        arrowstyle='->', mutation_scale=15,
                                        color='darkblue', linewidth=1.5)
        ax.add_patch(arrow)
    
    # Add legend
    legend_x = 16.5
    legend_y = 11
    ax.text(legend_x, legend_y, 'Legend:', fontsize=14, fontweight='bold', color=colors['text'])
    
    legend_items = [
        ('Input/Data', colors['input']),
        ('Preprocessing', colors['preprocessing']),
        ('Convolution', colors['conv']),
        ('Attention', colors['attention']),
        ('Output', colors['output'])
    ]
    
    for i, (label, color) in enumerate(legend_items):
        y_pos = legend_y - 0.4 - (i * 0.4)
        legend_box = FancyBboxPatch((legend_x, y_pos - 0.1), 0.3, 0.2,
                                    boxstyle="round,pad=0.02",
                                    facecolor=color,
                                    edgecolor='black')
        ax.add_patch(legend_box)
        ax.text(legend_x + 0.4, y_pos, label, fontsize=11, va='center')
    
    # Add key features text
    features_text = """Key Features:
• Multi-scale dilated convolutions
• Channel & temporal attention
• Residual connections
• Multi-task learning
• Differentiable bandpass filter"""
    
    ax.text(0.5, 4.5, features_text, fontsize=11, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8),
            verticalalignment='top', fontweight='bold')
    
    plt.tight_layout()
    return fig

def main():
    print("="*60)
    print("GENERATING MDAR ARCHITECTURE BLOCK DIAGRAM")
    print("="*60)
    
    # Create output directory
    import os
    output_dir = "graphs"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Generate the architecture diagram
        print("Creating MDAR architecture block diagram...")
        fig = create_mdar_architecture_diagram()
        
        # Save the diagram
        output_path = os.path.join(output_dir, 'mdar_architecture_diagram.png')
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.2)
        plt.close()
        
        print("="*60)
        print("ARCHITECTURE DIAGRAM GENERATED SUCCESSFULLY!")
        print("="*60)
        print(f"Generated diagram: {output_path}")
        print("="*60)
        print("\nArchitecture Summary:")
        print("• Input: 4-channel signal (R, G, B, RGB_mean)")
        print("• Preprocessing: Differentiable bandpass filter (0.7-4.0 Hz)")
        print("• Multi-scale branches: 4 parallel conv paths with different dilations")
        print("• Attention mechanisms: Channel and temporal attention")
        print("• Processing: Depthwise separable convolutions with residual connections")
        print("• Multi-task outputs: PPG waveform, heart rate, and confidence")
        print("="*60)
        
    except Exception as e:
        print(f"Error generating architecture diagram: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
