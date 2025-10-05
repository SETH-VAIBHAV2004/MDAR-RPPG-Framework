# Heart Rate Prediction Analysis Graphs 📊

This directory contains generated heart rate comparison graphs and the scripts used to create them.

## Generated Files

### 📈 Main Analysis Graphs
- **`hr_prediction_comparison.png`** - Comprehensive comparison with multiple visualizations
- **`hr_statistical_analysis.png`** - Statistical analysis including Bland-Altman plots
- **`simple_hr_comparison.png`** - Simple two-panel comparison plot

## Graph Scripts

### 🔧 `generate_hr_graph.py` - Comprehensive Analysis
Advanced script that generates detailed statistical analysis including:
- Time series comparison with error visualization
- Bland-Altman plots for agreement analysis
- Error distribution histograms
- Scatter plots with regression analysis
- Box plots for distribution comparison
- Performance metrics (MAE, RMSE, Correlation)

**Usage:**
```python
python generate_hr_graph.py
```

### 🎯 `simple_hr_graph.py` - Easy Customization
Simplified script for quick heart rate comparisons:
- Two-panel layout (comparison + error)
- Basic performance metrics
- Easy to customize with your own data

**Usage with your own data:**
```python
from simple_hr_graph import create_simple_hr_graph
import numpy as np

# Your data arrays
ground_truth = np.array([70, 72, 74, 73, 71, ...])  # Your ground truth data
predicted = np.array([71, 73, 75, 72, 70, ...])     # Your prediction data
time_points = np.array([0, 1, 2, 3, 4, ...])        # Optional time data

# Generate the graph
create_simple_hr_graph(ground_truth, predicted, time_points, "my_hr_analysis.png")
```

## Graph Features

### 📊 Visualization Components

1. **Time Series Plot**
   - Ground truth (blue solid line)
   - Predictions (red dashed line)
   - Error region (gray fill)

2. **Error Analysis**
   - Prediction error over time
   - Zero-error reference line
   - Error distribution visualization

3. **Statistical Metrics**
   - **MAE (Mean Absolute Error)**: Average absolute difference
   - **RMSE (Root Mean Square Error)**: Square root of mean squared errors
   - **Correlation**: Linear relationship strength (-1 to 1)

### 🎨 Customization Options

You can easily modify the scripts to:
- Change colors and styling
- Add different metrics
- Modify plot layouts
- Include additional analysis
- Export different file formats

## Understanding the Metrics

- **MAE < 3 BPM**: Excellent accuracy
- **MAE 3-5 BPM**: Good accuracy  
- **MAE > 5 BPM**: May need improvement
- **Correlation > 0.8**: Strong agreement
- **Correlation 0.6-0.8**: Moderate agreement
- **Correlation < 0.6**: Weak agreement

## Requirements

```
numpy
matplotlib
scipy
seaborn
```

Install with: `pip install numpy matplotlib scipy seaborn`

## rPPG Methods Analyzed

The graphs are designed for analysis of these rPPG methods:
- 🧠 **MDAR Neural Network** - Deep learning approach
- 📐 **POS Algorithm** - Plane Orthogonal to Skin method
- 🎨 **CHROM Method** - Chrominance-based detection
- 🔄 **Ensemble Fusion** - Combined approach

---

*Generated for Remote Photoplethysmography (rPPG) heart rate analysis*
