# Live Heart Rate Detection - Usage Guide

This improved system addresses fluctuation issues and provides accurate heart rate measurements using camera-based rPPG technology.

## Key Improvements Made

### 1. Signal Processing Enhancements
- **Narrower frequency band**: 45-150 BPM (0.75-2.5 Hz) instead of 48-180 BPM
- **Better filtering**: Smooth transition bands to reduce ringing artifacts
- **Enhanced detrending**: Removes DC component and linear trends
- **Zero-padding**: Better frequency resolution for more accurate peak detection
- **SNR validation**: Rejects low-quality signals (peak must be 2x stronger than average)

### 2. Physiological Constraints
- **Strict bounds**: 40-180 BPM validation on all estimates
- **Temporal consistency**: Rejects sudden changes >25 BPM unless part of consistent trend
- **Signal quality checks**: Ensures POS and CHROM methods agree (within 70%)
- **Multi-stage validation**: Individual estimate validation before fusion

### 3. Stability Features
- **Adaptive smoothing**: Conservative (20%) for large changes, normal (40%) otherwise
- **Locking mechanism**: Holds steady reading when consistent (std < 2 BPM)
- **Outlier rejection**: Identifies and discards implausible values
- **Cheek-based ROI**: Less sensitive to head movement than full-face

## Quick Start

### List Available Cameras
```bash
python list_cameras.py
```

### Basic Usage (Interactive Camera Selection)
```bash
python live_heartbeat.py
```

### Use Specific Camera (e.g., external webcam)
```bash
python live_heartbeat.py --camera 1
```

## Advanced Configuration

### Ultra-Stable Mode (Slower but More Stable)
```bash
python live_heartbeat.py --window 12.0 --lock-threshold 1.5 --unlock-threshold 6.0 --history-length 20
```

### Quick-Lock Mode (Faster but Less Stable)
```bash
python live_heartbeat.py --window 6.0 --lock-threshold 3.0 --unlock-threshold 10.0 --history-length 12
```

### Disable CLAHE (for LED/Fluorescent Lighting Issues)
```bash
python live_heartbeat.py --no-clahe
```

## Runtime Controls

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `c` | Toggle CLAHE normalization on/off |
| `r` | Reset lock and clear buffers |

## Understanding the Display

### Status Indicators
- **SEEKING**: Currently measuring, not yet locked
- **LOCKED (X.Xs)**: Steady reading held for X.X seconds
- **Buffer: XX/YYY**: Current samples / maximum buffer size
- **History: XX/YY**: Valid readings / maximum history

### Visual Elements
- **Green rectangle**: Face detection area
- **Yellow rectangles**: Left and right cheek ROIs used for measurement
- **HR value**: Current heart rate in BPM

## Troubleshooting

### Issue: Reading fluctuates wildly (>30 BPM changes)
**Solutions:**
1. Improve lighting - avoid harsh shadows or backlighting
2. Sit still and keep face centered in frame
3. Try disabling CLAHE: `--no-clahe`
4. Use ultra-stable mode (see above)

### Issue: No reading (shows 0.0 BPM)
**Solutions:**
1. Ensure good lighting on face
2. Move closer to camera (face should fill ~1/3 of frame)
3. Reduce head movement
4. Check if face detection is working (green rectangle visible)
5. Press `r` to reset if stuck

### Issue: Takes too long to lock
**Solutions:**
1. Use quick-lock mode (see above)
2. Ensure stable conditions (good lighting, minimal movement)
3. Try different camera position/angle

### Issue: Reading seems too high/low consistently
**Verification:**
1. Take manual pulse measurement for comparison
2. Note that camera-based readings have ~5-10 BPM typical error
3. Algorithm is tuned for resting heart rates (50-120 BPM range)

## Technical Parameters

### Default Settings
- **Window size**: 8 seconds of data
- **Lock threshold**: 2.0 BPM standard deviation
- **Unlock threshold**: 8.0 BPM deviation to unlock
- **History length**: 16 readings for stability analysis
- **Update rate**: Every 0.33 seconds
- **Frequency band**: 0.75-2.5 Hz (45-150 BPM)

### Recommended Conditions
- **Distance**: 0.5-2 meters from camera
- **Lighting**: Even, diffuse lighting (avoid direct sunlight)
- **Background**: Non-moving, contrasting background
- **Movement**: Minimal head movement, normal breathing
- **Duration**: Allow 10-15 seconds for initial lock

## Camera Selection Tips

### Built-in vs External Webcam
- **Built-in laptop cameras**: Often sufficient, may have lower resolution
- **External webcams**: Usually better quality, adjustable positioning
- **Recommended**: 720p or higher resolution, 30 FPS

### Positioning
- **Height**: Camera at eye level or slightly above
- **Distance**: Face should occupy 30-50% of frame height
- **Angle**: Direct face-on view, avoid extreme angles
- **Stability**: Mount camera to avoid shake

## Performance Expectations

### Accuracy
- **Typical error**: ±5-10 BPM compared to contact methods
- **Best conditions**: ±3-5 BPM
- **Difficult conditions**: ±10-15 BPM

### Response Time
- **Initial detection**: 3-10 seconds
- **Lock time**: 5-15 seconds depending on stability
- **Update rate**: New reading every ~0.33 seconds

### Limitations
- **Motion sensitivity**: Head movement affects accuracy
- **Lighting dependency**: Poor lighting degrades performance
- **Skin tone**: Works across different skin tones but lighting matters more
- **Not medical grade**: For research/fitness use only

## File Structure

```
E:\rppg\
├── live_heartbeat.py      # Main application
├── list_cameras.py        # Camera detection utility
├── train_mdar.py          # Model training (enhanced)
├── eval_model.py          # Model evaluation
├── models/
│   └── mdar.py           # MDAR neural network model
├── data/
│   └── ubfc_dataset.py   # Dataset loader
└── USAGE_GUIDE.md        # This guide
```

## Training Your Own Model

### Quick Training
```bash
python train_mdar.py
```

### Evaluate Trained Model
```bash
python eval_model.py --model_path outputs/mdar_enhanced/mdar_best.pth
```

The training uses improved settings for better accuracy:
- 85/15 train/validation split
- Enhanced loss function with frequency domain components
- Cosine annealing learning rate schedule
- Better data augmentation and regularization

---

**Note**: This system is for research and fitness use only. It is not intended for medical diagnosis or clinical applications.
