# Assets

This folder contains generated visualizations, plots, and results from model training and evaluation.

## Structure

Each model version gets its own subdirectory:

```
assets/
├── best_model_v1.pth/
│   ├── training_history.png
│   ├── detailed_training_history.png
│   ├── val_predictions.png
│   ├── test_predictions.png
│   ├── overlay_sample_*.png
│   └── ...
├── model_v2/
│   └── ...
```

## Contents

### Training Visualizations

- **`training_history.png`** - Basic training curves (loss & metrics)
- **`detailed_training_history.png`** - Extended plots with IoU, learning rate, etc.

### Prediction Visualizations

- **`val_predictions.png`** - Validation set predictions
- **`test_predictions.png`** - Test set predictions
- **`predictions_input.png`** - All MRI modalities with ground truth
- **`predictions_comparison.png`** - Ground truth vs prediction comparison

### Overlay Visualizations

- **`overlay_sample_*.png`** - Segmentation overlays on MRI modalities
- Transparent tumor regions overlaid on brain scans

## Notes

- All visualizations are saved at 300 DPI for publication quality
- PNG format is used for lossless quality
- Large image files should be excluded from Git using `.gitignore`
- Organize by model version for easy comparison
