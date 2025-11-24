"""
Utils package for Brain Tumor Segmentation.
"""

from .models import UNet, DoubleConv, Down, Up
from .datasets import BraTSDataset_25D
from .metrics import dice_score, dice_score_per_class, iou_score, iou_score_per_class
from .training import train_loop, train_one_epoch, validate, load_checkpoint
from .visualizations import (
    plot_training_history,
    visualize_predictions,
    visualize_single_sample,
    visualize_sample_with_overlay,
    plot_detailed_training_history
)
from .losses import (
    DiceLoss,
    FocalLoss,
    TverskyLoss,
    CombinedLoss,
)

__all__ = [
    # Models
    'UNet',
    'DoubleConv',
    'Down',
    'Up',
    # Datasets
    'BraTSDataset_25D',
    # Metrics
    'dice_score',
    'dice_score_per_class',
    'iou_score',
    'iou_score_per_class',
    # Training
    'train_loop',
    'train_one_epoch',
    'validate',
    'load_checkpoint',
    # Visualizations
    'plot_training_history',
    'visualize_predictions',
    'visualize_single_sample',
    'visualize_sample_with_overlay',
    'plot_detailed_training_history',
    # Losses
    'DiceLoss',
    'FocalLoss',
    'TverskyLoss',
    'CombinedLoss'
]
