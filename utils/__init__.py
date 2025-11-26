"""
Utils package for Brain Tumor Segmentation.
"""

from .models import UNet, DoubleConv, Down, Up
from .datasets import BraTSDataset_25D, BraTSDataset_2D
from .metrics import dice_score, iou_score
from .training import train_loop, train_one_epoch, validate, load_checkpoint
from .visualizations import (
    visualize_predictions,
    visualize_single_sample,
    visualize_sample_with_overlay,
    plot_detailed_training_history,
    history_to_json,
    display_patient_images,
    patient_information,
)
from .losses import (
    DiceLoss,
    CombinedLoss,
)
from .hpc import load_dataset_into_ram, running_on_hpc

__all__ = [
    # Models
    'UNet',
    'DoubleConv',
    'Down',
    'Up',
    # Datasets
    'BraTSDataset_25D',
    'BraTSDataset_2D',
    # Metrics
    'dice_score',
    'iou_score',
    # Training
    'train_loop',
    'train_one_epoch',
    'validate',
    'load_checkpoint',
    # Visualizations
    'visualize_predictions',
    'visualize_single_sample',
    'visualize_sample_with_overlay',
    'plot_detailed_training_history',
    'history_to_json',
    'display_patient_images',
    'patient_information',
    # Losses
    'DiceLoss',
    'CombinedLoss'
]
