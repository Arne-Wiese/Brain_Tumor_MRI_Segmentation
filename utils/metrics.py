"""
Evaluation metrics for brain tumor segmentation.
"""

import torch
import numpy as np


def dice_score(pred, target, num_classes=4):
    """
    Calculate Dice score for multi-class segmentation.

    Args:
        pred: Predicted logits (B, C, H, W)
        target: Ground truth labels (B, H, W)
        num_classes: Number of segmentation classes

    Returns:
        Mean Dice score across all classes
    """
    pred = torch.argmax(pred, dim=1)  # (B, H, W)
    dice_scores = []

    for class_id in range(num_classes):
        pred_class = (pred == class_id).float()
        target_class = (target == class_id).float()

        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()

        if union == 0:
            dice = 1.0  # Perfect score if both are empty
        else:
            dice = (2.0 * intersection) / union
            dice = dice.item()  # Convert tensor to float

        dice_scores.append(dice)

    return np.mean(dice_scores)


def dice_score_per_class(pred, target, num_classes=4):
    """
    Calculate Dice score for each class separately.

    Args:
        pred: Predicted logits (B, C, H, W)
        target: Ground truth labels (B, H, W)
        num_classes: Number of segmentation classes

    Returns:
        Dictionary with Dice score for each class
    """
    pred = torch.argmax(pred, dim=1)  # (B, H, W)
    dice_scores = {}

    class_names = ['Background', 'NCR/NET', 'ED', 'ET']

    for class_id in range(num_classes):
        pred_class = (pred == class_id).float()
        target_class = (target == class_id).float()

        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()

        if union == 0:
            dice = 1.0  # Perfect score if both are empty
        else:
            dice = (2.0 * intersection) / union
            dice = dice.item()  # Convert tensor to float

        dice_scores[class_names[class_id]] = dice

    return dice_scores
