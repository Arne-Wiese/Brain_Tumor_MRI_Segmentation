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


def iou_score(pred, target, num_classes=4):
    """
    Calculate IoU (Intersection over Union) score for multi-class segmentation.

    Args:
        pred: Predicted logits (B, C, H, W)
        target: Ground truth labels (B, H, W)
        num_classes: Number of segmentation classes

    Returns:
        Mean IoU score across all classes
    """
    pred = torch.argmax(pred, dim=1)  # (B, H, W)
    iou_scores = []

    for class_id in range(num_classes):
        pred_class = (pred == class_id).float()
        target_class = (target == class_id).float()

        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum() - intersection

        if union == 0:
            iou = 1.0  # Perfect score if both are empty
        else:
            iou = intersection / union
            iou = iou.item()  # Convert tensor to float

        iou_scores.append(iou)

    return np.mean(iou_scores)


def dice_score_background(pred, target):
    """Calculate Dice score for background class (class 0)."""
    pred = torch.argmax(pred, dim=1)
    pred_class = (pred == 0).float()
    target_class = (target == 0).float()

    intersection = (pred_class * target_class).sum()
    union = pred_class.sum() + target_class.sum()

    if union == 0:
        return 1.0
    else:
        dice = (2.0 * intersection) / union
        return dice.item()


def dice_score_necrotic(pred, target):
    """Calculate Dice score for necrotic tumor core class (class 1)."""
    pred = torch.argmax(pred, dim=1)
    pred_class = (pred == 1).float()
    target_class = (target == 1).float()

    intersection = (pred_class * target_class).sum()
    union = pred_class.sum() + target_class.sum()

    if union == 0:
        return 1.0
    else:
        dice = (2.0 * intersection) / union
        return dice.item()


def dice_score_edema(pred, target):
    """Calculate Dice score for peritumoral edema class (class 2)."""
    pred = torch.argmax(pred, dim=1)
    pred_class = (pred == 2).float()
    target_class = (target == 2).float()

    intersection = (pred_class * target_class).sum()
    union = pred_class.sum() + target_class.sum()

    if union == 0:
        return 1.0
    else:
        dice = (2.0 * intersection) / union
        return dice.item()


def dice_score_enhancing(pred, target):
    """Calculate Dice score for enhancing tumor class (class 3)."""
    pred = torch.argmax(pred, dim=1)
    pred_class = (pred == 3).float()
    target_class = (target == 3).float()

    intersection = (pred_class * target_class).sum()
    union = pred_class.sum() + target_class.sum()

    if union == 0:
        return 1.0
    else:
        dice = (2.0 * intersection) / union
        return dice.item()


def iou_score_background(pred, target):
    """Calculate IoU score for background class (class 0)."""
    pred = torch.argmax(pred, dim=1)
    pred_class = (pred == 0).float()
    target_class = (target == 0).float()

    intersection = (pred_class * target_class).sum()
    union = pred_class.sum() + target_class.sum() - intersection

    if union == 0:
        return 1.0
    else:
        iou = intersection / union
        return iou.item()


def iou_score_necrotic(pred, target):
    """Calculate IoU score for necrotic tumor core class (class 1)."""
    pred = torch.argmax(pred, dim=1)
    pred_class = (pred == 1).float()
    target_class = (target == 1).float()

    intersection = (pred_class * target_class).sum()
    union = pred_class.sum() + target_class.sum() - intersection

    if union == 0:
        return 1.0
    else:
        iou = intersection / union
        return iou.item()


def iou_score_edema(pred, target):
    """Calculate IoU score for peritumoral edema class (class 2)."""
    pred = torch.argmax(pred, dim=1)
    pred_class = (pred == 2).float()
    target_class = (target == 2).float()

    intersection = (pred_class * target_class).sum()
    union = pred_class.sum() + target_class.sum() - intersection

    if union == 0:
        return 1.0
    else:
        iou = intersection / union
        return iou.item()


def iou_score_enhancing(pred, target):
    """Calculate IoU score for enhancing tumor class (class 3)."""
    pred = torch.argmax(pred, dim=1)
    pred_class = (pred == 3).float()
    target_class = (target == 3).float()

    intersection = (pred_class * target_class).sum()
    union = pred_class.sum() + target_class.sum() - intersection

    if union == 0:
        return 1.0
    else:
        iou = intersection / union
        return iou.item()
