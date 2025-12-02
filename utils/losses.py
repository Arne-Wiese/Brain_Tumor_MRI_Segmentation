"""
Loss functions for brain tumor segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_background=True):
        super().__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits, targets):
        """
        logits:  (B, C, H, W) or (B, C, D, H, W)
        targets: (B, H, W)     or (B, D, H, W)
        """
        B, C = logits.shape[:2]

        # softmax
        probs = F.softmax(logits, dim=1)

        # one-hot
        targets_onehot = F.one_hot(targets, num_classes=C)  # (..., C)
        # move class to channel dim
        targets_onehot = targets_onehot.permute(0, -1, *range(1, targets.dim()))
        targets_onehot = targets_onehot.float()  # (B, C, ...)

        # optionally drop background
        if self.ignore_background:
            probs = probs[:, 1:]
            targets_onehot = targets_onehot[:, 1:]

        # flatten all spatial dims
        probs = probs.reshape(B, probs.shape[1], -1)
        targets_onehot = targets_onehot.reshape(B, targets_onehot.shape[1], -1)

        # dice per-class
        intersection = (probs * targets_onehot).sum(dim=2)
        union = probs.sum(dim=2) + targets_onehot.sum(dim=2)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Combined Cross-Entropy + Dice Loss

    Total Loss = λ * CE_Loss + (1 - λ) * Dice_Loss

    Args:
        ce_weight (float): Weight for cross-entropy loss
        dice_weight (float): Weight for dice loss
        class_weights (list or tensor): Class weights for CE loss
        smooth (float): Smoothing factor for Dice loss
        ignore_background (bool): Ignore background in Dice loss
    """

    def __init__(self, ce_weight=0.5, dice_weight=0.5, class_weights=None,
                 smooth=1.0, ignore_background=True):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        # Cross-Entropy Loss
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

        # Dice Loss
        self.dice_loss = DiceLoss(
            smooth=smooth, ignore_background=ignore_background)

    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, C, H, W) - logits from model
            targets: (B, H, W) - class indices

        Returns:
            total_loss: Combined loss
            ce: Cross-entropy loss component
            dice: Dice loss component
        """
        ce = self.ce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)

        total_loss = self.ce_weight * ce + self.dice_weight * dice

        return total_loss
