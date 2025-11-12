"""
Loss functions for brain tumor segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation

    Dice Coefficient = 2 * |A ∩ B| / (|A| + |B|)
    Dice Loss = 1 - Dice Coefficient

    Args:
        smooth (float): Smoothing factor to avoid division by zero
        ignore_background (bool): Whether to exclude background class
    """

    def __init__(self, smooth=1.0, ignore_background=True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, C, H, W) - logits from model
            targets: (B, H, W) - class indices
        """
        # Convert logits to probabilities
        probs = F.softmax(predictions, dim=1)  # (B, C, H, W)

        # Convert targets to one-hot encoding
        num_classes = predictions.shape[1]
        targets_one_hot = F.one_hot(
            targets, num_classes=num_classes)  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(
            0, 3, 1, 2).float()  # (B, C, H, W)

        # Decide which classes to include
        if self.ignore_background:
            probs = probs[:, 1:, :, :]  # Exclude background (class 0)
            targets_one_hot = targets_one_hot[:, 1:, :, :]

        # Flatten spatial dimensions
        probs = probs.reshape(
            probs.shape[0], probs.shape[1], -1)  # (B, C, H*W)
        targets_one_hot = targets_one_hot.reshape(targets_one_hot.shape[0],
                                                  # (B, C, H*W)
                                                  targets_one_hot.shape[1], -1)

        # Compute Dice coefficient per class
        intersection = (probs * targets_one_hot).sum(dim=2)  # (B, C)
        union = probs.sum(dim=2) + targets_one_hot.sum(dim=2)  # (B, C)

        dice_coeff = (2. * intersection + self.smooth) / \
            (union + self.smooth)  # (B, C)

        # Average across classes and batch
        dice_loss = 1 - dice_coeff.mean()

        return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        alpha (float or list): Weighting factor for each class
        gamma (float): Focusing parameter (higher = focus more on hard examples)
        ignore_index (int): Index to ignore (e.g., padding)
    """

    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, C, H, W) - logits from model
            targets: (B, H, W) - class indices
        """
        ce_loss = F.cross_entropy(predictions, targets, reduction='none',
                                  ignore_index=self.ignore_index)  # (B, H, W)

        # Get probabilities
        probs = F.softmax(predictions, dim=1)  # (B, C, H, W)

        # Get probability of true class
        targets_one_hot = F.one_hot(
            targets, num_classes=predictions.shape[1])  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(
            0, 3, 1, 2).float()  # (B, C, H, W)

        pt = (probs * targets_one_hot).sum(dim=1)  # (B, H, W)

        # Compute focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        # Apply alpha weighting if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                # Alpha per class
                alpha = torch.tensor(self.alpha, device=predictions.device)
                alpha_t = alpha.gather(
                    0, targets.flatten()).reshape_as(targets)
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss for segmentation - generalization of Dice Loss

    Tversky Index = TP / (TP + α*FP + β*FN)
    Tversky Loss = 1 - Tversky Index

    When α = β = 0.5, it becomes Dice Loss
    When α = β = 1.0, it becomes Jaccard/IoU Loss
    Higher α penalizes false positives more
    Higher β penalizes false negatives more

    Args:
        alpha (float): Weight for false positives (default: 0.7)
        beta (float): Weight for false negatives (default: 0.3)
        smooth (float): Smoothing factor to avoid division by zero
        ignore_background (bool): Whether to exclude background class
    """

    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0, ignore_background=True):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, C, H, W) - logits from model
            targets: (B, H, W) - class indices
        """
        # Convert logits to probabilities
        probs = F.softmax(predictions, dim=1)  # (B, C, H, W)

        # Convert targets to one-hot encoding
        num_classes = predictions.shape[1]
        targets_one_hot = F.one_hot(
            targets, num_classes=num_classes)  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(
            0, 3, 1, 2).float()  # (B, C, H, W)

        # Decide which classes to include
        if self.ignore_background:
            probs = probs[:, 1:, :, :]  # Exclude background (class 0)
            targets_one_hot = targets_one_hot[:, 1:, :, :]

        # Flatten spatial dimensions
        probs = probs.reshape(
            probs.shape[0], probs.shape[1], -1)  # (B, C, H*W)
        targets_one_hot = targets_one_hot.reshape(targets_one_hot.shape[0],
                                                  # (B, C, H*W)
                                                  targets_one_hot.shape[1], -1)

        # Compute Tversky components
        true_pos = (probs * targets_one_hot).sum(dim=2)  # (B, C)
        false_pos = (probs * (1 - targets_one_hot)).sum(dim=2)  # (B, C)
        false_neg = ((1 - probs) * targets_one_hot).sum(dim=2)  # (B, C)

        # Compute Tversky index
        tversky_index = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth
        )  # (B, C)

        # Average across classes and batch
        tversky_loss = 1 - tversky_index.mean()

        return tversky_loss


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
