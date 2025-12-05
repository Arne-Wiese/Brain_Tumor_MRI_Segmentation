"""
Evaluation metrics for brain tumor segmentation.
"""

import torch
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm


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


def dice_score_without_background(pred, target, num_classes=4):
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

    for class_id in range(1, num_classes):
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


def generate_confusion_matrix(model, dataloader, device, save_path=None, class_names=None):
    """
    Generate and visualize confusion matrix for segmentation model.

    Args:
        model: The segmentation model
        dataloader: DataLoader with validation/test data
        device: torch device (cuda/mps/cpu)
        save_path: Path to save the confusion matrix image (optional)
        class_names: List of class names (default: ['Background', 'Necrotic', 'Edema', 'Enhancing'])

    Returns:
        cm: Confusion matrix (numpy array)
        cm_normalized: Normalized confusion matrix
        metrics_dict: Dictionary with precision, recall, F1 for each class
    """

    if class_names is None:
        class_names = ['Background', 'Necrotic', 'Edema', 'Enhancing']

    # Collect all predictions and ground truth
    all_predictions = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Computing confusion matrix"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.append(predictions.cpu().numpy().flatten())
            all_targets.append(masks.cpu().numpy().flatten())

    # Flatten all predictions and targets
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_predictions,
                          labels=list(range(len(class_names))))

    # Normalize per row (shows how well each class is recognized)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Absolute counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=class_names,
                yticklabels=class_names)
    ax1.set_title('Confusion Matrix (Absolute Counts)',
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)

    # Normalized (in percent)
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens', ax=ax2,
                xticklabels=class_names,
                yticklabels=class_names)
    ax2.set_title('Confusion Matrix (Normalized per Class)',
                  fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")

    plt.show()

    # Calculate class-specific metrics
    metrics_dict = {}
    print("\nClass-specific Analysis:")
    print("=" * 60)

    for i, class_name in enumerate(class_names):
        true_positive = cm[i, i]
        false_positive = cm[:, i].sum() - true_positive
        false_negative = cm[i, :].sum() - true_positive

        precision = true_positive / \
            (true_positive + false_positive) if (true_positive +
                                                 false_positive) > 0 else 0
        recall = true_positive / \
            (true_positive + false_negative) if (true_positive +
                                                 false_negative) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision +
                                               recall) if (precision + recall) > 0 else 0

        metrics_dict[class_name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'true_positives': int(true_positive),
            'false_positives': int(false_positive),
            'false_negatives': int(false_negative)
        }

        print(f"\n{class_name} (Class {i}):")
        print(
            f"  Precision: {precision:.4f} (of all pixels predicted as {class_name})")
        print(f"  Recall:    {recall:.4f} (of all true {class_name} pixels)")
        print(f"  F1-Score:  {f1_score:.4f}")
        print(f"  True Positives:  {true_positive:,}")
        print(f"  False Positives: {false_positive:,}")
        print(f"  False Negatives: {false_negative:,}")

    return cm, cm_normalized, metrics_dict
