"""
Visualization utilities for brain tumor segmentation.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def visualize_predictions(model, dataloader, device, num_samples=3, save_path=None):
    """
    Visualize predictions from the model.

    Args:
        model: Trained model
        dataloader: DataLoader to get samples from
        device: Device to run inference on
        num_samples: Number of samples to visualize
        save_path: Optional path to save the figure
    """
    model.eval()

    # Get a batch
    images, masks = next(iter(dataloader))

    # Randomly select samples from the batch
    batch_size = images.size(0)
    if num_samples > batch_size:
        num_samples = batch_size
        print(f"Warning: num_samples reduced to {batch_size} (batch size)")

    # Random indices from the batch
    random_indices = torch.randperm(batch_size)[:num_samples]
    images = images[random_indices].to(device)
    masks = masks[random_indices].to(device)

    with torch.no_grad():
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

    # Move to CPU for visualization
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    predictions = predictions.cpu().numpy()

    # Get the actual colors from tab10 colormap
    tab10 = plt.get_cmap('tab10')
    colors = [tab10(i) for i in range(13)]

    # Plot
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))

    # Handle case when num_samples = 1
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Show FLAIR modality
        axes[i, 0].imshow(images[i, 0], cmap='gray')
        axes[i, 0].set_title('FLAIR')
        axes[i, 0].axis('off')

        # Show T1 modality
        axes[i, 1].imshow(images[i, 1], cmap='gray')
        axes[i, 1].set_title('T1')
        axes[i, 1].axis('off')

        # Show T1CE modality
        axes[i, 2].imshow(images[i, 2], cmap='gray')
        axes[i, 2].set_title('T1CE')
        axes[i, 2].axis('off')

        # Show ground truth mask
        axes[i, 3].imshow(masks[i], cmap='tab10', vmin=0, vmax=3)
        axes[i, 3].set_title('Ground Truth')
        axes[i, 3].axis('off')

        # Show prediction
        axes[i, 4].imshow(predictions[i], cmap='tab10', vmin=0, vmax=3)
        axes[i, 4].set_title('Prediction')
        axes[i, 4].axis('off')

     # Add legend with correct tab10 colors
    legend_elements = [
        Patch(facecolor=colors[0], label='Background (0)'),
        Patch(facecolor=colors[3], label='NCR/NET (1)'),
        Patch(facecolor=colors[6], label='Edema (2)'),
        Patch(facecolor=colors[12], label='Enhancing Tumor (3)')
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=10)

    plt.tight_layout()

    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved predictions to {save_path}")

    plt.show()


def visualize_single_sample(image, mask=None, prediction=None, save_path=None):
    """
    Visualize a single sample with all modalities.

    Args:
        image: Image tensor (4, H, W) - 4 modalities
        mask: Ground truth mask (H, W) - optional
        prediction: Predicted mask (H, W) - optional
        save_path: Optional path to save the figure
    """
    modality_names = ['FLAIR', 'T1', 'T1CE', 'T2']

    n_cols = 4 + (mask is not None) + (prediction is not None)
    fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 4))

    # Show all modalities
    for i in range(4):
        axes[i].imshow(image[i], cmap='gray')
        axes[i].set_title(modality_names[i])
        axes[i].axis('off')

    col_idx = 4

    # Show ground truth if provided
    if mask is not None:
        axes[col_idx].imshow(mask, cmap='tab10', vmin=0, vmax=3)
        axes[col_idx].set_title('Ground Truth')
        axes[col_idx].axis('off')
        col_idx += 1

    # Show prediction if provided
    if prediction is not None:
        axes[col_idx].imshow(prediction, cmap='tab10', vmin=0, vmax=3)
        axes[col_idx].set_title('Prediction')
        axes[col_idx].axis('off')

    plt.tight_layout()

    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")

    plt.show()


def plot_detailed_training_history(history, save_path=None):
    """
    Plot detailed training history with multiple metrics.

    Args:
        history: Dictionary containing training history with keys:
                 - train_loss, val_loss
                 - train_dice, val_dice
                 - train_iou, val_iou (optional)
                 - learning_rates (optional)
                 - epoch_times (optional)
        save_path: Optional path to save the figure
    """
    # Determine subplot layout based on available metrics
    has_iou = 'train_iou' in history and 'val_iou' in history
    has_lr = 'learning_rates' in history

    if has_iou and has_lr:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
    elif has_iou or has_lr:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot 1: Loss
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(epochs, history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Dice Score
    axes[1].plot(epochs, history['train_dice'], label='Train Dice', marker='o')
    axes[1].plot(epochs, history['val_dice'], label='Val Dice', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_title('Training and Validation Dice Score')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_ylim([0, 1])

    plot_idx = 2

    # Plot 3: IoU (if available)
    if has_iou and len(axes) > 2:
        axes[plot_idx].plot(epochs, history['train_iou'],
                            label='Train IoU', marker='o')
        axes[plot_idx].plot(epochs, history['val_iou'],
                            label='Val IoU', marker='s')
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('IoU')
        axes[plot_idx].set_title('Training and Validation IoU')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True)
        axes[plot_idx].set_ylim([0, 1])
        plot_idx += 1

    # Plot 4: Learning Rate (if available)
    if has_lr and len(axes) > plot_idx:
        axes[plot_idx].plot(epochs, history['learning_rates'], marker='o')
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Learning Rate')
        axes[plot_idx].set_title('Learning Rate Schedule')
        axes[plot_idx].grid(True)
        axes[plot_idx].set_yscale('log')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved detailed training history to {save_path}")

    plt.show()

    # Print summary statistics
    if 'epoch_times' in history:
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        print(f"Total epochs:          {len(epochs)}")
        print(
            f"Best validation Dice:  {max(history['val_dice']):.4f} (Epoch {np.argmax(history['val_dice']) + 1})")
        print(f"Final validation Dice: {history['val_dice'][-1]:.4f}")
        print(
            f"Best validation loss:  {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss']) + 1})")
        print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
        print(
            f"Total training time:   {sum(history['epoch_times'])/3600:.2f} hours")
        print(
            f"Avg time per epoch:    {np.mean(history['epoch_times'])/60:.2f} minutes")
        print("="*80)


def visualize_sample_with_overlay(image, mask=None, prediction=None, alpha=0.4, save_path=None):
    """
    Visualize a single sample with segmentation overlay on MRI modalities.

    Args:
        image: Image tensor (4, H, W) - 4 modalities
        mask: Ground truth mask (H, W) - optional
        prediction: Predicted mask (H, W) - optional
        alpha: Transparency for overlay
        save_path: Optional path to save the figure
    """
    modality_names = ['FLAIR', 'T1', 'T1CE', 'T2']

    n_rows = 2 if (mask is not None and prediction is not None) else 1
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4*n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Show modalities with ground truth overlay
    if mask is not None:
        for i in range(4):
            axes[0, i].imshow(image[i], cmap='gray')
            # Overlay mask (only non-background)
            mask_overlay = np.ma.masked_where(mask == 0, mask)
            axes[0, i].imshow(mask_overlay, cmap='jet',
                              alpha=alpha, vmin=0, vmax=3)
            axes[0, i].set_title(
                f'{modality_names[i]} + Ground Truth', fontsize=10)
            axes[0, i].axis('off')
    else:
        for i in range(4):
            axes[0, i].imshow(image[i], cmap='gray')
            axes[0, i].set_title(modality_names[i], fontsize=10)
            axes[0, i].axis('off')

    # Show modalities with prediction overlay
    if prediction is not None and n_rows == 2:
        for i in range(4):
            axes[1, i].imshow(image[i], cmap='gray')
            # Overlay prediction (only non-background)
            pred_overlay = np.ma.masked_where(prediction == 0, prediction)
            axes[1, i].imshow(pred_overlay, cmap='jet',
                              alpha=alpha, vmin=0, vmax=3)
            axes[1, i].set_title(
                f'{modality_names[i]} + Prediction', fontsize=10)
            axes[1, i].axis('off')

    # Add legend
    legend_elements = [
        Patch(facecolor='cyan', label='NCR/NET'),
        Patch(facecolor='yellow', label='Edema'),
        Patch(facecolor='red', label='Enhancing')
    ]
    fig.legend(handles=legend_elements, loc='upper center',
               bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=10)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved overlay visualization to {save_path}")

    plt.show()
