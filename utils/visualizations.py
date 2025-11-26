"""
Visualization utilities for brain tumor segmentation.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import json
import nibabel as nib


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

    # Get random samples from the entire dataset
    dataset = dataloader.dataset
    random_indices = torch.randperm(len(dataset))[:num_samples]

    images_list = []
    masks_list = []

    for idx in random_indices:
        img, mask = dataset[idx]
        images_list.append(img)
        masks_list.append(mask)

    images = torch.stack(images_list).to(device)
    masks = torch.stack(masks_list).to(device)

    with torch.no_grad():
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

    # Move to CPU for visualization
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    predictions = predictions.cpu().numpy()

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
        axes[i, 3].imshow(masks[i], cmap='jet', vmin=0, vmax=3)
        axes[i, 3].set_title('Ground Truth')
        axes[i, 3].axis('off')

        # Show prediction
        axes[i, 4].imshow(predictions[i], cmap='jet', vmin=0, vmax=3)
        axes[i, 4].set_title('Prediction')
        axes[i, 4].axis('off')

     # Add legend with correct tab10 colors
    legend_elements = [
        Patch(facecolor='cyan', label='NCR/NET (1)'),
        Patch(facecolor='yellow', label='Edema (2)'),
        Patch(facecolor='red', label='Enhancing (3)')
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=10)

    plt.tight_layout()
    # add title to the figure
    plt.suptitle('Model Predictions vs Ground Truth', fontsize=16)

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
    if 'epoch_train_times' in history:
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
            f"Total training time:   {sum(history['epoch_train_times'])/3600:.2f} hours")
        print(
            f"Avg time per epoch:    {np.mean(history['epoch_train_times'])/60:.2f} minutes")
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

    # Determine number of rows
    if mask is not None and prediction is not None:
        n_rows = 3  # Original, Ground Truth, Prediction
    elif mask is not None or prediction is not None:
        n_rows = 2  # Original + one overlay
    else:
        n_rows = 1  # Only original

    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4*n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Row 1: Show original modalities without overlay
    for i in range(4):
        axes[0, i].imshow(image[i], cmap='gray')
        axes[0, i].set_title(f'{modality_names[i]}', fontsize=10)
        axes[0, i].axis('off')

    current_row = 1

    # Row 2: Show modalities with ground truth overlay
    if mask is not None:
        for i in range(4):
            axes[current_row, i].imshow(image[i], cmap='gray')
            # Overlay mask (only non-background)
            mask_overlay = np.ma.masked_where(mask == 0, mask)
            axes[current_row, i].imshow(mask_overlay, cmap='jet',
                                        alpha=alpha, vmin=0, vmax=3)
            axes[current_row, i].set_title(
                f'{modality_names[i]} + Ground Truth', fontsize=10)
            axes[current_row, i].axis('off')
        current_row += 1

    # Row 3: Show modalities with prediction overlay
    if prediction is not None:
        for i in range(4):
            axes[current_row, i].imshow(image[i], cmap='gray')
            # Overlay prediction (only non-background)
            pred_overlay = np.ma.masked_where(prediction == 0, prediction)
            axes[current_row, i].imshow(pred_overlay, cmap='jet',
                                        alpha=alpha, vmin=0, vmax=3)
            axes[current_row, i].set_title(
                f'{modality_names[i]} + Prediction', fontsize=10)
            axes[current_row, i].axis('off')

    # Add legend
    legend_elements = [
        Patch(facecolor='cyan', label='NCR/NET (1)'),
        Patch(facecolor='yellow', label='Edema (2)'),
        Patch(facecolor='red', label='Enhancing (3)')
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=10)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved overlay visualization to {save_path}")

    plt.show()


def history_to_json(history, model_name, save_dir='logs', **kwargs):
    """
    Save training history to a JSON file.

    Args:
        history: Dictionary containing training history
            - 'train_loss': List of training losses
            - 'val_loss': List of validation losses
            - 'train_{metric}': List of training metrics
            - 'val_{metric}': List of validation metrics
            - 'epoch_train_times': List of epoch times in seconds
            - 'learning_rates': List of learning rates
        model_name: Name of the model (used for filename)
        save_dir: Directory to save the JSON file (default: 'logs')
        **kwargs: Additional parameters to save (e.g., batch_size, num_slices, slice_range, etc.)

    Returns:
        save_path: Path where the JSON file was saved
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create filename from model name
    json_filename = f"{model_name.replace('.pth', '')}_history.json"
    save_path = os.path.join(save_dir, json_filename)

    # Add summary statistics
    history_with_stats = history.copy()

    # Calculate total and average epoch time
    if 'epoch_train_times' in history:
        total_time = sum(history['epoch_train_times'])
        avg_time = total_time / \
            len(history['epoch_train_times']
                ) if history['epoch_train_times'] else 0

        history_with_stats['summary'] = {
            'total_epochs': len(history['epoch_train_times']),
            'total_training_time_seconds': round(total_time, 2),
            'total_training_time_minutes': round(total_time / 60, 2),
            'average_epoch_time_seconds': round(avg_time, 2),
            'best_train_loss': round(min(history['train_loss']), 4) if history['train_loss'] else None,
            'best_val_loss': round(min(history['val_loss']), 4) if history['val_loss'] else None,
        }

        # Add learning rate info
        if 'learning_rates' in history and history['learning_rates']:
            history_with_stats['summary']['initial_learning_rate'] = history['learning_rates'][0]
            history_with_stats['summary']['final_learning_rate'] = history['learning_rates'][-1]
            history_with_stats['summary']['min_learning_rate'] = min(
                history['learning_rates'])

        # Add best metrics if available
        for key in history.keys():
            if key.startswith('val_') and key not in ['val_loss']:
                metric_name = key.replace('val_', '')
                history_with_stats['summary'][f'best_val_{metric_name}'] = round(
                    max(history[key]), 4)

    # Add training configuration parameters
    if kwargs:
        history_with_stats['config'] = {}
        for key, value in kwargs.items():
            # Convert tuples to lists for JSON serialization
            if isinstance(value, tuple):
                history_with_stats['config'][key] = list(value)
            else:
                history_with_stats['config'][key] = value

    # Convert to JSON-serializable format (round floats)
    json_history = {}
    for key, value in history_with_stats.items():
        if isinstance(value, list):
            json_history[key] = [round(v, 6) if isinstance(
                v, float) else v for v in value]
        else:
            json_history[key] = value

    # Save to JSON
    with open(save_path, 'w') as f:
        json.dump(json_history, f, indent=4)

    print(f"✓ Saved training history to {save_path}")
    return save_path


def display_patient_images(path, patient_id, slice_idx):
    patient_path = f'{path}/{patient_id}'

    modalities = ['flair', 't1', 't1ce', 't2']
    imgs = []

    for mod in modalities:
        file_path = os.path.join(patient_path, f'{patient_id}_{mod}.nii')
        img = nib.load(file_path).get_fdata(dtype=np.float32)  # type: ignore
        imgs.append(img)

    seg_path = os.path.join(patient_path, f'{patient_id}_seg.nii')
    seg = nib.load(seg_path).get_fdata(dtype=np.float32)  # type: ignore

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    # Plot all 4 modalities
    for i, (img, mod) in enumerate(zip(imgs, modalities)):
        axes[i].imshow(img[:, :, slice_idx], cmap="gray")
        axes[i].set_title(mod)
        axes[i].axis('off')

    axes[4].imshow(seg[:, :, slice_idx], cmap="jet", vmin=0, vmax=3)
    axes[4].set_title("Segmentation")

    # add legend for segmentation colors
    legend_patches = [
        Patch(color='cyan', label='NCR/NET (1)'),
        Patch(color='yellow', label='ED (2)'),
        Patch(color='red', label='ET (3)'),
    ]
    fig.legend(handles=legend_patches, loc='lower center',
               bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=10)

    plt.suptitle(
        f"Display data of patient {patient_id.split("_")[-1]} of slice {slice_idx}")
    plt.tight_layout()
    plt.show()


def patient_information(path, patient_id, mod='seg', slice_idx=77, fixed_column_idx=120):
    """
    Print out data shape and one data column for fixed column and slice. 

    Args:
        path: Dictionary containing training history
        patient_id: ID of the patient to display (e.g., 'BraTS20_Training_001')
        mod: Modality to load (e.g., 'seg' for segmentation)
        slice_idx: Index of the slice to display
        fixed_column_idx: Fixed column index to display data for
    """
    patient_path = os.path.join(path, patient_id)

    file_path = os.path.join(patient_path, f'{patient_id}_{mod}.nii')
    file_data = nib.load(file_path).get_fdata(dtype=np.float32)

    print(f"Data shape: {file_data.shape}")
    print(
        f"Data for fixed column and slice: \n{file_data[:, fixed_column_idx, slice_idx]}")
