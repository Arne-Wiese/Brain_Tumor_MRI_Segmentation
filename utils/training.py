"""
Generic training loop for brain tumor segmentation.
"""

import torch
from tqdm import tqdm
import numpy as np
import time
from datetime import datetime
import json
import os
import utils

from utils.metrics import (
    dice_score,
    dice_score_background,
    dice_score_necrotic,
    dice_score_edema,
    dice_score_enhancing,
    iou_score,
    iou_score_background,
    iou_score_necrotic,
    iou_score_edema,
    iou_score_enhancing
)

metric_fns = {
    'dice': dice_score,
    'dice_background': dice_score_background,
    'dice_necrotic': dice_score_necrotic,
    'dice_edema': dice_score_edema,
    'dice_enhancing': dice_score_enhancing,
    'iou': iou_score,
    'iou_background': iou_score_background,
    'iou_necrotic': iou_score_necrotic,
    'iou_edema': iou_score_edema,
    'iou_enhancing': iou_score_enhancing
}

def train_one_epoch(model, dataloader, criterion, optimizer, device, metric_fns=None):
    """
    Train the model for one epoch.

    Args:
        model: PyTorch model
        dataloader: Training DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cpu/cuda/mps)
        metric_fns: Optional dict of metric functions {name: function} that take (predictions, targets)

    Returns:
        epoch_loss: Average loss for the epoch
        epoch_metrics: Dict of average metrics for the epoch (if metric_fns provided)
    """
    model.train()
    running_loss = 0.0
    running_metrics = {name: 0.0 for name in metric_fns.keys()} if metric_fns else {
    }

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        running_loss += loss.item()
        if metric_fns is not None:
            for name, metric_fn in metric_fns.items():
                metric_value = metric_fn(outputs, masks)
                running_metrics[name] += metric_value

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(dataloader)
    epoch_metrics = {name: value / len(dataloader) for name,
                     value in running_metrics.items()} if metric_fns else None

    return epoch_loss, epoch_metrics


def validate(model, dataloader, criterion, device, metric_fns=None):
    """
    Validate the model.

    Args:
        model: PyTorch model
        dataloader: Validation DataLoader
        criterion: Loss function
        device: Device to run on (cpu/cuda/mps)
        metric_fns: Optional dict of metric functions {name: function} that take (predictions, targets)

    Returns:
        epoch_loss: Average loss for the epoch
        epoch_metrics: Dict of average metrics for the epoch (if metric_fns provided)
    """
    model.eval()
    running_loss = 0.0
    running_metrics = {name: 0.0 for name in metric_fns.keys()} if metric_fns else {
    }

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Metrics
            running_loss += loss.item()
            if metric_fns is not None:
                for name, metric_fn in metric_fns.items():
                    metric_value = metric_fn(outputs, masks)
                    running_metrics[name] += metric_value

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(dataloader)
    epoch_metrics = {name: value / len(dataloader) for name,
                     value in running_metrics.items()} if metric_fns else None

    return epoch_loss, epoch_metrics

def save_history(history, save_dir, today=None):
    if today is None:
        today = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_file = f'{save_dir}/{today}_training_history.json'

    os.makedirs(save_dir, exist_ok=True)

    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4)

def train_loop(model, train_loader, val_loader, criterion, optimizer,
               num_epochs, device, scheduler=None,
               save_best_model=True, model_save_path='best_model.pth',
               early_stopping_patience=None, primary_metric=None, save_history_fn=lambda x: None):
    """
    Generic training loop with validation.

    Args:
        model: PyTorch model to train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        device: Device to train on (cpu/cuda/mps)
        scheduler: Optional learning rate scheduler
        metric_fns: Optional dict of metric functions {name: function} for evaluation
        save_best_model: Whether to save the best model based on validation metric
        model_save_path: Path to save the best model
        early_stopping_patience: Stop training if validation doesn't improve for N epochs
        primary_metric: Name of the primary metric to use for model selection (if None, uses first metric or loss)

    Returns:
        history: Dictionary containing training history
            - 'train_loss': List of training losses
            - 'train_{metric_name}': List of training metrics for each metric
            - 'val_loss': List of validation losses
            - 'val_{metric_name}': List of validation metrics for each metric
    """
    # Initialize history with loss and all metrics
    history = {'train_loss': [], 'val_loss': [],
            'epoch_train_times': [], 'learning_rates': []}
    if metric_fns is not None:
        for name in metric_fns.keys():
            history[f'train_{name}'] = []
            history[f'val_{name}'] = []

    try:

        # Determine primary metric for model selection
        if metric_fns is not None:
            if primary_metric is None:
                primary_metric = list(metric_fns.keys())[0]
            best_val_metric = 0.0 if metric_fns else float('inf')
        else:
            primary_metric = None
            best_val_metric = float('inf')

        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)

            print(f"Epoch {epoch + 1}/{num_epochs}")

            # start time measurement
            epoch_start_time = time.time()
            # Training
            train_loss, train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device, metric_fns
            )

            # Validation
            val_loss, val_metrics = validate(
                model, val_loader, criterion, device, metric_fns
            )

            epoch_time = time.time() - epoch_start_time
            history['epoch_train_times'].append(epoch_time)

            # Store metrics
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            if train_metrics is not None:
                for name, value in train_metrics.items():
                    history[f'train_{name}'].append(value)
            if val_metrics is not None:
                for name, value in val_metrics.items():
                    history[f'val_{name}'].append(value)

            print(f"  Epoch Time: {epoch_time:.2f}s")

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Epoch time: {epoch_time:.2f}s")
            print(f"Train Dice: {train_metrics['dice']:.4f} | Val Dice: {val_metrics['dice']:.4f}")
            print(f"Train IoU: {train_metrics['iou']:.4f} | Val IoU: {val_metrics['iou']:.4f}")


            # Learning rate scheduling
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Save best model
            if save_best_model:
                # Use primary metric if available, otherwise use loss
                if primary_metric is not None and val_metrics is not None:
                    current_val_metric = val_metrics[primary_metric]
                    improved = current_val_metric > best_val_metric
                    if improved:
                        best_val_metric = current_val_metric
                else:
                    improved = val_loss < best_val_loss
                    current_val_metric = None

                if improved:
                    best_val_loss = val_loss

                if improved:
                    # Save checkpoint with all metrics
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'learning_rate': current_lr,
                    }
                    if val_metrics is not None:
                        for name, value in val_metrics.items():
                            checkpoint[f'val_{name}'] = value

                    torch.save(checkpoint, model_save_path)

                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            # Early stopping
            if early_stopping_patience is not None:
                if epochs_without_improvement >= early_stopping_patience:
                    break

            save_history_fn(history)
    finally:
        save_history_fn(history)

    return history


def load_checkpoint(model, checkpoint_path, optimizer=None, device='cpu'):
    """
    Load a model checkpoint.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to the checkpoint file
        optimizer: Optional optimizer to load state into
        device: Device to load the model on

    Returns:
        checkpoint: Dictionary containing all checkpoint data
            - 'epoch': Epoch number from checkpoint
            - 'val_metric': Validation metric from checkpoint
            - 'val_loss': Validation loss from checkpoint
            - 'model_state_dict': Model state dictionary
            - 'optimizer_state_dict': Optimizer state dictionary (if present)
    """
    # Set weights_only=False to load full checkpoint with optimizer state and metrics
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint
