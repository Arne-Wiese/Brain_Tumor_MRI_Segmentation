"""
Generic training loop for brain tumor segmentation.
"""

import torch
from tqdm import tqdm
import numpy as np


def train_one_epoch(model, dataloader, criterion, optimizer, device, metric_fn=None):
    """
    Train the model for one epoch.

    Args:
        model: PyTorch model
        dataloader: Training DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cpu/cuda/mps)
        metric_fn: Optional metric function that takes (predictions, targets)

    Returns:
        epoch_loss: Average loss for the epoch
        epoch_metric: Average metric for the epoch (if metric_fn provided)
    """
    model.train()
    running_loss = 0.0
    running_metric = 0.0

    pbar = tqdm(dataloader, desc="Training", leave=False)
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
        if metric_fn is not None:
            metric_value = metric_fn(outputs, masks)
            running_metric += metric_value

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(dataloader)
    epoch_metric = running_metric / \
        len(dataloader) if metric_fn is not None else None

    return epoch_loss, epoch_metric


def validate(model, dataloader, criterion, device, metric_fn=None):
    """
    Validate the model.

    Args:
        model: PyTorch model
        dataloader: Validation DataLoader
        criterion: Loss function
        device: Device to run on (cpu/cuda/mps)
        metric_fn: Optional metric function that takes (predictions, targets)

    Returns:
        epoch_loss: Average loss for the epoch
        epoch_metric: Average metric for the epoch (if metric_fn provided)
    """
    model.eval()
    running_loss = 0.0
    running_metric = 0.0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Metrics
            running_loss += loss.item()
            if metric_fn is not None:
                metric_value = metric_fn(outputs, masks)
                running_metric += metric_value

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(dataloader)
    epoch_metric = running_metric / \
        len(dataloader) if metric_fn is not None else None

    return epoch_loss, epoch_metric


def train_loop(model, train_loader, val_loader, criterion, optimizer,
               num_epochs, device, scheduler=None, metric_fn=None,
               save_best_model=True, model_save_path='best_model.pth',
               early_stopping_patience=None, verbose=True):
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
        metric_fn: Optional metric function for evaluation
        save_best_model: Whether to save the best model based on validation metric
        model_save_path: Path to save the best model
        early_stopping_patience: Stop training if validation doesn't improve for N epochs
        verbose: Whether to print training progress

    Returns:
        history: Dictionary containing training history
            - 'train_loss': List of training losses
            - 'train_metric': List of training metrics
            - 'val_loss': List of validation losses
            - 'val_metric': List of validation metrics
    """
    history = {
        'train_loss': [],
        'train_metric': [],
        'val_loss': [],
        'val_metric': []
    }

    best_val_metric = 0.0 if metric_fn is not None else float('inf')
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        if verbose:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)

        # Training
        train_loss, train_metric = train_one_epoch(
            model, train_loader, criterion, optimizer, device, metric_fn
        )

        if verbose:
            if train_metric is not None:
                print(
                    f"Train Loss: {train_loss:.4f}, Train Metric: {train_metric:.4f}")
            else:
                print(f"Train Loss: {train_loss:.4f}")

        # Validation
        val_loss, val_metric = validate(
            model, val_loader, criterion, device, metric_fn
        )

        if verbose:
            if val_metric is not None:
                print(
                    f"Val Loss: {val_loss:.4f}, Val Metric: {val_metric:.4f}")
            else:
                print(f"Val Loss: {val_loss:.4f}")

        # Store metrics
        history['train_loss'].append(train_loss)
        history['train_metric'].append(train_metric)
        history['val_loss'].append(val_loss)
        history['val_metric'].append(val_metric)

        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Save best model
        if save_best_model:
            # Use metric if available, otherwise use loss
            if metric_fn is not None:
                improved = val_metric > best_val_metric
                if improved:
                    best_val_metric = val_metric
            else:
                improved = val_loss < best_val_loss
                if improved:
                    best_val_loss = val_loss

            if improved:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metric': val_metric,
                }, model_save_path)

                if verbose:
                    if metric_fn is not None:
                        print(
                            f"✓ Saved new best model with Val Metric: {val_metric:.4f}")
                    else:
                        print(
                            f"✓ Saved new best model with Val Loss: {val_loss:.4f}")

                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        # Early stopping
        if early_stopping_patience is not None:
            if epochs_without_improvement >= early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    if verbose:
        print("\nTraining completed!")
        if metric_fn is not None:
            print(f"Best validation metric: {best_val_metric:.4f}")
        print(f"Best validation loss: {best_val_loss:.4f}")

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
        epoch: Epoch number from checkpoint
        val_metric: Validation metric from checkpoint
    """
    # Set weights_only=False to load full checkpoint with optimizer state and metrics
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    val_metric = checkpoint.get('val_metric', None)

    return epoch, val_metric
