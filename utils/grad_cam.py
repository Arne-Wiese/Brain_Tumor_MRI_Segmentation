import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.metrics import dice_score


class GradCAM:
    """Grad-CAM implementation for CNNs."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(
            self._forward_hook)
        self.backward_hook = target_layer.register_full_backward_hook(
            self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input tensor (1, C, H, W)
            target_class: Target class for CAM (if None, uses predicted class)

        Returns:
            cam: Grad-CAM heatmap
        """
        # Forward pass
        output = self.model(input_tensor)

        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1)
        else:
            # Convert to tensor if it's an integer
            if isinstance(target_class, int):
                target_class = torch.tensor(
                    [target_class], device=output.device)
            elif not isinstance(target_class, torch.Tensor):
                target_class = torch.tensor(target_class, device=output.device)

        # Zero gradients
        self.model.zero_grad()

        # Create one-hot encoded target
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, target_class.unsqueeze(
            1).unsqueeze(2).unsqueeze(3).expand_as(output), 1.0)

        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)

        # Get gradients and activations
        gradients = self.gradients  # (B, C, H, W)
        activations = self.activations  # (B, C, H, W)

        # Global average pooling on gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        # Weighted sum of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)

        # Apply ReLU
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.squeeze().cpu().numpy()

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def visualize_best_worst_gradcam(model, dataloader, device, layer_name, save_path_best, save_path_worst, target_class=None, metric_fn=dice_score, n_samples=8, input_channel=0, threshold=1.0):
    """
    Visualize Grad-CAM for best and worst predictions based on a metric.

    Args:
        model: Trained model
        dataloader: Test dataloader
        device: Device to use
        layer_name: Name of the layer to visualize
        target_class: Target class for Grad-CAM (None=use prediction, 1=necrotic, 2=edema, 3=enhancing)
        metric_fn: Metric function (e.g., dice_score)
        n_samples: Number of best/worst samples to show
        input_channel: Which input modality to display (0=T1, 1=T1ce, 2=T2, 3=FLAIR)
    """
    model.eval()

    # Modality labels
    modality_labels = {
        0: 'T1',
        1: 'T1ce',
        2: 'T2',
        3: 'FLAIR'
    }

    # Class labels
    class_labels = {
        None: 'Predicted',
        0: 'Background',
        1: 'Necrotic',
        2: 'Edema',
        3: 'Enhancing'
    }

    # Get layer by name
    def get_layer_by_name(model, layer_name):
        parts = layer_name.split('.')
        module = model
        for part in parts:
            module = getattr(module, part)
        return module

    # Get target layer
    target_layer = get_layer_by_name(model, layer_name)

    # Get dataset from dataloader
    dataset = dataloader.dataset
    if hasattr(dataset, 'dataset'):
        base_dataset = dataset.dataset
        indices = dataset.indices
    else:
        base_dataset = dataset
        indices = range(len(dataset))

    # Store results
    results = []

    print("Generating Grad-CAMs...")
    with torch.no_grad():
        batch_idx = 0
        for images, masks in tqdm(dataloader):
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass for predictions
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            # Calculate metric for each sample
            for i in range(images.size(0)):
                pred = outputs[i].unsqueeze(0)
                mask = masks[i].unsqueeze(0)

                # Calculate metric
                metric_value = metric_fn(pred, mask)

                # Calculate sample index
                sample_idx = batch_idx * dataloader.batch_size + i

                # Get patient info
                if sample_idx < len(indices):
                    actual_idx = indices[sample_idx]
                    patient_id = base_dataset.patient_ids[actual_idx]
                    slice_idx = base_dataset.slice_indices[actual_idx]
                else:
                    patient_id = "Unknown"
                    slice_idx = 0

                results.append({
                    'metric': metric_value,
                    'image': images[i].cpu().numpy(),
                    'mask': masks[i].cpu().numpy(),
                    'prediction': predictions[i].cpu().numpy(),
                    'patient_id': patient_id,
                    'slice_idx': slice_idx,
                    'image_tensor': images[i].unsqueeze(0)  # Keep for Grad-CAM
                })

            batch_idx += 1

    # Sort by metric
    results.sort(key=lambda x: x['metric'], reverse=True)

    # Get best and worst samples - ensure different patients
    def _get_diverse_samples(results, n_samples, reverse=False):
        """Get n_samples from different patients."""
        selected = []
        seen_patients = set()

        sample_list = results[::-1] if reverse else results

        for sample in sample_list:
            patient_id = sample['patient_id']
            if patient_id not in seen_patients:
                selected.append(sample)
                seen_patients.add(patient_id)
                if len(selected) >= n_samples:
                    break

        return selected

    best_candidates = [x for x in results if x['metric'] < threshold]
    best_samples = _get_diverse_samples(
        best_candidates, n_samples, reverse=False)
    worst_samples = _get_diverse_samples(results, n_samples, reverse=True)

    # Generate Grad-CAMs for selected samples
    print("Computing Grad-CAMs for selected samples...")
    grad_cam = GradCAM(model, target_layer)

    for samples in [best_samples, worst_samples]:
        for sample in tqdm(samples):
            model.train()  # Need gradients
            input_tensor = sample['image_tensor'].to(device)
            input_tensor.requires_grad = True

            # Generate Grad-CAM
            cam = grad_cam.generate_cam(
                input_tensor, target_class=target_class)
            sample['gradcam'] = cam

            model.eval()

    grad_cam.remove_hooks()

    # Create legend patches
    legend_patches = [
        mpatches.Patch(facecolor='cyan', label='NCR/NET (1)'),
        mpatches.Patch(facecolor='yellow', label='Edema (2)'),
        mpatches.Patch(facecolor='red', label='Enhancing (3)')
    ]

    # Visualization title
    cam_title = f"Grad-CAM ({class_labels[target_class]})"

    # Visualize best samples
    fig, axes = plt.subplots(4, n_samples, figsize=(22, 11))
    fig.suptitle(f'Best {n_samples} Predictions - Grad-CAM - Layer: {layer_name} - Input: {modality_labels[input_channel]}',
                 fontsize=16, fontweight='bold', y=0.98)

    for i, sample in enumerate(best_samples):
        # Row 0: Input image
        axes[0, i].imshow(sample['image'][input_channel], cmap='gray')
        axes[0, i].set_title(
            f'Dice: {sample["metric"]:.3f}\n{sample["patient_id"]}\nSlice: {sample["slice_idx"]}', fontsize=9)
        if i == 0:
            axes[0, i].set_ylabel(f'Input\n({modality_labels[input_channel]})',
                                  fontsize=12, rotation=0, ha='right', va='center', labelpad=20)
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            for spine in axes[0, i].spines.values():
                spine.set_visible(False)
        else:
            axes[0, i].axis('off')

        # Row 1: Grad-CAM heatmap
        cam_resized = F.interpolate(
            torch.tensor(sample['gradcam']).unsqueeze(0).unsqueeze(0),
            size=sample['image'][input_channel].shape,
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()

        axes[1, i].imshow(sample['image'][input_channel], cmap='gray')
        axes[1, i].imshow(cam_resized, cmap='jet', alpha=0.3)
        if i == 0:
            axes[1, i].set_ylabel(
                cam_title, fontsize=12, rotation=0, ha='right', va='center', labelpad=20)
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])
            for spine in axes[1, i].spines.values():
                spine.set_visible(False)
        else:
            axes[1, i].axis('off')

        # Row 2: Ground Truth
        axes[2, i].imshow(sample['mask'], cmap='jet', vmin=0, vmax=3)
        if i == 0:
            axes[2, i].set_ylabel(
                'Ground Truth', fontsize=12, rotation=0, ha='right', va='center', labelpad=20)
            axes[2, i].set_xticks([])
            axes[2, i].set_yticks([])
            for spine in axes[2, i].spines.values():
                spine.set_visible(False)
        else:
            axes[2, i].axis('off')

        # Row 3: Prediction
        axes[3, i].imshow(sample['prediction'], cmap='jet', vmin=0, vmax=3)
        if i == 0:
            axes[3, i].set_ylabel(
                'Prediction', fontsize=12, rotation=0, ha='right', va='center', labelpad=20)
            axes[3, i].set_xticks([])
            axes[3, i].set_yticks([])
            for spine in axes[3, i].spines.values():
                spine.set_visible(False)
        else:
            axes[3, i].axis('off')

    fig.legend(handles=legend_patches, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.03), fontsize=10, frameon=True)

    plt.tight_layout()
    plt.savefig(save_path_best, dpi=150, bbox_inches='tight')
    plt.show()

    # Visualize worst samples
    fig, axes = plt.subplots(4, n_samples, figsize=(22, 11))
    fig.suptitle(f'Worst {n_samples} Predictions - Grad-CAM - Layer: {layer_name} - Input: {modality_labels[input_channel]}',
                 fontsize=16, fontweight='bold', y=0.98)

    for i, sample in enumerate(worst_samples):
        # Row 0: Input image
        axes[0, i].imshow(sample['image'][input_channel], cmap='gray')
        axes[0, i].set_title(
            f'Dice: {sample["metric"]:.3f}\n{sample["patient_id"]}\nSlice: {sample["slice_idx"]}', fontsize=9)
        if i == 0:
            axes[0, i].set_ylabel(f'Input\n({modality_labels[input_channel]})',
                                  fontsize=12, rotation=0, ha='right', va='center', labelpad=20)
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            for spine in axes[0, i].spines.values():
                spine.set_visible(False)
        else:
            axes[0, i].axis('off')

        # Row 1: Grad-CAM heatmap
        cam_resized = F.interpolate(
            torch.tensor(sample['gradcam']).unsqueeze(0).unsqueeze(0),
            size=sample['image'][input_channel].shape,
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()

        axes[1, i].imshow(sample['image'][input_channel], cmap='gray')
        axes[1, i].imshow(cam_resized, cmap='jet', alpha=0.5)
        if i == 0:
            axes[1, i].set_ylabel(
                cam_title, fontsize=12, rotation=0, ha='right', va='center', labelpad=20)
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])
            for spine in axes[1, i].spines.values():
                spine.set_visible(False)
        else:
            axes[1, i].axis('off')

        # Row 2: Ground Truth
        axes[2, i].imshow(sample['mask'], cmap='jet', vmin=0, vmax=3)
        if i == 0:
            axes[2, i].set_ylabel(
                'Ground Truth', fontsize=12, rotation=0, ha='right', va='center', labelpad=20)
            axes[2, i].set_xticks([])
            axes[2, i].set_yticks([])
            for spine in axes[2, i].spines.values():
                spine.set_visible(False)
        else:
            axes[2, i].axis('off')

        # Row 3: Prediction
        axes[3, i].imshow(sample['prediction'], cmap='jet', vmin=0, vmax=3)
        if i == 0:
            axes[3, i].set_ylabel(
                'Prediction', fontsize=12, rotation=0, ha='right', va='center', labelpad=20)
            axes[3, i].set_xticks([])
            axes[3, i].set_yticks([])
            for spine in axes[3, i].spines.values():
                spine.set_visible(False)
        else:
            axes[3, i].axis('off')

    fig.legend(handles=legend_patches, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.03), fontsize=10, frameon=True)

    plt.tight_layout()
    plt.savefig(save_path_worst, dpi=150, bbox_inches='tight')
    plt.show()

    return best_samples, worst_samples
