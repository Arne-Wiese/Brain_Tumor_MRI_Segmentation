"""
Dataset classes for BraTS 2020 brain tumor segmentation.
"""

import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset


class BraTSDataset_25D(Dataset):
    """
    Custom PyTorch Dataset for BraTS 2020 brain tumor segmentation.

    Loads 3D MRI volumes and extracts 2.5D slices for training by aggregating multiple consecutive slices.
    """

    def __init__(self, patient_list, data_dir, slice_range=(2, 153), n_slices=5):
        """
        Args:
            patient_list: List of patient folder names (e.g., ['BraTS20_Training_001', ...])
            data_dir: Path to the data directory
            slice_range: Tuple (min_slice, max_slice) to focus on brain region with tumors
            n_slices: Number of consecutive slices to aggregate (must be odd for symmetry)
        """
        self.data_dir = data_dir
        self.slice_range = slice_range
        self.n_slices = n_slices
        self.half_slices = n_slices // 2
        self.samples = []  # Will store (patient_id, slice_index) pairs

        # For each patient, identify which slices contain tumor
        for patient_id in patient_list:
            # Only include slices within our range that have enough context slices
            for slice_idx in range(slice_range[0], slice_range[1]):
                self.samples.append((patient_id, slice_idx))

    def __len__(self):
        """Return total number of samples (2D slices)"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load and return one sample (a single 2D slice with aggregated context from n slices).

        Returns:
            image: Tensor of shape (4, 240, 240) - 4 MRI modalities with aggregated slices
            mask: Tensor of shape (240, 240) - segmentation labels (center slice)
        """
        patient_id, slice_idx = self.samples[idx]
        patient_path = os.path.join(self.data_dir, patient_id)

        # Load all 4 modalities for this slice and neighboring slices
        modalities = ['flair', 't1', 't1ce', 't2']
        aggregated_slices = []

        for modality in modalities:
            file_path = os.path.join(
                patient_path, f"{patient_id}_{modality}.nii")
            volume = nib.load(file_path).get_fdata()

            # Collect n_slices around the center slice
            slices_to_aggregate = []
            for offset in range(-self.half_slices, self.half_slices + 1):
                idx_to_load = slice_idx + offset
                slice_2d = volume[:, :, idx_to_load]
                slice_2d = self.normalize(slice_2d)
                slices_to_aggregate.append(slice_2d)

            # Aggregate by taking the mean across slices
            aggregated_slice = np.mean(slices_to_aggregate, axis=0)
            aggregated_slices.append(aggregated_slice)

        # Stack into (4, 240, 240) tensor
        image = np.stack(aggregated_slices, axis=0)

        # Load segmentation mask (only the center slice)
        seg_path = os.path.join(patient_path, f"{patient_id}_seg.nii")
        seg_volume = nib.load(seg_path).get_fdata()
        mask = seg_volume[:, :, slice_idx]

        # Convert labels: 0->0, 1->1, 2->2, 4->3 (to have consecutive labels)
        mask[mask == 4] = 3

        # Convert to PyTorch tensors
        image = torch.FloatTensor(image)
        mask = torch.LongTensor(mask.astype(np.int64))

        return image, mask

    def normalize(self, slice_2d):
        """Normalize a 2D slice to [0, 1] range"""
        min_val = slice_2d.min()
        max_val = slice_2d.max()

        if max_val - min_val > 0:
            return (slice_2d - min_val) / (max_val - min_val)
        else:
            return slice_2d
