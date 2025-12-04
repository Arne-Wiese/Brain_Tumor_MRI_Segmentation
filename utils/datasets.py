"""
Dataset classes for BraTS 2020 brain tumor segmentation.
"""

import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


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
        self.patient_ids = []  # Store patient IDs for each sample
        self.slice_indices = []  # Store slice indices for each sample

        # For each patient, identify which slices contain tumor
        for patient_id in patient_list:
            # Only include slices within our range that have enough context slices
            for slice_idx in range(slice_range[0], slice_range[1]):
                self.samples.append((patient_id, slice_idx))
                self.patient_ids.append(patient_id)
                self.slice_indices.append(slice_idx)

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
            volume = nib.load(file_path).get_fdata()  # type: ignore

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
        seg_volume = nib.load(seg_path).get_fdata()  # type: ignore
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


class BraTSDataset_2D(Dataset):
    """
    Custom PyTorch Dataset for BraTS 2020 brain tumor segmentation.

    Loads 3D MRI volumes and extracts 2D slices for training.
    """

    def __init__(self, patient_list, data_dir, slice_range=(2, 153)):
        """
        Args:
            patient_list: List of patient folder names (e.g., ['BraTS20_Training_001', ...])
            data_dir: Path to the data directory
            slice_range: Tuple (min_slice, max_slice) to focus on brain region with tumors
        """
        self.data_dir = data_dir
        self.slice_range = slice_range
        self.samples = []  # Will store (patient_id, slice_index) pairs
        self.patient_ids = []  # Store patient IDs for each sample
        self.slice_indices = []  # Store slice indices for each sample

        # For each patient, identify which slices contain tumor
        for patient_id in patient_list:
            # Only include slices within our range that have enough context slices
            for slice_idx in range(slice_range[0], slice_range[1]):
                self.samples.append((patient_id, slice_idx))
                self.patient_ids.append(patient_id)
                self.slice_indices.append(slice_idx)

    def __len__(self):
        """Return total number of samples (2D slices)"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load and return one sample (a single 2D slice).

        Returns:
            image: Tensor of shape (4, 240, 240) - 4 MRI modalities
            mask: Tensor of shape (240, 240) - segmentation labels (center slice)
        """
        patient_id, slice_idx = self.samples[idx]
        patient_path = os.path.join(self.data_dir, patient_id)

        # Load all 4 modalities for this slice and neighboring slices
        modalities = ['flair', 't1', 't1ce', 't2']
        slices = []

        for modality in modalities:
            file_path = os.path.join(
                patient_path, f"{patient_id}_{modality}.nii")
            volume = nib.load(file_path).get_fdata()  # type: ignore

            # Collect slices
            modality_slices = []
            slice_2d = volume[:, :, slice_idx]
            slice_2d = self.normalize(slice_2d)
            slices.append(slice_2d)

        # Stack into (4, 240, 240) tensor
        image = np.stack(slices, axis=0)

        # Load segmentation mask (only the center slice)
        seg_path = os.path.join(patient_path, f"{patient_id}_seg.nii")
        seg_volume = nib.load(seg_path).get_fdata()  # type: ignore
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


MODALITIES = ["flair", "t1", "t1ce", "t2"]


def pad_to_multiple(tensor, multiple=16):
    if tensor.dim() == 3:
        d, h, w = tensor.shape
    elif tensor.dim() == 4:
        c, d, h, w = tensor.shape
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

    pad_d = (multiple - d % multiple) % multiple
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    # F.pad pads in order: (W_left, W_right, H_top, H_bottom, D_front, D_back)
    return F.pad(tensor, (0, pad_w, 0, pad_h, 0, pad_d))


def load_patient(patient_id, data_dir):
    """Load all modalities and segmentation for one patient"""
    patient_path = os.path.join(data_dir, patient_id)
    modalities = MODALITIES

    image_arrays = []
    for mod in modalities:
        file_path = os.path.join(patient_path, f"{patient_id}_{mod}.nii")
        img = nib.load(file_path).get_fdata(dtype=np.float32)  # type: ignore
        image_arrays.append(img)

    seg_path = os.path.join(patient_path, f"{patient_id}_seg.nii")
    seg = nib.load(seg_path).get_fdata(dtype=np.float32)  # type: ignore

    # Stack modalities -> shape (4, D, H, W)
    image = np.stack(image_arrays, axis=0)
    # removed line to get shape (D, H, W)
    # seg = np.expand_dims(seg, axis=0)  # shape (1, D, H, W)

    # remap label 4 to 3 to preserve continuity (there is no label 3 in the data)
    seg[seg == 4] = 3

    return image, seg


def cleanup_item(image, seg, transform=None):
    # Normalize each modality to [0, 1]
    image = (image - image.min(axis=(1, 2, 3), keepdims=True)) / (
        image.max(axis=(1, 2, 3), keepdims=True) -
        image.min(axis=(1, 2, 3), keepdims=True) + 1e-8
    )

    # Apply transform if any
    if transform:
        image, seg = transform(image, seg)

    # Convert to torch tensors
    image = torch.from_numpy(image).float()
    seg = torch.from_numpy(seg).long()

    # fix issues with height not being properly divisible
    image = pad_to_multiple(image)
    seg = pad_to_multiple(seg)

    return image, seg


class BraTSDataset3D(Dataset):
    """
    PyTorch Dataset for BraTS 2020 Brain Tumor Segmentation (2D U-Net)

    Design Decisions:
    1. Filter empty slices: Only keep slices with tumor pixels (labels 1,2,4)
    2. Label remapping: {0,1,2,4} → {0,1,2,3} for PyTorch compatibility
    3. Normalization: Percentile clipping + Z-score on BRAIN REGION ONLY
    4. Input: 4-channel (T1, T1ce, T2, FLAIR) x 240x240 full slices
    5. Patient-level split: Prevent data leakage
    """

    def __init__(self, data_dir, patient_ids=None, transform=None, in_memory=False):
        """
        Args:
            data_dir (str): Path or filesystem handle to training data (with patient folders)
            patient_ids (list[str], optional): Subset of patient folder names
            transform (callable, optional): Transform applied to each sample
            in_memory (bool): If True, preloads all patients into RAM
        """
        self.data_dir = data_dir
        self.transform = transform

        # Gather all patient folders
        self.patient_ids = patient_ids

        self.in_memory = in_memory
        if in_memory:
            print("Preloading all patient data into memory...")
            self.cache = {pid: self._load_patient(
                pid) for pid in self.patient_ids}
        else:
            self.cache = {}

    def _load_patient(self, patient_id):
        return load_patient(patient_id, self.data_dir)

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        if pid in self.cache:
            image, seg = self.cache[pid]
        else:
            image, seg = self._load_patient(pid)

        return cleanup_item(image, seg, self.transform)

    def _convert_labels(self, mask):
        """
        Remap labels from {0,1,2,4} to {0,1,2,3}

        Original BraTS labels:
            0 = Background
            1 = Necrotic/Non-enhancing tumor
            2 = Edema
            4 = Enhancing tumor

        Remapped labels (for PyTorch):
            0 = Background
            1 = Necrotic/Non-enhancing tumor
            2 = Edema
            3 = Enhancing tumor

        Args:
            mask (np.ndarray): Segmentation mask with original labels

        Returns:
            np.ndarray: Mask with remapped labels
        """
        mask = mask.copy()
        mask[mask == 4] = 3  # Remap enhancing tumor
        return mask
