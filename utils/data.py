"""
Helper data functions for BraTS 2020 brain tumor segmentation.
"""

from sklearn.model_selection import train_test_split
import os
import nibabel as nib
import numpy as np
from tqdm import tqdm

from utils import hpc
if not hpc.running_on_hpc():
    import kagglehub


def load_dataset():
    if hpc.running_on_hpc():
        return hpc.load_dataset_into_ram()
    else:
        # Download latest version
        path = kagglehub.dataset_download(
            "awsaf49/brats20-dataset-training-validation")
        print("Path to dataset files:", path)
        return path + '/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'


def load_patients(path):
    return sorted([d for d in os.listdir(path)
                   if os.path.isdir(os.path.join(path, d))])


def split_patients(patients, split1=0.3, split2=0.5):
    # First split: 70% train, 30% temp
    train_patients, temp_patients = train_test_split(
        patients, test_size=split1, random_state=42
    )

    # Second split: split temp into 50/50 (15% val, 15% test of original)
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=split2, random_state=42
    )

    return train_patients, val_patients, test_patients


def load_seg_data(data_path, patient_folder, slice_range):
    patient_path = os.path.join(data_path, patient_folder)

    # Load ground truth segmentation mask
    seg_path = os.path.join(patient_path, f"{patient_folder}_seg.nii")
    seg_volume = nib.load(seg_path).get_fdata()  # type: ignore
    segmentation = seg_volume[:, :, slice_range[0]: slice_range[1]]

    return segmentation


def count_tumor_slices(segmentation):
    tumor_slices = []

    # Loop through all 155 slices
    for i in range(segmentation.shape[2]):
        slice_2d = segmentation[:, :, i]

        # Check if slice has any tumor labels (1, 2, or 4)
        if np.any(np.isin(slice_2d, [1, 2, 4])):
            tumor_slices.append(i)

    return tumor_slices


def data_statistics(data_path, all_patients, slice_range):
    # Initialize counters
    total_slices = 0
    total_tumor_slices = 0
    class_counts = {0: 0, 1: 0, 2: 0, 4: 0}
    class_names = {0: 'Background', 1: 'Necrotic', 2: 'Edema', 4: 'Enhancing'}

    # Loop through all patients with progress bar
    for patient in tqdm(all_patients, desc="Processing"):
        seg = load_seg_data(data_path, patient, slice_range)

        # Count slices
        total_slices += seg.shape[2]
        tumor_slices_list = count_tumor_slices(seg)
        total_tumor_slices += len(tumor_slices_list)

        # Accumulate pixel counts per class
        unique, counts = np.unique(seg, return_counts=True)
        for label, count in zip(unique, counts):
            if int(label) in class_counts:
                class_counts[int(label)] += int(count)

    # Print final statistics
    print(f"\nDATASET-WIDE STATISTICS ({len(all_patients)} patients):")
    print(f"   {'─'*60}")
    print(f"   Total slices:              {total_slices:>10,}")
    print(
        f"   Slices WITH tumor:         {total_tumor_slices:>10,}  ({total_tumor_slices/total_slices*100:>5.1f}%)")
    print(
        f"   Slices WITHOUT tumor:      {total_slices-total_tumor_slices:>10,}  ({(total_slices-total_tumor_slices)/total_slices*100:>5.1f}%)")
    print(f"\n   Overall Class Distribution:")
    print(f"   {'─'*60}")

    total_pixels = sum(class_counts.values())
    for label in [0, 1, 2, 4]:
        count = class_counts[label]
        pct = (count / total_pixels) * 100
        print(
            f"   Class {label} ({class_names[label]:<15}): {count:>15,} pixels  ({pct:>5.2f}%)")
