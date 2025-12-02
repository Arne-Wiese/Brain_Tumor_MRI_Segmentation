"""
Helper data functions for BraTS 2020 brain tumor segmentation.
"""

from sklearn.model_selection import train_test_split
import os

import hpc
if not hpc.running_on_hpc():
    import kagglehub

def load_dataset():
    if hpc.running_on_hpc():
        return hpc.load_dataset_into_ram()
    else:
        # Download latest version
        path = kagglehub.dataset_download("awsaf49/brats20-dataset-training-validation")
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