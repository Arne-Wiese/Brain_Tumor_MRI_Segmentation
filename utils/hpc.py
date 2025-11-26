import zipfile
import os

def running_on_hpc():
    return os.path.isfile("/user/gent/451/vsc45110/data/dl/Brain_Tumor_MRI_Segmentation/data/brats20-dataset-training-validation.zip")

def load_dataset_into_ram():
    zip_path = "/user/gent/451/vsc45110/data/dl/Brain_Tumor_MRI_Segmentation/data/brats20-dataset-training-validation.zip"
    ram_dir = "/dev/shm/brats_data"

    os.makedirs(ram_dir, exist_ok=True)

    DATA_DIR = os.path.join(ram_dir, "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")

    if not os.path.isdir(DATA_DIR):
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(ram_dir)

    return DATA_DIR