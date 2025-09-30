# %% [code]
# pcam_data_utils.py

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

print("Utility script 'pcam_data_utils.py' loaded.")

# --- PyTorch Dataset Class ---
class PCamDataset(Dataset):
    """Custom PyTorch Dataset for the PatchCamelyon dataset."""
    def __init__(self, dataframe, image_dir, transform=None):
        self.labels_df = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_id = self.labels_df.iloc[idx, 0]
        img_path = os.path.join(self.image_dir, f"{img_id}.tif")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.labels_df.iloc[idx, 1]
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label.unsqueeze(0)

# --- Augmentation Pipelines ---
def get_transforms():
    """Returns a dictionary of augmentation pipelines for training and validation."""
    train_transform = A.Compose([
        A.RandomRotate90(p=0.5),
        
        # --- FIXED CODE ---
        # The function 'A.Flip' is from a newer version of albumentations.
        # We use the explicit functions available in the Kaggle environment.
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # --- END OF FIX ---

        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    return {'train': train_transform, 'val': val_transform}