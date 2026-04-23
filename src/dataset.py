"""
dataset.py — PyTorch Dataset for 6-channel RGB+Multispectral fusion input.

Constructs a 6-channel tensor: [R, G, B, NIR, NDVI, GNDVI]
- RGB channels from field_images/rgb/
- NIR channel from field_images/nir/
- NDVI = (NIR - R) / (NIR + R + ε)
- GNDVI = (NIR - G) / (NIR + G + ε)
"""

import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from src.config import RGB_DIR, NIR_DIR, IMAGE_SIZE

import os


# ============================================================
# Custom Augmentations for multi-channel tensors
# ============================================================

class RandomHorizontalFlip:
    """Randomly flip the tensor horizontally with probability p."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1).item() < self.p:
            return torch.flip(tensor, dims=[-1])
        return tensor


class RandomVerticalFlip:
    """Randomly flip the tensor vertically with probability p."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1).item() < self.p:
            return torch.flip(tensor, dims=[-2])
        return tensor


class RandomRotation:
    """Randomly rotate the tensor by an angle in [-degrees, +degrees]."""
    def __init__(self, degrees=15):
        self.degrees = degrees

    def __call__(self, tensor):
        angle = float(torch.randint(-self.degrees, self.degrees + 1, (1,)).item())
        return TF.rotate(tensor, angle)


# ============================================================
# Dataset
# ============================================================

class AgriVisionDataset(Dataset):
    """
    Agriculture Vision dataset for stress classification.

    Each sample returns:
        - tensor: (6, IMAGE_SIZE, IMAGE_SIZE) float32 — [R, G, B, NIR, NDVI, GNDVI]
        - label:  scalar long tensor — 0=Healthy, 1=Early Stress, 2=Severe Stress
    """

    def __init__(self, manifest_df: pd.DataFrame, transform=None):
        """
        Args:
            manifest_df: DataFrame with columns [image_id, split, label, ...]
            transform: Optional transform (augmentation) applied to the 6-ch tensor
        """
        self.data = manifest_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = row["image_id"]
        label = int(row["label"])

        # ---- Load RGB image (3 channels) ----
        rgb_path = os.path.join(RGB_DIR, f"{image_id}.jpg")
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            raise FileNotFoundError(f"RGB image not found: {rgb_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))

        # ---- Load NIR image (1 channel) ----
        nir_path = os.path.join(NIR_DIR, f"{image_id}.jpg")
        nir = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
        if nir is None:
            raise FileNotFoundError(f"NIR image not found: {nir_path}")
        nir = cv2.resize(nir, (IMAGE_SIZE, IMAGE_SIZE))

        # ---- Normalize to [0, 1] ----
        rgb = rgb.astype(np.float32) / 255.0
        nir = nir.astype(np.float32) / 255.0

        # ---- Extract individual channels ----
        r = rgb[:, :, 0]
        g = rgb[:, :, 1]
        b = rgb[:, :, 2]

        # ---- Compute Vegetation Indices ----
        eps = 1e-7

        # NDVI = (NIR - Red) / (NIR + Red + ε)
        ndvi = (nir - r) / (nir + r + eps)
        ndvi = np.clip(ndvi, -1.0, 1.0)

        # GNDVI = (NIR - Green) / (NIR + Green + ε)
        gndvi = (nir - g) / (nir + g + eps)
        gndvi = np.clip(gndvi, -1.0, 1.0)

        # ---- Stack into 6-channel tensor [R, G, B, NIR, NDVI, GNDVI] ----
        tensor = np.stack([r, g, b, nir, ndvi, gndvi], axis=0)
        tensor = torch.from_numpy(tensor).float()

        # ---- Apply augmentations ----
        if self.transform:
            tensor = self.transform(tensor)

        label_tensor = torch.tensor(label, dtype=torch.long)

        return tensor, label_tensor


def get_train_transforms():
    """Return the augmentation pipeline for training."""
    from torchvision import transforms
    return transforms.Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation(degrees=15),
    ])


def get_val_transforms():
    """Return transforms for validation/test (no augmentation)."""
    return None
