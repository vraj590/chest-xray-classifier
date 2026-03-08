"""
dataset.py — Data loading, augmentation, and splitting for Chest X-Ray dataset.

Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
Expected folder structure after download:
    data/
        chest_xray/
            train/
                NORMAL/
                PNEUMONIA/
            val/
                NORMAL/
                PNEUMONIA/
            test/
                NORMAL/
                PNEUMONIA/
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─── Constants ────────────────────────────────────────────────────────────────
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]   # ImageNet stats (works well for X-rays too)
STD  = [0.229, 0.224, 0.225]

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


# ─── Augmentation pipelines ───────────────────────────────────────────────────
def get_train_transforms() -> A.Compose:
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


# ─── Dataset ──────────────────────────────────────────────────────────────────
class ChestXRayDataset(Dataset):
    """
    Loads chest X-ray images from a folder organised as:
        root/CLASS_NAME/image.jpeg
    """

    def __init__(self, root: str, transform: Optional[A.Compose] = None):
        self.root = Path(root)
        self.transform = transform
        self.samples: list[Tuple[Path, int]] = []

        for class_name, idx in CLASS_TO_IDX.items():
            class_dir = self.root / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Class folder not found: {class_dir}")
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                    self.samples.append((img_path, idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency weights to handle class imbalance."""
        counts = np.zeros(len(CLASS_NAMES))
        for _, label in self.samples:
            counts[label] += 1
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(CLASS_NAMES)   # normalise
        return torch.tensor(weights, dtype=torch.float32)


# ─── DataLoaders factory ──────────────────────────────────────────────────────
def build_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).
    """
    data_dir = Path(data_dir)

    train_ds = ChestXRayDataset(data_dir / "train", transform=get_train_transforms())
    val_ds   = ChestXRayDataset(data_dir / "val",   transform=get_val_transforms())
    test_ds  = ChestXRayDataset(data_dir / "test",  transform=get_val_transforms())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"  Train: {len(train_ds):,} images")
    print(f"  Val  : {len(val_ds):,} images")
    print(f"  Test : {len(test_ds):,} images")
    print(f"  Class weights: {train_ds.get_class_weights().tolist()}")

    return train_loader, val_loader, test_loader
