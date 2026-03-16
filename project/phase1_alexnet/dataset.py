"""
PyTorch Dataset and DataLoader for landslide image classification.
Uses albumentations for augmentation and OpenCV for image reading.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class LandslideDataset(Dataset):
    """
    Dataset that loads images from:
        processed/<split>/landslide/
        processed/<split>/non_landslide/

    Labels: landslide = 1, non_landslide = 0
    """

    def __init__(self, root_dir: str, split: str, transform=None):
        """
        Args:
            root_dir:  Path to data/processed/
            split:     'train' or 'test'
            transform: albumentations Compose transform
        """
        assert split in ("train", "val", "test"), "split must be 'train', 'val', or 'test'"
        self.transform = transform
        self.samples = []  # list of (image_path, label)

        for cls_name, label in config.LABEL_MAP.items():
            cls_dir = os.path.join(root_dir, split, cls_name)
            if not os.path.isdir(cls_dir):
                print(f"WARNING: Directory not found: {cls_dir}")
                continue
            for f in sorted(Path(cls_dir).iterdir()):
                if f.suffix.lower() in SUPPORTED_EXTENSIONS:
                    self.samples.append((str(f), label))

        if len(self.samples) == 0:
            print(
                f"WARNING: No images found in {root_dir}/{split}/. "
                "Run data_utils.split_dataset() first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        # Read image as BGR then convert to RGB
        img = cv2.imread(img_path)
        if img is None:
            # Return a blank image if file is corrupted
            img_sz = config.VIT_IMG_SIZE if config.ACTIVE_MODEL == "vit_b_16" else config.IMG_SIZE
            img = np.zeros((*img_sz, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        return img, label


def get_train_transforms(img_size=None):
    """Augmentation pipeline for training images."""
    if img_size is None:
        img_size = config.VIT_IMG_SIZE if config.ACTIVE_MODEL == "vit_b_16" else config.IMG_SIZE
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation((0, 360)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
    ])


def get_test_transforms(img_size=None):
    """Deterministic pipeline for validation/test images."""
    if img_size is None:
        img_size = config.VIT_IMG_SIZE if config.ACTIVE_MODEL == "vit_b_16" else config.IMG_SIZE
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
    ])


def get_dataloaders(
    processed_dir: str = None,
    batch_size: int = None,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
):
    """
    Build train and test DataLoaders.

    Args:
        processed_dir:        Path to data/processed/ (defaults to config value).
        batch_size:           Batch size (defaults to config.BATCH_SIZE).
        num_workers:          Number of DataLoader workers.
        use_weighted_sampler: If True, oversample minority class during training.

    Returns:
        (train_loader, test_loader)
    """
    if processed_dir is None:
        processed_dir = config.PROCESSED_DATA_DIR
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    train_dataset = LandslideDataset(processed_dir, "train", get_train_transforms())
    # Use 'val' split if present, else fall back to 'test'
    val_split = "val" if os.path.isdir(os.path.join(processed_dir, "val")) else "test"
    test_dataset = LandslideDataset(processed_dir, val_split, get_test_transforms())

    print(f"Train set: {len(train_dataset)} images")
    print(f"Val   set: {len(test_dataset)} images (split='{val_split}')")

    sampler = None
    shuffle = True
    if use_weighted_sampler and len(train_dataset) > 0:
        labels = [train_dataset.samples[i][1] for i in range(len(train_dataset))]
        class_counts = [labels.count(i) for i in range(config.NUM_CLASSES)]
        weights = [1.0 / class_counts[lbl] for lbl in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False  # mutually exclusive with sampler

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
