"""
PyTorch Dataset and DataLoader for landslide 6-class image classification.

Includes advanced augmentation strategies:
  - Mixup (Zhang et al., 2018): interpolates between two training images
  - CutMix (Yun et al., 2019): pastes a patch from one image onto another
  - RandAugment-style transforms for training robustness
"""

import os
import random
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
    Dataset for 6-class landslide classification.
    Loads images from: processed/<split>/<class_name>/
    """

    def __init__(self, root_dir: str, split: str, transform=None):
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

        img = cv2.imread(img_path)
        if img is None:
            img_sz = config.IMG_SIZE
            img = np.zeros((*img_sz, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        return img, label


# ─── Augmentation Pipelines ──────────────────────────────────────────────────

def get_train_transforms(img_size=None):
    """
    Strong augmentation pipeline for training.
    Includes geometric + photometric augmentations for maximum robustness.
    """
    if img_size is None:
        if config.ACTIVE_MODEL == "swinv2_s":
            img_size = config.SWINV2_IMG_SIZE
        else:
            img_size = config.CONVNEXT_IMG_SIZE

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(0, 360)),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.85, 1.15),
            shear=10,
        ),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.3,
            hue=0.08,
        ),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
    ])


def get_test_transforms(img_size=None):
    """Deterministic pipeline for validation/test images."""
    if img_size is None:
        if config.ACTIVE_MODEL == "swinv2_s":
            img_size = config.SWINV2_IMG_SIZE
        else:
            img_size = config.CONVNEXT_IMG_SIZE

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
    ])


# ─── Mixup & CutMix ──────────────────────────────────────────────────────────

def mixup_data(x, y, alpha=0.3):
    """
    Mixup: creates convex combinations of training examples.
    Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018.

    Returns mixed inputs, pairs of targets, and lambda value.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1.0 - lam)  # Ensure lam >= 0.5 so label order is consistent
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """
    CutMix: cuts and pastes patches between training images.
    Reference: Yun et al., "CutMix: Regularization Strategy to Train Strong
               Classifiers with Localizable Features", ICCV 2019.

    Returns mixed inputs, pairs of targets, and lambda value.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    _, _, h, w = x.shape

    # Sample bounding box
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)

    cy = np.random.randint(h)
    cx = np.random.randint(w)

    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # Adjust lambda to match the actual area ratio
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (h * w)

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for mixup/cutmix mixed targets."""
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


# ─── DataLoaders ──────────────────────────────────────────────────────────────

def get_dataloaders(
    processed_dir: str = None,
    batch_size: int = None,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
    model_name: str = None,
):
    """
    Build train and validation DataLoaders with class-balanced sampling.

    Args:
        processed_dir:        Path to data/processed/.
        batch_size:           Batch size.
        num_workers:          Number of DataLoader workers.
        use_weighted_sampler: If True, oversample minority classes.
        model_name:           Model name to determine image size.

    Returns:
        (train_loader, val_loader)
    """
    if processed_dir is None:
        processed_dir = config.PROCESSED_DATA_DIR
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if model_name is None:
        model_name = config.ACTIVE_MODEL

    # Select image size based on model
    if model_name == "swinv2_s":
        img_size = config.SWINV2_IMG_SIZE
    else:
        img_size = config.CONVNEXT_IMG_SIZE

    train_dataset = LandslideDataset(processed_dir, "train", get_train_transforms(img_size))
    val_split = "val" if os.path.isdir(os.path.join(processed_dir, "val")) else "test"
    val_dataset = LandslideDataset(processed_dir, val_split, get_test_transforms(img_size))

    print(f"Train set: {len(train_dataset)} images")
    print(f"Val   set: {len(val_dataset)} images (split='{val_split}')")

    sampler = None
    shuffle = True
    if use_weighted_sampler and len(train_dataset) > 0:
        labels = [train_dataset.samples[i][1] for i in range(len(train_dataset))]
        class_counts = [max(labels.count(i), 1) for i in range(config.NUM_CLASSES)]
        weights = [1.0 / class_counts[lbl] for lbl in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batch for stable mixup/cutmix
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
