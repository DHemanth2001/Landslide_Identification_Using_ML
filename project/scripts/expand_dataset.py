"""
Offline dataset expansion via aggressive augmentation.

Expands the training set by generating multiple augmented copies of each image,
especially for minority classes. This is critical for small datasets.

Strategy (binary_combined dataset):
  - non_landslide (majority, 3265): 3x augmentation  → 3265 × 4 = 13,060
  - landslide      (minority, 1284): 9x augmentation  → 1284 × 10 = 12,840
  - Total training images: ~25,900 (balanced)

This runs ONCE before training, saves augmented images to disk.
"""

import os
import sys
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Augmentation multiplier per class (binary classification)
# Minority class gets more augmentation to balance the dataset
AUGMENT_MULTIPLIER = {
    "non_landslide": 3,         # 4189 × (1+3) = 16,756 total
    "landslide": 11,            # 1361 × (1+11) = 16,332 total
}


def random_augment(img: np.ndarray) -> np.ndarray:
    """Apply random augmentations to a single image (OpenCV, BGR format)."""
    h, w = img.shape[:2]
    result = img.copy()

    # Random horizontal flip
    if random.random() > 0.5:
        result = cv2.flip(result, 1)

    # Random vertical flip
    if random.random() > 0.5:
        result = cv2.flip(result, 0)

    # Random rotation (0, 90, 180, 270 + small random angle)
    angle = random.choice([0, 90, 180, 270]) + random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # Random brightness/contrast
    alpha = random.uniform(0.7, 1.4)   # contrast
    beta = random.randint(-30, 30)      # brightness
    result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)

    # Random Gaussian blur
    if random.random() > 0.5:
        ksize = random.choice([3, 5])
        result = cv2.GaussianBlur(result, (ksize, ksize), 0)

    # Random noise
    if random.random() > 0.6:
        noise = np.random.normal(0, 8, result.shape).astype(np.int16)
        result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Random scale + crop (simulates zoom)
    if random.random() > 0.5:
        scale = random.uniform(0.85, 1.2)
        new_h, new_w = int(h * scale), int(w * scale)
        result = cv2.resize(result, (new_w, new_h))
        # Center crop back to original size
        if new_h > h:
            y_off = (new_h - h) // 2
            x_off = (new_w - w) // 2
            result = result[y_off:y_off + h, x_off:x_off + w]
        else:
            # Pad if smaller
            pad_y = (h - new_h) // 2
            pad_x = (w - new_w) // 2
            result = cv2.copyMakeBorder(result, pad_y, h - new_h - pad_y,
                                        pad_x, w - new_w - pad_x,
                                        cv2.BORDER_REFLECT_101)

    # Random color channel shuffle
    if random.random() > 0.8:
        channels = list(range(3))
        random.shuffle(channels)
        result = result[:, :, channels]

    # Random hue shift
    if random.random() > 0.6:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + random.randint(-10, 10)) % 180
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return result


def expand_dataset(
    source_dir: str = None,
    output_dir: str = None,
    split: str = "train",
):
    """
    Expand the training dataset by generating augmented copies.

    Args:
        source_dir:  Path to data/processed/ (original dataset).
        output_dir:  Path to save expanded dataset. Defaults to data/processed_expanded/.
        split:       Which split to expand ('train' only).
    """
    if source_dir is None:
        source_dir = os.path.join(config.DATA_DIR, "binary_combined")
    if output_dir is None:
        output_dir = os.path.join(config.DATA_DIR, "binary_combined_expanded")

    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")

    # Copy val and test splits as-is
    for s in ["val", "test"]:
        src_split = os.path.join(source_dir, s)
        dst_split = os.path.join(output_dir, s)
        if os.path.isdir(src_split):
            for cls_name in config.LABEL_MAP.keys():
                src_cls = os.path.join(src_split, cls_name)
                dst_cls = os.path.join(dst_split, cls_name)
                if not os.path.isdir(src_cls):
                    continue
                os.makedirs(dst_cls, exist_ok=True)
                for f in sorted(Path(src_cls).iterdir()):
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        dst_path = os.path.join(dst_cls, f.name)
                        if not os.path.exists(dst_path):
                            img = cv2.imread(str(f))
                            if img is not None:
                                cv2.imwrite(dst_path, img)
            print(f"Copied {s} split as-is.")

    # Expand training split
    total_original = 0
    total_augmented = 0

    for cls_name in config.LABEL_MAP.keys():
        src_cls = os.path.join(source_dir, split, cls_name)
        dst_cls = os.path.join(output_dir, split, cls_name)
        os.makedirs(dst_cls, exist_ok=True)

        if not os.path.isdir(src_cls):
            print(f"WARNING: {src_cls} not found, skipping.")
            continue

        images = sorted([f for f in Path(src_cls).iterdir()
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        multiplier = AUGMENT_MULTIPLIER.get(cls_name, 4)

        print(f"\n{cls_name}: {len(images)} originals × {multiplier} augmentations")

        for img_path in tqdm(images, desc=f"  {cls_name}"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Save original
            dst_orig = os.path.join(dst_cls, img_path.name)
            cv2.imwrite(dst_orig, img)
            total_original += 1

            # Generate augmented copies
            stem = img_path.stem
            ext = img_path.suffix
            for aug_idx in range(multiplier):
                aug_img = random_augment(img)
                aug_name = f"{stem}_aug{aug_idx:02d}{ext}"
                cv2.imwrite(os.path.join(dst_cls, aug_name), aug_img)
                total_augmented += 1

    print(f"\nDataset expansion complete!")
    print(f"  Original images: {total_original}")
    print(f"  Augmented images: {total_augmented}")
    print(f"  Total training images: {total_original + total_augmented}")
    print(f"  Saved to: {output_dir}")

    # Print class distribution
    print(f"\nClass distribution (expanded):")
    for cls_name in config.LABEL_MAP.keys():
        cls_dir = os.path.join(output_dir, split, cls_name)
        if os.path.isdir(cls_dir):
            count = len([f for f in Path(cls_dir).iterdir()
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
            print(f"  {cls_name:>20s}: {count}")


if __name__ == "__main__":
    expand_dataset()
    print("\nDone! Now update config.py PROCESSED_DATA_DIR or pass the expanded path to training.")
