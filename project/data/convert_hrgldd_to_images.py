"""
Convert HR-GLDD numpy arrays to JPEG image files organized for binary classification.

Strategy:
- trainX/valX/testX: float32 arrays, shape (N, 128, 128, 4) — 4-band satellite (RGB + NIR)
- trainY/valY/testY: float32 masks, shape (N, 128, 128, 1) — pixel-level labels

For binary classification:
- Patches where >5% pixels are landslide  → class: landslide
- Patches where 0% pixels are landslide   → class: non_landslide
- For non-landslide: use augmented/flipped versions of empty-mask patches,
  OR generate synthetic background patches from low-value regions.

Since all patches have some landslide content, we apply:
1. Original patches with high landslide coverage → landslide/
2. Flipped/cropped sub-patches of the same images with ZERO landslide pixels → non_landslide/
   (crop 64x64 regions from corners where mask==0)
"""

import os
import sys
import numpy as np
import cv2

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(DATA_DIR)
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

sys.path.insert(0, PROJECT_DIR)
import config


def normalize_to_uint8(img_float: np.ndarray) -> np.ndarray:
    """Convert float32 [0,1] to uint8 [0,255] RGB image."""
    # Use bands 0,1,2 as RGB (or 2,1,0 for natural color)
    rgb = img_float[:, :, :3]
    rgb_clipped = np.clip(rgb, 0.0, 1.0)
    return (rgb_clipped * 255).astype(np.uint8)


def extract_nonlandslide_crops(X: np.ndarray, Y: np.ndarray, target_count: int) -> list:
    """
    Extract 64x64 sub-crops from areas where the mask is entirely 0 (no landslide).
    Resize to 128x128. Returns list of uint8 images.
    """
    crops = []
    np.random.seed(42)
    step = 64

    for i in range(len(X)):
        mask = Y[i, :, :, 0]  # (128,128)
        img = normalize_to_uint8(X[i])

        for r in range(0, 128 - step, step):
            for c in range(0, 128 - step, step):
                sub_mask = mask[r:r+step, c:c+step]
                if sub_mask.sum() == 0:  # no landslide pixels
                    sub_img = img[r:r+step, c:c+step]
                    resized = cv2.resize(sub_img, (128, 128), interpolation=cv2.INTER_LINEAR)
                    crops.append(resized)
                    if len(crops) >= target_count:
                        return crops

    # If not enough, add augmented versions
    for i in range(len(X)):
        if len(crops) >= target_count:
            break
        img = normalize_to_uint8(X[i])
        mask = Y[i, :, :, 0]
        if mask.mean() < 0.1:  # < 10% landslide
            flipped = cv2.flip(img, 1)
            crops.append(flipped)

    return crops[:target_count]


def convert_split(split_name: str, X: np.ndarray, Y: np.ndarray, out_dir: str) -> dict:
    """Convert one split to landslide/ and non_landslide/ image folders."""
    ls_dir  = os.path.join(out_dir, split_name, "landslide")
    nls_dir = os.path.join(out_dir, split_name, "non_landslide")
    os.makedirs(ls_dir,  exist_ok=True)
    os.makedirs(nls_dir, exist_ok=True)

    # Determine patch-level label (>5% landslide pixels → landslide)
    patch_labels = ((Y.sum(axis=(1,2,3)) / (128*128)) > 0.05).astype(int)
    ls_indices  = np.where(patch_labels == 1)[0]
    nls_indices = np.where(patch_labels == 0)[0]

    print(f"  {split_name}: {len(ls_indices)} landslide patches, {len(nls_indices)} non-ls patches (from masks)")

    # Save landslide patches
    saved_ls = 0
    for idx in ls_indices:
        img = normalize_to_uint8(X[idx])
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        path = os.path.join(ls_dir, f"ls_{split_name}_{idx:04d}.jpg")
        cv2.imwrite(path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_ls += 1

    # Generate non-landslide images
    target_nls = len(ls_indices) * 3  # 3x more non-landslide (matches project ratio)
    nls_images = extract_nonlandslide_crops(X, Y, target_count=target_nls)

    # Also save any zero-mask patches if available
    for idx in nls_indices:
        if len(nls_images) >= target_nls:
            break
        img = normalize_to_uint8(X[idx])
        nls_images.append(img)

    saved_nls = 0
    for j, img in enumerate(nls_images[:target_nls]):
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        path = os.path.join(nls_dir, f"nls_{split_name}_{j:04d}.jpg")
        cv2.imwrite(path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_nls += 1

    print(f"  Saved: {saved_ls} landslide, {saved_nls} non_landslide → {out_dir}/{split_name}/")
    return {"landslide": saved_ls, "non_landslide": saved_nls}


def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(data_dir, "processed")

    print("Converting HR-GLDD numpy arrays to JPEG images...\n")

    totals = {}
    for split in ["train", "val", "test"]:
        print(f"Processing {split}...")
        X = np.load(os.path.join(data_dir, f"{split}X.npy"))
        Y = np.load(os.path.join(data_dir, f"{split}Y.npy"))
        counts = convert_split(split, X, Y, processed_dir)
        totals[split] = counts
        del X, Y  # free memory

    print("\n=== Conversion Summary ===")
    for split, counts in totals.items():
        total = counts["landslide"] + counts["non_landslide"]
        print(f"  {split:6s}: {counts['landslide']:4d} landslide + {counts['non_landslide']:4d} non_landslide = {total} total")

    # Also copy val into test folder if test is too small
    print("\nDone! Images ready in data/processed/")


if __name__ == "__main__":
    main()
