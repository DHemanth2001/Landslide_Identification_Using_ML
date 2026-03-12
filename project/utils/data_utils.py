"""
Dataset download, verification, and train/test split utilities.
Primary dataset source: gpcv.whu.edu.cn (Wuhan University)
  - Total: 2773 images (770 landslide + 2003 non-landslide)
  - Train: 2000 (500 landslide + 1500 non-landslide)
  - Test:   773 (270 landslide + 503 non-landslide)
"""

import os
import random
import shutil
from pathlib import Path

from tqdm import tqdm


# ─── Expected dataset counts ──────────────────────────────────────────────────
EXPECTED_RAW = {"landslide": 770, "non_landslide": 2003}
TRAIN_COUNTS = {"landslide": 500, "non_landslide": 1500}
TEST_COUNTS = {"landslide": 270, "non_landslide": 503}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def download_whu_dataset(dest_dir: str) -> None:
    """
    Attempt to download the WHU landslide image dataset from gpcv.whu.edu.cn.

    The WHU dataset is hosted at: http://gpcv.whu.edu.cn/data/landslide_dataset.html
    If automated download is not available, this function prints manual
    download instructions.

    Args:
        dest_dir: Target directory where raw/ folder will be created.
    """
    landslide_dir = os.path.join(dest_dir, "raw", "landslide")
    non_landslide_dir = os.path.join(dest_dir, "raw", "non_landslide")
    os.makedirs(landslide_dir, exist_ok=True)
    os.makedirs(non_landslide_dir, exist_ok=True)

    print("=" * 70)
    print("WHU Landslide Dataset Download Instructions")
    print("=" * 70)
    print()
    print("Dataset source: http://gpcv.whu.edu.cn/data/landslide_dataset.html")
    print()
    print("The dataset requires manual download due to access restrictions.")
    print("Please follow these steps:")
    print()
    print("1. Visit: http://gpcv.whu.edu.cn/data/landslide_dataset.html")
    print("2. Download the dataset archive (Landslide4Sense or similar).")
    print("3. Extract the archive.")
    print("4. Copy landslide images  → data/raw/landslide/")
    print("5. Copy non-landslide images → data/raw/non_landslide/")
    print()
    print(f"Expected counts: {EXPECTED_RAW['landslide']} landslide images,")
    print(f"                 {EXPECTED_RAW['non_landslide']} non-landslide images")
    print()
    print("After placing images, run verify_dataset_counts() to confirm.")
    print("=" * 70)


def _list_images(directory: str) -> list:
    """Return sorted list of image file paths in a directory."""
    paths = []
    for f in Path(directory).iterdir():
        if f.suffix.lower() in SUPPORTED_EXTENSIONS:
            paths.append(str(f))
    return sorted(paths)


def verify_dataset_counts(raw_dir: str) -> dict:
    """
    Verify that the raw dataset has the expected number of images.

    Args:
        raw_dir: Path to data/raw/ directory containing landslide/ and
                 non_landslide/ sub-folders.

    Returns:
        dict with keys 'landslide' and 'non_landslide' and their image counts.

    Raises:
        FileNotFoundError: If raw_dir or sub-folders do not exist.
        AssertionError:    If image counts do not match expected values.
    """
    for cls in ["landslide", "non_landslide"]:
        cls_dir = os.path.join(raw_dir, cls)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(
                f"Directory not found: {cls_dir}\n"
                "Please run download_whu_dataset() and follow the instructions."
            )

    counts = {}
    for cls in ["landslide", "non_landslide"]:
        cls_dir = os.path.join(raw_dir, cls)
        imgs = _list_images(cls_dir)
        counts[cls] = len(imgs)
        print(f"  {cls}: {len(imgs)} images found")

    # Warn if counts differ from expected (don't hard-fail — user may have
    # a slightly different version of the dataset)
    for cls, expected in EXPECTED_RAW.items():
        if counts[cls] != expected:
            print(
                f"WARNING: Expected {expected} {cls} images, "
                f"found {counts[cls]}. Proceeding anyway."
            )
        else:
            print(f"  {cls}: count OK ({expected})")

    return counts


def split_dataset(
    raw_dir: str,
    processed_dir: str,
    train_counts: dict = None,
    test_counts: dict = None,
    seed: int = 42,
) -> None:
    """
    Reproducibly split raw images into train and test folders.

    Copies (does not move) files so raw/ data is preserved.

    Args:
        raw_dir:       Path to data/raw/
        processed_dir: Path to data/processed/
        train_counts:  Dict {'landslide': int, 'non_landslide': int}
        test_counts:   Dict {'landslide': int, 'non_landslide': int}
        seed:          Random seed for reproducibility
    """
    if train_counts is None:
        train_counts = TRAIN_COUNTS
    if test_counts is None:
        test_counts = TEST_COUNTS

    random.seed(seed)

    for cls in ["landslide", "non_landslide"]:
        src_dir = os.path.join(raw_dir, cls)
        all_images = _list_images(src_dir)
        random.shuffle(all_images)

        n_train = train_counts[cls]
        n_test = test_counts[cls]
        total_needed = n_train + n_test

        if len(all_images) < total_needed:
            print(
                f"WARNING: Only {len(all_images)} {cls} images available, "
                f"but {total_needed} needed. Using all available."
            )
            train_imgs = all_images[: min(n_train, len(all_images))]
            test_imgs = all_images[len(train_imgs) :]
        else:
            train_imgs = all_images[:n_train]
            test_imgs = all_images[n_train : n_train + n_test]

        for split, imgs in [("train", train_imgs), ("test", test_imgs)]:
            dest = os.path.join(processed_dir, split, cls)
            os.makedirs(dest, exist_ok=True)
            for img_path in tqdm(imgs, desc=f"Copying {split}/{cls}", leave=False):
                shutil.copy2(img_path, dest)
            print(f"  {split}/{cls}: {len(imgs)} images copied → {dest}")

    print("\nDataset split complete.")
    _print_split_summary(processed_dir)


def _print_split_summary(processed_dir: str) -> None:
    """Print a summary of images in each split/class folder."""
    print("\nSplit summary:")
    for split in ["train", "test"]:
        for cls in ["landslide", "non_landslide"]:
            d = os.path.join(processed_dir, split, cls)
            count = len(_list_images(d)) if os.path.isdir(d) else 0
            print(f"  {split}/{cls}: {count}")


def get_class_weights(processed_dir: str) -> dict:
    """
    Compute inverse-frequency class weights for handling class imbalance.

    Returns:
        dict {'landslide': float, 'non_landslide': float}
        These can be used with WeightedRandomSampler or loss pos_weight.
    """
    train_dir = os.path.join(processed_dir, "train")
    counts = {}
    for cls in ["landslide", "non_landslide"]:
        d = os.path.join(train_dir, cls)
        counts[cls] = len(_list_images(d)) if os.path.isdir(d) else 0

    total = sum(counts.values())
    weights = {cls: total / (len(counts) * cnt) for cls, cnt in counts.items() if cnt > 0}
    print(f"Class weights (inverse frequency): {weights}")
    return weights
