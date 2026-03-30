"""
Integrate multiple landslide datasets into unified binary classification format.

Combines:
  1. HR-GLDD (existing) — 128x128 RGB JPGs
  2. Bijie Landslide Dataset — variable-size RGB PNGs → resized to 128x128
  3. CAS Wenchuan — 512x512 TIFs with masks → classified & resized to 128x128
  4. CAS Lombok — 512x512 TIFs with masks → classified & resized to 128x128
  5. CAS Palu — 512x512 TIFs with masks → classified & resized to 128x128

Output: data/binary_combined/{train,val,test}/{landslide,non_landslide}/
"""

import os
import sys
import shutil
import random
import numpy as np
from PIL import Image
from pathlib import Path

random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

# Source datasets
HRGLDD_DIR = DATA_DIR / "binary"  # existing train/val/test split
BIJIE_DIR = DATA_DIR / "bijie_raw" / "Bijie-landslide-dataset"
CAS_DIRS = {
    "wenchuan": DATA_DIR / "cas_wenchuan_raw",
    "lombok": DATA_DIR / "cas_lombok_raw",
    "palu": DATA_DIR / "cas_palu_raw",
}

# Output
OUTPUT_DIR = DATA_DIR / "binary_combined"
TARGET_SIZE = (128, 128)
LANDSLIDE_MASK_THRESHOLD = 0.05  # >5% landslide pixels → classify as landslide


def copy_hrgldd():
    """Copy existing HR-GLDD data as the base."""
    counts = {"landslide": 0, "non_landslide": 0}
    for split in ["train", "val", "test"]:
        for cls in ["landslide", "non_landslide"]:
            src = HRGLDD_DIR / split / cls
            dst = OUTPUT_DIR / split / cls
            dst.mkdir(parents=True, exist_ok=True)
            if not src.exists():
                continue
            for f in src.iterdir():
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                    shutil.copy2(f, dst / f.name)
                    counts[cls] += 1
    print(f"  HR-GLDD copied: {counts['landslide']} landslide, {counts['non_landslide']} non-landslide")
    return counts


def integrate_bijie():
    """Integrate Bijie dataset — resize to 128x128 and add to train split."""
    if not BIJIE_DIR.exists():
        print("  Bijie dataset not found, skipping.")
        return {"landslide": 0, "non_landslide": 0}

    counts = {"landslide": 0, "non_landslide": 0}
    mapping = {
        "landslide": BIJIE_DIR / "landslide" / "image",
        "non_landslide": BIJIE_DIR / "non-landslide" / "image",
    }

    for cls, src_dir in mapping.items():
        if not src_dir.exists():
            continue
        # Split 80/10/10
        files = sorted([f for f in src_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".tif")])
        random.shuffle(files)
        n = len(files)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)

        splits = {
            "train": files[:train_end],
            "val": files[train_end:val_end],
            "test": files[val_end:],
        }

        for split, split_files in splits.items():
            dst = OUTPUT_DIR / split / cls
            dst.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                try:
                    img = Image.open(f).convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)
                    out_name = f"bijie_{f.stem}.jpg"
                    img.save(dst / out_name, "JPEG", quality=95)
                    counts[cls] += 1
                except Exception as e:
                    print(f"    Warning: could not process {f.name}: {e}")

    print(f"  Bijie integrated: {counts['landslide']} landslide, {counts['non_landslide']} non-landslide")
    return counts


def integrate_cas_subset(name, cas_dir):
    """Integrate a CAS dataset subset — classify by mask, resize to 128x128."""
    if not cas_dir.exists():
        print(f"  CAS {name} not found, skipping.")
        return {"landslide": 0, "non_landslide": 0}

    img_dir = cas_dir / "img"
    mask_dir = cas_dir / "mask"

    if not img_dir.exists() or not mask_dir.exists():
        print(f"  CAS {name}: img or mask dir missing, skipping.")
        return {"landslide": 0, "non_landslide": 0}

    counts = {"landslide": 0, "non_landslide": 0}

    # Classify each image based on mask
    landslide_files = []
    non_landslide_files = []

    for f in sorted(img_dir.iterdir()):
        if f.suffix.lower() not in (".tif", ".tiff", ".png", ".jpg"):
            continue
        # Find matching mask
        mask_candidates = [
            mask_dir / f.name,
            mask_dir / f.with_suffix(".TIF").name,
            mask_dir / f.with_suffix(".tif").name,
            mask_dir / f.with_suffix(".png").name,
        ]
        mask_path = None
        for mc in mask_candidates:
            if mc.exists():
                mask_path = mc
                break
        if mask_path is None:
            continue

        try:
            mask = np.array(Image.open(mask_path))
            ratio = mask.sum() / max(mask.size, 1)
            if ratio > LANDSLIDE_MASK_THRESHOLD:
                landslide_files.append(f)
            else:
                non_landslide_files.append(f)
        except Exception:
            continue

    # Split and save
    for cls, files in [("landslide", landslide_files), ("non_landslide", non_landslide_files)]:
        random.shuffle(files)
        n = len(files)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)

        splits = {
            "train": files[:train_end],
            "val": files[train_end:val_end],
            "test": files[val_end:],
        }

        for split, split_files in splits.items():
            dst = OUTPUT_DIR / split / cls
            dst.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                try:
                    img = Image.open(f).convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)
                    out_name = f"cas_{name}_{f.stem}.jpg"
                    img.save(dst / out_name, "JPEG", quality=95)
                    counts[cls] += 1
                except Exception as e:
                    print(f"    Warning: could not process {f.name}: {e}")

    print(f"  CAS {name} integrated: {counts['landslide']} landslide, {counts['non_landslide']} non-landslide")
    return counts


def main():
    print("=" * 60)
    print("INTEGRATING MULTIPLE LANDSLIDE DATASETS")
    print("=" * 60)

    # Clean output dir
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    total = {"landslide": 0, "non_landslide": 0}

    # 1. HR-GLDD (base)
    print("\n[1/5] Copying HR-GLDD base dataset...")
    c = copy_hrgldd()
    total["landslide"] += c["landslide"]
    total["non_landslide"] += c["non_landslide"]

    # 2. Bijie
    print("\n[2/5] Integrating Bijie landslide dataset...")
    c = integrate_bijie()
    total["landslide"] += c["landslide"]
    total["non_landslide"] += c["non_landslide"]

    # 3-5. CAS subsets
    for i, (name, cas_dir) in enumerate(CAS_DIRS.items(), 3):
        print(f"\n[{i}/5] Integrating CAS {name}...")
        c = integrate_cas_subset(name, cas_dir)
        total["landslide"] += c["landslide"]
        total["non_landslide"] += c["non_landslide"]

    # Summary
    print("\n" + "=" * 60)
    print("INTEGRATION COMPLETE")
    print("=" * 60)
    print(f"Total landslide:     {total['landslide']}")
    print(f"Total non-landslide: {total['non_landslide']}")
    print(f"Grand total:         {total['landslide'] + total['non_landslide']}")

    # Per-split counts
    for split in ["train", "val", "test"]:
        ls = len(list((OUTPUT_DIR / split / "landslide").glob("*"))) if (OUTPUT_DIR / split / "landslide").exists() else 0
        nls = len(list((OUTPUT_DIR / split / "non_landslide").glob("*"))) if (OUTPUT_DIR / split / "non_landslide").exists() else 0
        print(f"  {split:5s}: {ls} landslide, {nls} non-landslide = {ls + nls} total")

    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
