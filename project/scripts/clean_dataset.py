"""
Rebuild binary_combined from scratch with stricter quality controls.

Sources included:
  1. HR-GLDD (data/binary/)          — Keep as-is, copy train/val/test splits.
  2. Bijie  (data/bijie_raw/...)     — Resize to 128x128, split 80/10/10.
  3. CAS Lombok (data/cas_lombok_raw/) — 20% mask threshold, resize 128x128, split 80/10/10.
  4. CAS Palu   (data/cas_palu_raw/)   — 20% mask threshold, resize 128x128, split 80/10/10.

Removed:
  - Kaggle-divided data (too noisy — 70% of images have <5% landslide pixels)
  - CAS Wenchuan (not included in clean rebuild)

Output: data/binary_combined_clean/{train,val,test}/{landslide,non_landslide}/
"""

import os
import sys
import shutil
import random
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict

# Reproducibility
random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

# Source datasets
HRGLDD_DIR = DATA_DIR / "binary"
BIJIE_DIR = DATA_DIR / "bijie_raw" / "Bijie-landslide-dataset"
CAS_DIRS = {
    "lombok": DATA_DIR / "cas_lombok_raw",
    "palu": DATA_DIR / "cas_palu_raw",
}

# Output
OUTPUT_DIR = DATA_DIR / "binary_combined_clean"
TARGET_SIZE = (128, 128)
LANDSLIDE_MASK_THRESHOLD = 0.20  # >20% landslide pixels → classify as landslide

# Valid image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Label map (matches config.py)
LABEL_MAP = {
    "non_landslide": 0,
    "landslide": 1,
}
CLASS_NAMES = list(LABEL_MAP.keys())

# ── Statistics tracker ────────────────────────────────────────────────────────
# stats[source][split][class] = count
stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))


def is_image(path):
    """Check if a file has a valid image extension."""
    return path.suffix.lower() in IMAGE_EXTS


def copy_hrgldd():
    """Copy existing HR-GLDD data as-is, preserving train/val/test splits."""
    source_name = "HR-GLDD"
    if not HRGLDD_DIR.exists():
        print(f"  [SKIP] HR-GLDD directory not found: {HRGLDD_DIR}")
        return

    for split in ["train", "val", "test"]:
        for cls in CLASS_NAMES:
            src = HRGLDD_DIR / split / cls
            dst = OUTPUT_DIR / split / cls
            dst.mkdir(parents=True, exist_ok=True)

            if not src.exists():
                continue

            for f in sorted(src.iterdir()):
                if not is_image(f):
                    continue
                shutil.copy2(f, dst / f.name)
                stats[source_name][split][cls] += 1

    total_ls = sum(stats[source_name][s]["landslide"] for s in ["train", "val", "test"])
    total_nls = sum(stats[source_name][s]["non_landslide"] for s in ["train", "val", "test"])
    print(f"  HR-GLDD copied: {total_ls} landslide, {total_nls} non_landslide")


def split_and_save_files(files, cls, source_prefix, source_name):
    """Shuffle files, split 80/10/10, resize to 128x128, and save as JPEG."""
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
                out_name = f"{source_prefix}_{f.stem}.jpg"
                img.save(dst / out_name, "JPEG", quality=95)
                stats[source_name][split][cls] += 1
            except Exception as e:
                print(f"    Warning: could not process {f.name}: {e}")


def integrate_bijie():
    """Integrate Bijie dataset — use folder structure for labels, resize, split 80/10/10."""
    source_name = "Bijie"
    if not BIJIE_DIR.exists():
        print(f"  [SKIP] Bijie directory not found: {BIJIE_DIR}")
        return

    # Bijie folder structure: landslide/image/ and non-landslide/image/
    mapping = {
        "landslide": BIJIE_DIR / "landslide" / "image",
        "non_landslide": BIJIE_DIR / "non-landslide" / "image",
    }

    for cls, src_dir in mapping.items():
        if not src_dir.exists():
            print(f"    Warning: Bijie {cls} dir not found: {src_dir}")
            continue

        files = sorted([f for f in src_dir.iterdir() if is_image(f)])
        split_and_save_files(files, cls, "bijie", source_name)

    total_ls = sum(stats[source_name][s]["landslide"] for s in ["train", "val", "test"])
    total_nls = sum(stats[source_name][s]["non_landslide"] for s in ["train", "val", "test"])
    print(f"  Bijie integrated: {total_ls} landslide, {total_nls} non_landslide")


def integrate_cas_subset(name, cas_dir):
    """Integrate a CAS dataset — classify by mask with 20% threshold, resize, split 80/10/10."""
    source_name = f"CAS-{name.capitalize()}"
    if not cas_dir.exists():
        print(f"  [SKIP] CAS {name} directory not found: {cas_dir}")
        return

    img_dir = cas_dir / "img"
    mask_dir = cas_dir / "mask"

    if not img_dir.exists() or not mask_dir.exists():
        print(f"  [SKIP] CAS {name}: img/ or mask/ directory missing.")
        return

    landslide_files = []
    non_landslide_files = []

    for f in sorted(img_dir.iterdir()):
        if not is_image(f):
            continue

        # Find matching mask file (try multiple extensions)
        mask_path = None
        for ext in [f.suffix, ".TIF", ".tif", ".png", ".jpg"]:
            candidate = mask_dir / f.with_suffix(ext).name
            if candidate.exists():
                mask_path = candidate
                break

        if mask_path is None:
            continue

        try:
            mask = np.array(Image.open(mask_path))
            # Normalize mask to binary: any nonzero value counts as landslide pixel
            if mask.ndim > 2:
                mask = mask.mean(axis=-1)
            binary_mask = (mask > 0).astype(np.float32)
            ratio = binary_mask.sum() / max(binary_mask.size, 1)

            if ratio > LANDSLIDE_MASK_THRESHOLD:
                landslide_files.append(f)
            else:
                non_landslide_files.append(f)
        except Exception as e:
            print(f"    Warning: could not read mask for {f.name}: {e}")
            continue

    prefix = f"cas_{name}"
    split_and_save_files(landslide_files, "landslide", prefix, source_name)
    split_and_save_files(non_landslide_files, "non_landslide", prefix, source_name)

    total_ls = sum(stats[source_name][s]["landslide"] for s in ["train", "val", "test"])
    total_nls = sum(stats[source_name][s]["non_landslide"] for s in ["train", "val", "test"])
    print(f"  CAS {name} integrated: {total_ls} landslide, {total_nls} non_landslide")


def print_statistics():
    """Print detailed statistics: per-source, per-class, per-split."""
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)

    all_sources = sorted(stats.keys())
    splits = ["train", "val", "test"]

    # Per-source breakdown
    print("\n--- Per-Source Counts ---")
    print(f"{'Source':<18} {'Train LS':>10} {'Train NLS':>10} {'Val LS':>8} {'Val NLS':>9} {'Test LS':>9} {'Test NLS':>10} {'Total':>8}")
    print("-" * 92)

    grand_totals = defaultdict(lambda: defaultdict(int))

    for source in all_sources:
        row = []
        source_total = 0
        for split in splits:
            for cls in CLASS_NAMES:
                count = stats[source][split][cls]
                row.append(count)
                grand_totals[split][cls] += count
                source_total += count
        row.append(source_total)
        print(f"{source:<18} {row[0]:>10} {row[1]:>10} {row[2]:>8} {row[3]:>9} {row[4]:>9} {row[5]:>10} {row[6]:>8}")

    # Totals row
    print("-" * 92)
    total_row = []
    grand_total = 0
    for split in splits:
        for cls in CLASS_NAMES:
            total_row.append(grand_totals[split][cls])
            grand_total += grand_totals[split][cls]
    total_row.append(grand_total)
    print(f"{'TOTAL':<18} {total_row[0]:>10} {total_row[1]:>10} {total_row[2]:>8} {total_row[3]:>9} {total_row[4]:>9} {total_row[5]:>10} {total_row[6]:>8}")

    # Per-class totals
    print("\n--- Total Per Class ---")
    for cls in CLASS_NAMES:
        total = sum(grand_totals[s][cls] for s in splits)
        print(f"  {cls:<16}: {total}")
    print(f"  {'GRAND TOTAL':<16}: {grand_total}")

    # Per-split totals
    print("\n--- Total Per Split ---")
    for split in splits:
        ls = grand_totals[split]["landslide"]
        nls = grand_totals[split]["non_landslide"]
        total = ls + nls
        balance = (ls / total * 100) if total > 0 else 0
        print(f"  {split:<6}: {ls:>6} landslide + {nls:>6} non_landslide = {total:>6} total  ({balance:.1f}% landslide)")

    print(f"\nOutput directory: {OUTPUT_DIR}")


def main():
    print("=" * 70)
    print("CLEAN DATASET REBUILD")
    print("=" * 70)
    print(f"Mask threshold: {LANDSLIDE_MASK_THRESHOLD:.0%} (only images with >{LANDSLIDE_MASK_THRESHOLD:.0%}")
    print(f"  landslide pixels are labeled as landslide)")
    print(f"Target size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    print(f"Sources: HR-GLDD, Bijie, CAS Lombok, CAS Palu")
    print(f"Excluded: Kaggle-divided (too noisy), CAS Wenchuan")

    # Clean output directory
    if OUTPUT_DIR.exists():
        print(f"\nRemoving existing output: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    # 1. HR-GLDD (base — clean classification dataset, keep splits)
    print("\n[1/4] Copying HR-GLDD base dataset...")
    copy_hrgldd()

    # 2. Bijie (folder-based labels, resize, split 80/10/10)
    print("\n[2/4] Integrating Bijie landslide dataset...")
    integrate_bijie()

    # 3. CAS Lombok (20% mask threshold, resize, split 80/10/10)
    print("\n[3/4] Integrating CAS Lombok...")
    integrate_cas_subset("lombok", CAS_DIRS["lombok"])

    # 4. CAS Palu (20% mask threshold, resize, split 80/10/10)
    print("\n[4/4] Integrating CAS Palu...")
    integrate_cas_subset("palu", CAS_DIRS["palu"])

    # Print full statistics
    print_statistics()

    print("\nDone.")


if __name__ == "__main__":
    main()
