import os
import shutil
import random
from pathlib import Path

def create_multiclass_dataset(processed_dir):
    """
    Simulates a multi-class dataset by randomly moving existing `landslide` images
    into 5 new subdirectories: 'rockfall', 'mudflow', 'debris_flow', 'rotational_slide', 'translational_slide'.
    Leaves 'non_landslide' untouched.
    """
    splits = ['train', 'val', 'test']
    subclasses = ['rockfall', 'mudflow', 'debris_flow', 'rotational_slide', 'translational_slide']

    for split in splits:
        split_dir = os.path.join(processed_dir, split)
        if not os.path.exists(split_dir):
            continue

        landslide_dir = os.path.join(split_dir, "landslide")
        
        # Create subclasses directories
        for sc in subclasses:
            os.makedirs(os.path.join(split_dir, sc), exist_ok=True)
            
        if not os.path.exists(landslide_dir):
            # Already split in a previous run? Check if subclasses exist and possess files
            print(f"{split_dir}/landslide does not exist, assuming already processed.")
            continue

        images = [f for f in os.listdir(landslide_dir) if os.path.isfile(os.path.join(landslide_dir, f))]
        print(f"Distributing {len(images)} images in {split}/landslide to 5 sub-classes...")

        for img in images:
            src = os.path.join(landslide_dir, img)
            random_class = random.choice(subclasses)
            dst = os.path.join(split_dir, random_class, img)
            shutil.move(src, dst)
            
        # Clean up empty landslide dir
        if not os.listdir(landslide_dir):
            os.rmdir(landslide_dir)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    create_multiclass_dataset(PROCESSED_DATA_DIR)
    print("Multi-class dataset generation complete!")
