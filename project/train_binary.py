"""Train binary classification models (landslide vs non_landslide)."""

import os
import sys
import torch

# CUDA_LAUNCH_BLOCKING removed — it serializes GPU ops and kills performance

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from phase1_alexnet.train import run_training
from utils.plot_utils import plot_training_history

if __name__ == "__main__":
    # Clear any stale CUDA state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print("=" * 70)
    print("BINARY CLASSIFICATION: landslide vs non_landslide")
    print(f"Data: {config.PROCESSED_DATA_DIR}")
    print(f"Classes: {config.CLASS_NAMES}")
    print(f"NUM_CLASSES: {config.NUM_CLASSES}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)

    # Train ConvNeXt-CBAM-FPN
    print("\n\n>>> Training ConvNeXt-CBAM-FPN <<<\n")
    model1, hist1 = run_training(
        model_name="convnext_cbam_fpn",
        num_epochs=50,
        batch_size=32,
        learning_rate=5e-5,
    )
    plot_training_history(
        hist1,
        save_path=os.path.join(config.PLOTS_DIR, "training_history_convnext.png"),
    )

    # Clear GPU memory between models
    del model1
    torch.cuda.empty_cache()

    # Train SwinV2-Small (ensemble partner)
    print("\n\n>>> Training SwinV2-Small <<<\n")
    model2, hist2 = run_training(
        model_name="swinv2_s",
        num_epochs=50,
        batch_size=32,
        learning_rate=5e-5,
    )
    plot_training_history(
        hist2,
        save_path=os.path.join(config.PLOTS_DIR, "training_history_swinv2.png"),
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
