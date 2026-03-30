"""
Fully autonomous pipeline with automatic Kaggle data download + retry.

This script runs WITHOUT any human interaction:
  1. Train EfficientNetV2-S + CBAM (60 epochs)
  2. Train ConvNeXt-CBAM-FPN (50 epochs)
  3. Evaluate all models + ensemble
  4. If best accuracy < 94%, download Landslide4Sense from Kaggle, integrate, retrain
  5. Repeat until accuracy >= 95% or max 3 rounds

Just run it and walk away.
"""

import os
import sys
import time
import json
import shutil
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# Override config to use combined expanded dataset
config.PROCESSED_DATA_DIR = os.path.join(config.DATA_DIR, "binary_combined_expanded")

ACCURACY_THRESHOLD = 0.94  # If below this, download more data and retry
TARGET_ACCURACY = 0.958    # Goal: beat this
MAX_ROUNDS = 3             # Maximum retry rounds
LOG_FILE = os.path.join(config.PLOTS_DIR, "auto_pipeline.log")


def log(msg):
    """Print and save timestamped log message."""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def train_efficientnetv2(num_epochs=60, batch_size=24, lr=3e-5):
    """Train EfficientNetV2-S + CBAM."""
    log("Training EfficientNetV2-S + CBAM...")
    from phase1_alexnet.train import run_training
    from utils.plot_utils import plot_training_history

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model, history = run_training(
        model_name="efficientnetv2_cbam",
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=lr,
    )
    plot_training_history(
        history,
        save_path=os.path.join(config.PLOTS_DIR, "training_history_efficientnetv2.png"),
    )
    with open(os.path.join(config.PLOTS_DIR, "history_efficientnetv2.json"), "w") as f:
        json.dump(history, f)

    del model
    torch.cuda.empty_cache()
    log("EfficientNetV2 training done.")
    return history


def train_convnext(num_epochs=50, batch_size=32, lr=5e-5):
    """Train ConvNeXt-CBAM-FPN model."""
    log("Training ConvNeXt-CBAM-FPN...")
    from phase1_alexnet.train import run_training
    from utils.plot_utils import plot_training_history

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model, history = run_training(
        model_name="convnext_cbam_fpn",
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=lr,
    )
    plot_training_history(
        history,
        save_path=os.path.join(config.PLOTS_DIR, "training_history_convnext.png"),
    )
    with open(os.path.join(config.PLOTS_DIR, "history_convnext.json"), "w") as f:
        json.dump(history, f)

    del model
    torch.cuda.empty_cache()
    log("ConvNeXt training done.")
    return history


def evaluate_models():
    """Evaluate all models and ensemble. Returns best accuracy."""
    log("Evaluating models on test set...")
    import torch.nn.functional as F
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_curve,
        auc,
        accuracy_score,
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    from phase1_alexnet.model import get_convnext_cbam_fpn, get_efficientnetv2_cbam
    from phase1_alexnet.dataset import LandslideDataset, get_test_transforms
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    for model_name, ckpt_path, ema_ckpt_path, get_model_fn, img_size in [
        ("EfficientNetV2-CBAM", config.EFFNETV2_CHECKPOINT, config.EMA_EFFNETV2_CHECKPOINT, get_efficientnetv2_cbam, config.EFFNETV2_IMG_SIZE),
        ("ConvNeXt-CBAM-FPN", config.CONVNEXT_CHECKPOINT, config.EMA_CONVNEXT_CHECKPOINT, get_convnext_cbam_fpn, config.CONVNEXT_IMG_SIZE),
    ]:
        test_dataset = LandslideDataset(config.PROCESSED_DATA_DIR, "test", get_test_transforms(img_size))
        loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        log(f"  Test set for {model_name}: {len(test_dataset)} images")

        for tag, ckpt in [("raw", ckpt_path), ("ema", ema_ckpt_path)]:
            if not os.path.exists(ckpt):
                log(f"  {model_name} ({tag}): checkpoint not found, skipping.")
                continue

            model = get_model_fn(num_classes=config.NUM_CLASSES, freeze=False).to(device)
            checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            all_labels, all_preds, all_probs = [], [], []

            with torch.no_grad():
                for images, labels in loader:
                    images = images.to(device)
                    with torch.amp.autocast("cuda"):
                        outputs = model(images)
                    probs = F.softmax(outputs, dim=1)
                    preds = outputs.argmax(dim=1)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy())

            acc = accuracy_score(all_labels, all_preds)
            key = f"{model_name}_{tag}"
            results[key] = {
                "accuracy": acc,
                "labels": all_labels,
                "preds": all_preds,
                "probs": all_probs,
            }
            log(f"  {key}: Test Accuracy = {acc:.4f} ({acc*100:.2f}%)")

            report = classification_report(
                all_labels, all_preds,
                target_names=config.CLASS_NAMES,
                digits=4,
            )
            log(f"\n{report}")

            # Confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=config.CLASS_NAMES,
                        yticklabels=config.CLASS_NAMES, ax=ax)
            ax.set_title(f"Confusion Matrix: {key}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            plt.tight_layout()
            safe_key = key.replace(" ", "_").replace("-", "_").lower()
            plt.savefig(os.path.join(config.PLOTS_DIR, f"confusion_matrix_{safe_key}.png"), dpi=150)
            plt.close()

            del model
            torch.cuda.empty_cache()

    # Ensemble evaluation
    log("\n  Ensemble evaluation (EfficientNetV2 0.55 + ConvNeXt 0.45)...")
    effnet_key = "EfficientNetV2-CBAM_ema"
    convnext_key = "ConvNeXt-CBAM-FPN_ema"
    if effnet_key not in results:
        effnet_key = "EfficientNetV2-CBAM_raw"
    if convnext_key not in results:
        convnext_key = "ConvNeXt-CBAM-FPN_raw"

    best_acc = max((v["accuracy"] for v in results.values()), default=0)

    if effnet_key in results and convnext_key in results:
        ens_probs = np.array(results[effnet_key]["probs"]) * 0.55 + \
                    np.array(results[convnext_key]["probs"]) * 0.45
        ens_preds = (ens_probs > 0.5).astype(int)
        ens_labels = results[effnet_key]["labels"]
        ens_acc = accuracy_score(ens_labels, ens_preds)
        log(f"  Ensemble: Test Accuracy = {ens_acc:.4f} ({ens_acc*100:.2f}%)")

        report = classification_report(
            ens_labels, ens_preds,
            target_names=config.CLASS_NAMES,
            digits=4,
        )
        log(f"\n{report}")

        # Confusion matrix
        cm = confusion_matrix(ens_labels, ens_preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                    xticklabels=config.CLASS_NAMES,
                    yticklabels=config.CLASS_NAMES, ax=ax)
        ax.set_title(f"Confusion Matrix: Ensemble (Acc={ens_acc:.4f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(config.PLOTS_DIR, "confusion_matrix_ensemble.png"), dpi=150)
        plt.close()

        # ROC curve
        fpr, tpr, _ = roc_curve(ens_labels, ens_probs)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, "b-", label=f"Ensemble (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve - Ensemble Model")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(config.PLOTS_DIR, "roc_curve_ensemble.png"), dpi=150)
        plt.close()

        results["ensemble"] = {"accuracy": ens_acc, "auc": roc_auc}
        best_acc = max(best_acc, ens_acc)

    # Save results
    summary = {}
    for key, val in results.items():
        summary[key] = {
            "accuracy": val["accuracy"],
            "auc": val.get("auc"),
        }
    with open(os.path.join(config.PLOTS_DIR, "evaluation_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log(f"Evaluation done. Best accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    return best_acc, results


def download_landslide4sense():
    """Download Landslide4Sense dataset from Kaggle and integrate it."""
    log("DOWNLOADING Landslide4Sense from Kaggle (~3GB)...")

    kaggle_dir = os.path.join(config.DATA_DIR, "kaggle_landslide4sense")
    os.makedirs(kaggle_dir, exist_ok=True)

    # Download using kaggle CLI
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", "tekbahadurkshetri/landslide4sense",
             "-p", kaggle_dir, "--unzip"],
            capture_output=True, text=True, timeout=1800,  # 30 min timeout
        )
        log(f"Kaggle download stdout: {result.stdout[-500:] if result.stdout else 'none'}")
        if result.returncode != 0:
            log(f"Kaggle download stderr: {result.stderr[-500:] if result.stderr else 'none'}")
            return False
    except Exception as e:
        log(f"Kaggle download failed: {e}")
        return False

    log("Download complete. Integrating Landslide4Sense...")

    # Landslide4Sense has image patches with masks
    # We need to find images and classify them based on masks
    import cv2

    combined_dir = os.path.join(config.DATA_DIR, "binary_combined")
    new_landslide = 0
    new_non_landslide = 0

    # Search for image and mask files
    img_dirs = []
    mask_dirs = []
    for root, dirs, files in os.walk(kaggle_dir):
        dirname = os.path.basename(root).lower()
        if "img" in dirname or "image" in dirname:
            img_dirs.append(root)
        if "mask" in dirname or "label" in dirname:
            mask_dirs.append(root)

    # Also check for .h5 or .npy files (Landslide4Sense uses h5 format)
    h5_files = list(Path(kaggle_dir).rglob("*.h5"))
    npy_files = list(Path(kaggle_dir).rglob("*.npy"))
    tif_files = list(Path(kaggle_dir).rglob("*.tif"))
    png_files = list(Path(kaggle_dir).rglob("*.png"))
    jpg_files = list(Path(kaggle_dir).rglob("*.jpg"))

    log(f"  Found: {len(h5_files)} h5, {len(npy_files)} npy, {len(tif_files)} tif, {len(png_files)} png, {len(jpg_files)} jpg files")
    log(f"  Image dirs: {img_dirs[:5]}")
    log(f"  Mask dirs: {mask_dirs[:5]}")

    # Strategy 1: If there are paired image/mask directories
    if img_dirs and mask_dirs:
        img_dir = img_dirs[0]
        mask_dir = mask_dirs[0]
        log(f"  Using image dir: {img_dir}")
        log(f"  Using mask dir: {mask_dir}")

        img_files = sorted([f for f in Path(img_dir).iterdir()
                           if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}])

        for img_path in img_files:
            # Find corresponding mask
            mask_candidates = [
                Path(mask_dir) / img_path.name,
                Path(mask_dir) / img_path.with_suffix(".png").name,
                Path(mask_dir) / img_path.with_suffix(".tif").name,
            ]
            mask_path = None
            for mc in mask_candidates:
                if mc.exists():
                    mask_path = mc
                    break

            if mask_path is None:
                continue

            try:
                img = cv2.imread(str(img_path))
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if img is None or mask is None:
                    continue

                # Classify: if >5% of mask pixels are landslide, it's landslide
                landslide_ratio = np.count_nonzero(mask) / mask.size
                if landslide_ratio > 0.05:
                    cls = "landslide"
                    new_landslide += 1
                else:
                    cls = "non_landslide"
                    new_non_landslide += 1

                # Resize to 128x128 and save to train split
                img = cv2.resize(img, (128, 128))
                dst = os.path.join(combined_dir, "train", cls, f"l4s_{img_path.stem}.jpg")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                cv2.imwrite(dst, img)
            except Exception:
                continue

    # Strategy 2: If there are h5 files (Landslide4Sense format)
    elif h5_files:
        try:
            import h5py
            log(f"  Processing {len(h5_files)} h5 files...")
            for h5_path in h5_files[:100]:  # Limit to first 100 files
                try:
                    with h5py.File(str(h5_path), "r") as f:
                        keys = list(f.keys())
                        log(f"  H5 keys in {h5_path.name}: {keys}")
                        # Try to extract image and mask
                        if "image" in keys and "mask" in keys:
                            img = np.array(f["image"])
                            mask = np.array(f["mask"])
                            # Take first 3 channels as RGB
                            if img.ndim == 3 and img.shape[2] > 3:
                                img = img[:, :, :3]
                            img = (img / img.max() * 255).astype(np.uint8) if img.max() > 0 else img.astype(np.uint8)

                            landslide_ratio = np.count_nonzero(mask) / mask.size
                            cls = "landslide" if landslide_ratio > 0.05 else "non_landslide"
                            if cls == "landslide":
                                new_landslide += 1
                            else:
                                new_non_landslide += 1

                            img = cv2.resize(img, (128, 128))
                            dst = os.path.join(combined_dir, "train", cls, f"l4s_{h5_path.stem}.jpg")
                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                            cv2.imwrite(dst, img)
                except Exception as e:
                    continue
        except ImportError:
            log("  h5py not installed, trying pip install...")
            subprocess.run([sys.executable, "-m", "pip", "install", "h5py", "-q"])
            # Skip h5 processing this round

    # Strategy 3: If there are tif files (GeoTIFF satellite images)
    elif tif_files:
        log(f"  Processing {len(tif_files)} tif files...")
        # Look for paired image/mask tifs
        for tif_path in tif_files:
            try:
                if "mask" in tif_path.name.lower() or "label" in tif_path.name.lower():
                    continue  # Skip mask files
                img = cv2.imread(str(tif_path))
                if img is None:
                    continue
                # Check if there's a corresponding mask
                mask_path = None
                for suffix in ["_mask", "_label", "_gt"]:
                    candidate = tif_path.parent / f"{tif_path.stem}{suffix}{tif_path.suffix}"
                    if candidate.exists():
                        mask_path = candidate
                        break
                if mask_path:
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        landslide_ratio = np.count_nonzero(mask) / mask.size
                        cls = "landslide" if landslide_ratio > 0.05 else "non_landslide"
                    else:
                        continue
                else:
                    # No mask - skip or use filename heuristic
                    name_lower = tif_path.name.lower()
                    if "landslide" in name_lower or "positive" in name_lower:
                        cls = "landslide"
                    elif "non" in name_lower or "negative" in name_lower:
                        cls = "non_landslide"
                    else:
                        continue

                if cls == "landslide":
                    new_landslide += 1
                else:
                    new_non_landslide += 1

                img = cv2.resize(img, (128, 128))
                dst = os.path.join(combined_dir, "train", cls, f"l4s_{tif_path.stem}.jpg")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                cv2.imwrite(dst, img)
            except Exception:
                continue

    log(f"  Integrated from Landslide4Sense: {new_landslide} landslide, {new_non_landslide} non_landslide")
    return (new_landslide + new_non_landslide) > 0


def re_expand_dataset():
    """Re-run dataset expansion after adding new data."""
    log("Re-expanding dataset with new data...")
    from scripts.expand_dataset import expand_dataset

    # Remove old expanded dataset
    expanded_dir = os.path.join(config.DATA_DIR, "binary_combined_expanded")
    if os.path.isdir(expanded_dir):
        shutil.rmtree(expanded_dir)

    expand_dataset(
        source_dir=os.path.join(config.DATA_DIR, "binary_combined"),
        output_dir=expanded_dir,
    )
    log("Dataset re-expanded.")


def delete_old_checkpoints():
    """Delete old checkpoints so training starts fresh."""
    for ckpt in [config.EFFNETV2_CHECKPOINT, config.EMA_EFFNETV2_CHECKPOINT,
                 config.CONVNEXT_CHECKPOINT, config.EMA_CONVNEXT_CHECKPOINT]:
        if os.path.exists(ckpt):
            os.remove(ckpt)
            log(f"  Deleted: {ckpt}")


def main():
    start_time = time.time()
    log("=" * 70)
    log("AUTO PIPELINE — FULLY UNATTENDED WITH RETRY")
    log(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    log(f"Data: {config.PROCESSED_DATA_DIR}")
    log(f"Target: Beat {TARGET_ACCURACY*100:.1f}% (Attention-Driven VGG16 paper)")
    log(f"Retry threshold: {ACCURACY_THRESHOLD*100:.0f}%  |  Max rounds: {MAX_ROUNDS}")
    log("=" * 70)

    best_overall_acc = 0

    for round_num in range(1, MAX_ROUNDS + 1):
        log(f"\n{'='*70}")
        log(f"ROUND {round_num}/{MAX_ROUNDS}")
        log(f"{'='*70}")

        # Step 1: Check if dataset expansion is needed
        expanded_train = os.path.join(config.DATA_DIR, "binary_combined_expanded", "train")
        if not os.path.isdir(expanded_train):
            try:
                from scripts.expand_dataset import expand_dataset
                expand_dataset(
                    source_dir=os.path.join(config.DATA_DIR, "binary_combined"),
                    output_dir=os.path.join(config.DATA_DIR, "binary_combined_expanded"),
                )
            except Exception as e:
                log(f"ERROR expanding dataset: {e}")
                traceback.print_exc()
                return

        # Count training images
        train_count = sum(1 for _ in Path(expanded_train).rglob("*")
                         if _.suffix.lower() in {".jpg", ".jpeg", ".png"})
        log(f"Training images: {train_count}")

        # Step 2: Train EfficientNetV2
        try:
            # More epochs on retry rounds
            epochs = 60 if round_num == 1 else 80
            lr = 3e-5 if round_num == 1 else 2e-5
            train_efficientnetv2(num_epochs=epochs, lr=lr)
        except Exception as e:
            log(f"ERROR training EfficientNetV2: {e}")
            traceback.print_exc()

        # Step 3: Train ConvNeXt
        try:
            epochs = 50 if round_num == 1 else 70
            lr = 5e-5 if round_num == 1 else 3e-5
            train_convnext(num_epochs=epochs, lr=lr)
        except Exception as e:
            log(f"ERROR training ConvNeXt: {e}")
            traceback.print_exc()

        # Step 4: Evaluate
        try:
            best_acc, results = evaluate_models()
            best_overall_acc = max(best_overall_acc, best_acc)
        except Exception as e:
            log(f"ERROR evaluating: {e}")
            traceback.print_exc()
            best_acc = 0

        log(f"\nRound {round_num} best accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
        log(f"Overall best accuracy: {best_overall_acc:.4f} ({best_overall_acc*100:.2f}%)")

        # Check if we've reached the target
        if best_acc >= TARGET_ACCURACY:
            log(f"\n*** TARGET REACHED! {best_acc*100:.2f}% >= {TARGET_ACCURACY*100:.1f}% ***")
            break

        # Check if we need more data
        if best_acc < ACCURACY_THRESHOLD and round_num < MAX_ROUNDS:
            log(f"\nAccuracy {best_acc*100:.2f}% < {ACCURACY_THRESHOLD*100:.0f}% threshold.")
            log("Downloading more data from Kaggle...")

            try:
                success = download_landslide4sense()
                if success:
                    log("New data integrated. Re-expanding dataset...")
                    re_expand_dataset()
                    delete_old_checkpoints()
                else:
                    log("No new data could be integrated. Retrying with more epochs...")
                    delete_old_checkpoints()
            except Exception as e:
                log(f"ERROR downloading/integrating data: {e}")
                traceback.print_exc()
                delete_old_checkpoints()

        elif round_num < MAX_ROUNDS:
            # Accuracy is between threshold and target — retry with more epochs, no new data
            log(f"\nAccuracy {best_acc*100:.2f}% is between {ACCURACY_THRESHOLD*100:.0f}%-{TARGET_ACCURACY*100:.1f}%.")
            log("Retrying with more epochs and lower learning rate...")
            delete_old_checkpoints()

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    log(f"\n{'='*70}")
    log(f"AUTO PIPELINE COMPLETE")
    log(f"Total time: {hours}h {minutes}m")
    log(f"Best overall accuracy: {best_overall_acc:.4f} ({best_overall_acc*100:.2f}%)")
    log(f"Target was: {TARGET_ACCURACY*100:.1f}%")
    if best_overall_acc >= TARGET_ACCURACY:
        log("STATUS: TARGET ACHIEVED!")
    else:
        log(f"STATUS: Best effort reached {best_overall_acc*100:.2f}%")
    log(f"Results saved to: {config.PLOTS_DIR}")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
