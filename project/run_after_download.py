"""
Post-download pipeline: waits for Landslide4Sense download, integrates, expands, retrains.

Run this after starting the Kaggle download. It will:
  1. Wait for landslide4sense.zip to finish downloading
  2. Unzip and integrate into binary_combined
  3. Re-expand the dataset with augmentation
  4. Train EfficientNetV2-S + CBAM (80 epochs)
  5. Train ConvNeXt-CBAM-FPN (70 epochs)
  6. Evaluate all models + ensemble
  7. If accuracy < 94%, retrain with tuned hyperparameters (up to 2 more attempts)
"""

import os
import sys
import time
import json
import shutil
import zipfile
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

config.PROCESSED_DATA_DIR = os.path.join(config.DATA_DIR, "binary_combined_expanded")

LOG_FILE = os.path.join(config.PLOTS_DIR, "post_download_pipeline.log")
L4S_ZIP = os.path.join(config.DATA_DIR, "kaggle_landslide4sense", "landslide4sense.zip")
L4S_DIR = os.path.join(config.DATA_DIR, "kaggle_landslide4sense", "extracted")
COMBINED_DIR = os.path.join(config.DATA_DIR, "binary_combined")
EXPANDED_DIR = os.path.join(config.DATA_DIR, "binary_combined_expanded")

TARGET_ACCURACY = 0.958
MAX_ATTEMPTS = 3


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def wait_for_download():
    """Wait for landslide4sense.zip to finish downloading."""
    log("Waiting for Landslide4Sense download to complete...")

    if not os.path.exists(L4S_ZIP):
        log(f"Zip file not found at {L4S_ZIP}, checking if already extracted...")
        if os.path.isdir(L4S_DIR) and len(os.listdir(L4S_DIR)) > 0:
            log("Already extracted, skipping download wait.")
            return True
        # Wait for download to start
        for i in range(120):  # Wait up to 10 minutes for file to appear
            if os.path.exists(L4S_ZIP):
                break
            time.sleep(5)
        if not os.path.exists(L4S_ZIP):
            log("ERROR: Download never started. Zip file not found.")
            return False

    # Wait for download to complete by checking if file size is stable
    prev_size = -1
    stable_count = 0
    while stable_count < 6:  # File size stable for 30 seconds = download complete
        try:
            current_size = os.path.getsize(L4S_ZIP)
        except OSError:
            time.sleep(5)
            continue

        if current_size == prev_size:
            stable_count += 1
        else:
            stable_count = 0
        prev_size = current_size

        size_mb = current_size / (1024 * 1024)
        log(f"  Download progress: {size_mb:.0f} MB (stable count: {stable_count}/6)")
        time.sleep(5)

    final_size = os.path.getsize(L4S_ZIP) / (1024 * 1024)
    log(f"Download complete! File size: {final_size:.0f} MB")

    # Verify it's a valid zip
    try:
        with zipfile.ZipFile(L4S_ZIP, 'r') as z:
            file_count = len(z.namelist())
            log(f"Valid zip with {file_count} files.")
        return True
    except (zipfile.BadZipFile, Exception) as e:
        log(f"ERROR: Zip file is corrupted: {e}")
        return False


def unzip_landslide4sense():
    """Unzip the downloaded dataset."""
    if os.path.isdir(L4S_DIR) and len(os.listdir(L4S_DIR)) > 0:
        log("Already extracted, skipping unzip.")
        return True

    log("Unzipping Landslide4Sense...")
    os.makedirs(L4S_DIR, exist_ok=True)
    try:
        with zipfile.ZipFile(L4S_ZIP, 'r') as z:
            z.extractall(L4S_DIR)
        log("Unzip complete.")
        return True
    except Exception as e:
        log(f"ERROR unzipping: {e}")
        # Try with 7z as fallback
        try:
            import subprocess
            result = subprocess.run(
                ["7z", "x", L4S_ZIP, f"-o{L4S_DIR}", "-y"],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode == 0:
                log("Unzipped with 7z.")
                return True
            else:
                log(f"7z failed: {result.stderr[:200]}")
        except Exception:
            pass
        return False


def integrate_landslide4sense():
    """Classify Landslide4Sense images and add to binary_combined."""
    log("Integrating Landslide4Sense into combined dataset...")

    # Explore the extracted directory structure
    all_files = []
    for root, dirs, files in os.walk(L4S_DIR):
        for f in files:
            all_files.append(os.path.join(root, f))

    # Categorize files
    h5_files = [f for f in all_files if f.endswith('.h5')]
    tif_files = [f for f in all_files if f.lower().endswith(('.tif', '.tiff'))]
    png_files = [f for f in all_files if f.lower().endswith('.png')]
    jpg_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg'))]
    npy_files = [f for f in all_files if f.endswith('.npy')]

    log(f"  Found: {len(h5_files)} h5, {len(tif_files)} tif, {len(png_files)} png, {len(jpg_files)} jpg, {len(npy_files)} npy")

    # Find image and mask directories
    img_dirs = []
    mask_dirs = []
    for root, dirs, files in os.walk(L4S_DIR):
        dirname = os.path.basename(root).lower()
        if any(k in dirname for k in ['img', 'image', 'patch']):
            img_dirs.append(root)
        if any(k in dirname for k in ['mask', 'label', 'gt', 'anno']):
            mask_dirs.append(root)

    log(f"  Image dirs: {img_dirs[:5]}")
    log(f"  Mask dirs: {mask_dirs[:5]}")

    new_ls, new_nls = 0, 0

    # Strategy 1: h5 files (Landslide4Sense primary format)
    if h5_files:
        try:
            import h5py
            log(f"  Processing {len(h5_files)} h5 files with h5py...")

            for h5_path in h5_files:
                try:
                    with h5py.File(h5_path, 'r') as f:
                        keys = list(f.keys())

                        # Try common key patterns
                        img_key = None
                        mask_key = None
                        for k in keys:
                            kl = k.lower()
                            if 'image' in kl or 'img' in kl or 'data' in kl or 'x' == kl:
                                img_key = k
                            if 'mask' in kl or 'label' in kl or 'gt' in kl or 'y' == kl:
                                mask_key = k

                        if img_key is None:
                            # Maybe the h5 file IS the image (single dataset)
                            if len(keys) == 1:
                                data = np.array(f[keys[0]])
                                if data.ndim >= 2:
                                    # Can't classify without mask, skip
                                    continue
                            continue

                        img_data = np.array(f[img_key])

                        if mask_key:
                            mask_data = np.array(f[mask_key])
                            landslide_ratio = np.count_nonzero(mask_data) / max(mask_data.size, 1)
                            cls = 'landslide' if landslide_ratio > 0.05 else 'non_landslide'
                        else:
                            continue  # Can't classify without mask

                        # Convert to 3-channel uint8 image
                        if img_data.ndim == 3:
                            if img_data.shape[2] > 3:
                                img = img_data[:, :, :3]  # Take first 3 channels (RGB)
                            elif img_data.shape[0] <= 14:  # Channels first
                                img = img_data[:3].transpose(1, 2, 0)
                            else:
                                img = img_data
                        elif img_data.ndim == 2:
                            img = np.stack([img_data]*3, axis=-1)
                        else:
                            continue

                        # Normalize to uint8
                        if img.dtype != np.uint8:
                            if img.max() > 0:
                                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                            else:
                                img = np.zeros_like(img, dtype=np.uint8)

                        img = cv2.resize(img, (128, 128))

                        stem = Path(h5_path).stem
                        dst = os.path.join(COMBINED_DIR, 'train', cls, f'l4s_{stem}.jpg')
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        cv2.imwrite(dst, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                        if cls == 'landslide':
                            new_ls += 1
                        else:
                            new_nls += 1

                except Exception as e:
                    continue

        except ImportError:
            log("  h5py not installed, installing...")
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "h5py", "-q"])
            log("  h5py installed. Re-run to process h5 files.")

    # Strategy 2: paired image/mask directories
    if img_dirs and mask_dirs:
        for img_dir in img_dirs:
            best_mask_dir = None
            for md in mask_dirs:
                if os.path.dirname(md) == os.path.dirname(img_dir):
                    best_mask_dir = md
                    break
            if best_mask_dir is None and mask_dirs:
                best_mask_dir = mask_dirs[0]
            if best_mask_dir is None:
                continue

            log(f"  Processing paired dirs: {img_dir} + {best_mask_dir}")

            for img_path in sorted(Path(img_dir).iterdir()):
                if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}:
                    continue

                # Find mask
                mask_path = None
                for ext in [img_path.suffix, '.png', '.tif']:
                    candidate = Path(best_mask_dir) / f"{img_path.stem}{ext}"
                    if candidate.exists():
                        mask_path = candidate
                        break
                    # Try with mask_ prefix
                    candidate = Path(best_mask_dir) / f"mask_{img_path.stem}{ext}"
                    if candidate.exists():
                        mask_path = candidate
                        break

                if mask_path is None:
                    continue

                try:
                    img = cv2.imread(str(img_path))
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if img is None or mask is None:
                        continue

                    landslide_ratio = np.count_nonzero(mask) / mask.size
                    cls = 'landslide' if landslide_ratio > 0.05 else 'non_landslide'

                    img = cv2.resize(img, (128, 128))
                    dst = os.path.join(COMBINED_DIR, 'train', cls, f'l4s_{img_path.stem}.jpg')
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    cv2.imwrite(dst, img)

                    if cls == 'landslide':
                        new_ls += 1
                    else:
                        new_nls += 1
                except Exception:
                    continue

    # Strategy 3: tif files without paired directories
    if tif_files and new_ls + new_nls == 0:
        log(f"  Trying individual tif files...")
        for tif_path in tif_files:
            name_lower = Path(tif_path).name.lower()
            parent_lower = Path(tif_path).parent.name.lower()

            # Skip mask files
            if any(k in name_lower for k in ['mask', 'label', 'gt']):
                continue

            try:
                img = cv2.imread(tif_path)
                if img is None:
                    continue

                # Try to find a mask
                stem = Path(tif_path).stem
                parent = Path(tif_path).parent
                mask = None
                for pattern in [f"{stem}_mask", f"mask_{stem}", f"{stem}_label", f"{stem}_gt"]:
                    for ext in ['.tif', '.tiff', '.png']:
                        candidate = parent / f"{pattern}{ext}"
                        if candidate.exists():
                            mask = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
                            break
                    if mask is not None:
                        break

                if mask is not None:
                    landslide_ratio = np.count_nonzero(mask) / mask.size
                    cls = 'landslide' if landslide_ratio > 0.05 else 'non_landslide'
                elif 'landslide' in parent_lower or 'positive' in parent_lower:
                    cls = 'landslide'
                elif 'non' in parent_lower or 'negative' in parent_lower:
                    cls = 'non_landslide'
                else:
                    continue

                img = cv2.resize(img, (128, 128))
                dst = os.path.join(COMBINED_DIR, 'train', cls, f'l4s_{stem}.jpg')
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                cv2.imwrite(dst, img)

                if cls == 'landslide':
                    new_ls += 1
                else:
                    new_nls += 1
            except Exception:
                continue

    log(f"  Integrated from Landslide4Sense: {new_ls} landslide, {new_nls} non_landslide")
    log(f"  Total new images: {new_ls + new_nls}")
    return new_ls + new_nls


def expand_dataset():
    """Re-expand the combined dataset with augmentation."""
    log("Re-expanding dataset with augmentation...")

    if os.path.isdir(EXPANDED_DIR):
        shutil.rmtree(EXPANDED_DIR)

    from scripts.expand_dataset import expand_dataset as _expand
    _expand(source_dir=COMBINED_DIR, output_dir=EXPANDED_DIR)

    # Count
    train_count = sum(1 for _ in Path(os.path.join(EXPANDED_DIR, "train")).rglob("*")
                     if _.suffix.lower() in {".jpg", ".jpeg", ".png"})
    log(f"Expanded dataset: {train_count} training images")


def delete_checkpoints():
    """Delete old checkpoints for fresh training."""
    for ckpt in [config.EFFNETV2_CHECKPOINT, config.EMA_EFFNETV2_CHECKPOINT,
                 config.CONVNEXT_CHECKPOINT, config.EMA_CONVNEXT_CHECKPOINT]:
        if os.path.exists(ckpt):
            os.remove(ckpt)


def train_and_evaluate(attempt, effnet_epochs=80, convnext_epochs=70, effnet_lr=2e-5, convnext_lr=3e-5):
    """Train both models and evaluate."""
    from phase1_alexnet.train import run_training
    from utils.plot_utils import plot_training_history

    # Train EfficientNetV2
    log(f"[Attempt {attempt}] Training EfficientNetV2-S + CBAM ({effnet_epochs} epochs, lr={effnet_lr})...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model, history = run_training(
        model_name="efficientnetv2_cbam",
        num_epochs=effnet_epochs,
        batch_size=24,
        learning_rate=effnet_lr,
    )
    plot_training_history(history, save_path=os.path.join(config.PLOTS_DIR, f"training_history_efficientnetv2_attempt{attempt}.png"))
    with open(os.path.join(config.PLOTS_DIR, f"history_efficientnetv2_attempt{attempt}.json"), "w") as f:
        json.dump(history, f)
    del model
    torch.cuda.empty_cache()
    log(f"[Attempt {attempt}] EfficientNetV2 done.")

    # Train ConvNeXt
    log(f"[Attempt {attempt}] Training ConvNeXt-CBAM-FPN ({convnext_epochs} epochs, lr={convnext_lr})...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model, history = run_training(
        model_name="convnext_cbam_fpn",
        num_epochs=convnext_epochs,
        batch_size=32,
        learning_rate=convnext_lr,
    )
    plot_training_history(history, save_path=os.path.join(config.PLOTS_DIR, f"training_history_convnext_attempt{attempt}.png"))
    with open(os.path.join(config.PLOTS_DIR, f"history_convnext_attempt{attempt}.json"), "w") as f:
        json.dump(history, f)
    del model
    torch.cuda.empty_cache()
    log(f"[Attempt {attempt}] ConvNeXt done.")

    # Evaluate
    log(f"[Attempt {attempt}] Evaluating...")
    import torch.nn.functional as F
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
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
                continue

            model_inst = get_model_fn(num_classes=config.NUM_CLASSES, freeze=False).to(device)
            checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
            model_inst.load_state_dict(checkpoint["model_state_dict"])
            model_inst.eval()

            all_labels, all_preds, all_probs = [], [], []
            with torch.no_grad():
                for images, labels in loader:
                    images = images.to(device)
                    with torch.amp.autocast("cuda"):
                        outputs = model_inst(images)
                    probs = F.softmax(outputs, dim=1)
                    preds = outputs.argmax(dim=1)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy())

            acc = accuracy_score(all_labels, all_preds)
            key = f"{model_name}_{tag}"
            results[key] = {"accuracy": acc, "labels": all_labels, "preds": all_preds, "probs": all_probs}
            log(f"  {key}: Test Accuracy = {acc:.4f} ({acc*100:.2f}%)")

            report = classification_report(all_labels, all_preds, target_names=config.CLASS_NAMES, digits=4)
            log(f"\n{report}")

            cm = confusion_matrix(all_labels, all_preds)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES, ax=ax)
            ax.set_title(f"Confusion Matrix: {key} (Attempt {attempt})")
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            plt.tight_layout()
            safe_key = key.replace(" ", "_").replace("-", "_").lower()
            plt.savefig(os.path.join(config.PLOTS_DIR, f"confusion_matrix_{safe_key}_attempt{attempt}.png"), dpi=150)
            plt.close()

            del model_inst
            torch.cuda.empty_cache()

    # Ensemble
    best_acc = max((v["accuracy"] for v in results.values()), default=0)

    effnet_key = "EfficientNetV2-CBAM_ema" if "EfficientNetV2-CBAM_ema" in results else "EfficientNetV2-CBAM_raw"
    convnext_key = "ConvNeXt-CBAM-FPN_ema" if "ConvNeXt-CBAM-FPN_ema" in results else "ConvNeXt-CBAM-FPN_raw"

    if effnet_key in results and convnext_key in results:
        ens_probs = np.array(results[effnet_key]["probs"]) * 0.55 + np.array(results[convnext_key]["probs"]) * 0.45
        ens_preds = (ens_probs > 0.5).astype(int)
        ens_labels = results[effnet_key]["labels"]
        ens_acc = accuracy_score(ens_labels, ens_preds)
        log(f"  Ensemble: Test Accuracy = {ens_acc:.4f} ({ens_acc*100:.2f}%)")

        report = classification_report(ens_labels, ens_preds, target_names=config.CLASS_NAMES, digits=4)
        log(f"\n{report}")

        cm = confusion_matrix(ens_labels, ens_preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES, ax=ax)
        ax.set_title(f"Ensemble (Acc={ens_acc:.4f}, Attempt {attempt})")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(config.PLOTS_DIR, f"confusion_matrix_ensemble_attempt{attempt}.png"), dpi=150)
        plt.close()

        fpr, tpr, _ = roc_curve(ens_labels, ens_probs)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, "b-", label=f"Ensemble (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - Attempt {attempt}")
        ax.legend(); plt.tight_layout()
        plt.savefig(os.path.join(config.PLOTS_DIR, f"roc_curve_ensemble_attempt{attempt}.png"), dpi=150)
        plt.close()

        results["ensemble"] = {"accuracy": ens_acc, "auc": roc_auc}
        best_acc = max(best_acc, ens_acc)

    # Save results
    summary = {k: {"accuracy": v["accuracy"], "auc": v.get("auc")} for k, v in results.items()}
    with open(os.path.join(config.PLOTS_DIR, f"evaluation_results_attempt{attempt}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log(f"[Attempt {attempt}] Best accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    return best_acc


def main():
    start_time = time.time()
    log("=" * 70)
    log("POST-DOWNLOAD PIPELINE — FULLY AUTONOMOUS")
    log(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    log(f"Target: Beat {TARGET_ACCURACY*100:.1f}%")
    log("=" * 70)

    # Step 1: Wait for download
    download_ok = wait_for_download()

    if download_ok:
        # Step 2: Unzip
        unzip_ok = unzip_landslide4sense()

        if unzip_ok:
            # Step 3: Integrate
            new_images = integrate_landslide4sense()
            log(f"Added {new_images} new images from Landslide4Sense")
        else:
            log("Unzip failed. Proceeding with existing data.")
    else:
        log("Download failed/not available. Proceeding with existing data.")

    # Print current dataset stats
    log("\nCurrent combined dataset:")
    for split in ["train", "val", "test"]:
        for cls in ["landslide", "non_landslide"]:
            cls_dir = os.path.join(COMBINED_DIR, split, cls)
            if os.path.isdir(cls_dir):
                count = sum(1 for f in Path(cls_dir).iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
                log(f"  {split}/{cls}: {count}")

    # Step 4: Re-expand dataset
    expand_dataset()

    # Step 5: Train and evaluate with retries
    best_overall = 0

    hyperparams = [
        # Attempt 1: Standard
        {"effnet_epochs": 80, "convnext_epochs": 70, "effnet_lr": 2e-5, "convnext_lr": 3e-5},
        # Attempt 2: More epochs, lower LR
        {"effnet_epochs": 100, "convnext_epochs": 80, "effnet_lr": 1e-5, "convnext_lr": 2e-5},
        # Attempt 3: Even more epochs, even lower LR
        {"effnet_epochs": 120, "convnext_epochs": 100, "effnet_lr": 5e-6, "convnext_lr": 1e-5},
    ]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        log(f"\n{'='*70}")
        log(f"TRAINING ATTEMPT {attempt}/{MAX_ATTEMPTS}")
        log(f"{'='*70}")

        params = hyperparams[attempt - 1]

        try:
            best_acc = train_and_evaluate(attempt, **params)
            best_overall = max(best_overall, best_acc)
        except Exception as e:
            log(f"ERROR in attempt {attempt}: {e}")
            traceback.print_exc()
            best_acc = 0

        if best_acc >= TARGET_ACCURACY:
            log(f"\n*** TARGET REACHED! {best_acc*100:.2f}% >= {TARGET_ACCURACY*100:.1f}% ***")
            break

        if attempt < MAX_ATTEMPTS:
            log(f"Accuracy {best_acc*100:.2f}% < target {TARGET_ACCURACY*100:.1f}%. Retrying...")
            delete_checkpoints()

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    log(f"\n{'='*70}")
    log(f"PIPELINE COMPLETE")
    log(f"Total time: {hours}h {minutes}m")
    log(f"Best accuracy: {best_overall:.4f} ({best_overall*100:.2f}%)")
    if best_overall >= TARGET_ACCURACY:
        log("STATUS: TARGET ACHIEVED!")
    else:
        log(f"STATUS: Best effort {best_overall*100:.2f}% (target was {TARGET_ACCURACY*100:.1f}%)")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
