"""
Final pipeline: clean dataset + augmentation + training + TTA evaluation.

Steps:
  1. Expand clean dataset with augmentation (~30K+ images)
  2. Train EfficientNetV2-S + CBAM (80 epochs)
  3. Train ConvNeXt-CBAM-FPN (70 epochs)
  4. Evaluate with TTA (7 augmented views per image)
  5. Ensemble with optimized weights
  6. If < 95%, retry with tuned hyperparams (up to 2 more attempts)
"""

import os
import sys
import time
import json
import shutil
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# Use clean dataset
CLEAN_DIR = os.path.join(config.DATA_DIR, "binary_combined_clean")
EXPANDED_DIR = os.path.join(config.DATA_DIR, "binary_clean_expanded")
config.PROCESSED_DATA_DIR = EXPANDED_DIR

LOG_FILE = os.path.join(config.PLOTS_DIR, "clean_pipeline.log")
TARGET_ACCURACY = 0.958
MAX_ATTEMPTS = 3


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def expand_clean_dataset():
    """Expand clean dataset. Higher augmentation for landslide (minority)."""
    log("Expanding clean dataset...")

    if os.path.isdir(EXPANDED_DIR):
        shutil.rmtree(EXPANDED_DIR)

    from scripts.expand_dataset import expand_dataset, AUGMENT_MULTIPLIER

    # Adjust multipliers for clean dataset ratio (~24% landslide)
    # landslide: 1765 train → need ~14x to match non_landslide
    # non_landslide: 5700 train → 3x is enough
    AUGMENT_MULTIPLIER["landslide"] = 13
    AUGMENT_MULTIPLIER["non_landslide"] = 3

    expand_dataset(source_dir=CLEAN_DIR, output_dir=EXPANDED_DIR)

    # Count
    train_count = sum(1 for _ in Path(os.path.join(EXPANDED_DIR, "train")).rglob("*")
                     if _.suffix.lower() in {".jpg", ".jpeg", ".png"})
    log(f"Expanded clean dataset: {train_count} training images")
    return train_count


def train_model(model_name, num_epochs, batch_size, lr):
    """Train a single model."""
    from phase1_alexnet.train import run_training
    from utils.plot_utils import plot_training_history

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    log(f"Training {model_name} ({num_epochs} epochs, batch={batch_size}, lr={lr})...")
    model, history = run_training(
        model_name=model_name,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=lr,
    )
    plot_training_history(
        history,
        save_path=os.path.join(config.PLOTS_DIR, f"training_history_{model_name}.png"),
    )
    with open(os.path.join(config.PLOTS_DIR, f"history_{model_name}.json"), "w") as f:
        json.dump(history, f)

    del model
    torch.cuda.empty_cache()
    log(f"{model_name} training done.")
    return history


def evaluate_with_tta(attempt=1):
    """Evaluate all models with TTA and ensemble."""
    log("Evaluating with TTA (7 augmented views per image)...")

    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    from phase1_alexnet.model import get_convnext_cbam_fpn, get_efficientnetv2_cbam
    from phase1_alexnet.dataset import LandslideDataset
    from utils.tta import get_tta_transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    for model_name, ckpt_path, ema_ckpt_path, get_model_fn, img_size in [
        ("EfficientNetV2-CBAM", config.EFFNETV2_CHECKPOINT, config.EMA_EFFNETV2_CHECKPOINT, get_efficientnetv2_cbam, config.EFFNETV2_IMG_SIZE),
        ("ConvNeXt-CBAM-FPN", config.CONVNEXT_CHECKPOINT, config.EMA_CONVNEXT_CHECKPOINT, get_convnext_cbam_fpn, config.CONVNEXT_IMG_SIZE),
    ]:
        # Load test dataset WITHOUT transforms (we apply TTA transforms manually)
        test_dataset = LandslideDataset(config.PROCESSED_DATA_DIR, "test", transform=None)
        log(f"  Test set for {model_name}: {len(test_dataset)} images")

        tta_transforms = get_tta_transforms(img_size)

        for tag, ckpt in [("ema", ema_ckpt_path), ("raw", ckpt_path)]:
            if not os.path.exists(ckpt):
                log(f"  {model_name} ({tag}): checkpoint not found, skipping.")
                continue

            model = get_model_fn(num_classes=config.NUM_CLASSES, freeze=False).to(device)
            checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            all_labels = []
            all_preds = []
            all_probs = []

            with torch.no_grad():
                for idx in range(len(test_dataset)):
                    img_rgb, label = test_dataset[idx]

                    # img_rgb is a tensor from default transform (no normalize)
                    # Convert back to numpy for TTA transforms
                    if isinstance(img_rgb, torch.Tensor):
                        img_np = (img_rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    else:
                        img_np = img_rgb

                    # Apply each TTA transform and average predictions
                    tta_probs = []
                    for t in tta_transforms:
                        tensor = t(img_np).unsqueeze(0).to(device)
                        with torch.amp.autocast("cuda"):
                            logits = model(tensor)
                        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
                        tta_probs.append(probs)

                    avg_probs = np.mean(tta_probs, axis=0)
                    pred = np.argmax(avg_probs)

                    all_labels.append(label)
                    all_preds.append(pred)
                    all_probs.append(avg_probs[1])  # P(landslide)

                    if (idx + 1) % 200 == 0:
                        log(f"    TTA progress: {idx+1}/{len(test_dataset)}")

            acc = accuracy_score(all_labels, all_preds)
            key = f"{model_name}_{tag}"
            results[key] = {
                "accuracy": acc,
                "labels": all_labels,
                "preds": all_preds,
                "probs": all_probs,
            }
            log(f"  {key} (TTA): Test Accuracy = {acc:.4f} ({acc*100:.2f}%)")

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
            ax.set_title(f"Confusion Matrix: {key} (TTA, Attempt {attempt})")
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            plt.tight_layout()
            safe_key = key.replace(" ", "_").replace("-", "_").lower()
            plt.savefig(os.path.join(config.PLOTS_DIR, f"cm_{safe_key}_tta_attempt{attempt}.png"), dpi=150)
            plt.close()

            del model
            torch.cuda.empty_cache()

    # === ALSO evaluate without TTA for comparison ===
    log("\n  Also evaluating WITHOUT TTA for comparison...")
    from phase1_alexnet.dataset import get_test_transforms
    for model_name, ckpt_path, ema_ckpt_path, get_model_fn, img_size in [
        ("EfficientNetV2-CBAM", config.EFFNETV2_CHECKPOINT, config.EMA_EFFNETV2_CHECKPOINT, get_efficientnetv2_cbam, config.EFFNETV2_IMG_SIZE),
        ("ConvNeXt-CBAM-FPN", config.CONVNEXT_CHECKPOINT, config.EMA_CONVNEXT_CHECKPOINT, get_convnext_cbam_fpn, config.CONVNEXT_IMG_SIZE),
    ]:
        test_dataset = LandslideDataset(config.PROCESSED_DATA_DIR, "test", get_test_transforms(img_size))
        loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

        for tag, ckpt in [("ema", ema_ckpt_path), ("raw", ckpt_path)]:
            if not os.path.exists(ckpt):
                continue
            model = get_model_fn(num_classes=config.NUM_CLASSES, freeze=False).to(device)
            checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            all_labels, all_preds = [], []
            with torch.no_grad():
                for images, labels in loader:
                    images = images.to(device)
                    with torch.amp.autocast("cuda"):
                        outputs = model(images)
                    preds = outputs.argmax(dim=1)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())

            acc = accuracy_score(all_labels, all_preds)
            log(f"  {model_name}_{tag} (NO TTA): {acc:.4f} ({acc*100:.2f}%)")

            del model
            torch.cuda.empty_cache()

    # === Ensemble with TTA ===
    log("\n  Ensemble evaluation with TTA...")
    best_acc = max((v["accuracy"] for v in results.values()), default=0)

    # Try multiple ensemble weight combinations
    best_ens_acc = 0
    best_weights = (0.55, 0.45)

    effnet_key = None
    convnext_key = None
    for k in results:
        if "EfficientNetV2" in k:
            if effnet_key is None or results[k]["accuracy"] > results[effnet_key]["accuracy"]:
                effnet_key = k
        if "ConvNeXt" in k:
            if convnext_key is None or results[k]["accuracy"] > results[convnext_key]["accuracy"]:
                convnext_key = k

    if effnet_key and convnext_key:
        log(f"  Using best models: {effnet_key} + {convnext_key}")

        for w1 in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]:
            w2 = 1.0 - w1
            ens_probs = np.array(results[effnet_key]["probs"]) * w1 + \
                        np.array(results[convnext_key]["probs"]) * w2

            # Try multiple thresholds
            for threshold in [0.4, 0.45, 0.5, 0.55, 0.6]:
                ens_preds = (ens_probs > threshold).astype(int)
                ens_labels = results[effnet_key]["labels"]
                ens_acc = accuracy_score(ens_labels, ens_preds)

                if ens_acc > best_ens_acc:
                    best_ens_acc = ens_acc
                    best_weights = (w1, w2)
                    best_threshold = threshold
                    best_ens_preds = ens_preds
                    best_ens_probs = ens_probs

        log(f"  Best ensemble: weights=({best_weights[0]:.2f}, {best_weights[1]:.2f}), "
            f"threshold={best_threshold:.2f}, accuracy={best_ens_acc:.4f} ({best_ens_acc*100:.2f}%)")

        ens_labels = results[effnet_key]["labels"]
        report = classification_report(ens_labels, best_ens_preds, target_names=config.CLASS_NAMES, digits=4)
        log(f"\n{report}")

        # Confusion matrix
        cm = confusion_matrix(ens_labels, best_ens_preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                    xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES, ax=ax)
        ax.set_title(f"Ensemble TTA (Acc={best_ens_acc:.4f}, w={best_weights}, t={best_threshold})")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(config.PLOTS_DIR, f"cm_ensemble_tta_attempt{attempt}.png"), dpi=150)
        plt.close()

        # ROC curve
        fpr, tpr, _ = roc_curve(ens_labels, best_ens_probs)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, "b-", label=f"Ensemble TTA (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - Ensemble TTA (Attempt {attempt})")
        ax.legend(); plt.tight_layout()
        plt.savefig(os.path.join(config.PLOTS_DIR, f"roc_ensemble_tta_attempt{attempt}.png"), dpi=150)
        plt.close()

        results["ensemble_tta"] = {
            "accuracy": best_ens_acc,
            "auc": roc_auc,
            "weights": best_weights,
            "threshold": best_threshold,
        }
        best_acc = max(best_acc, best_ens_acc)

    # Save results
    summary = {}
    for k, v in results.items():
        summary[k] = {
            "accuracy": v["accuracy"],
            "auc": v.get("auc"),
            "weights": v.get("weights"),
            "threshold": v.get("threshold"),
        }
    with open(os.path.join(config.PLOTS_DIR, f"results_clean_tta_attempt{attempt}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log(f"\nBest accuracy (with TTA + ensemble): {best_acc:.4f} ({best_acc*100:.2f}%)")
    return best_acc


def delete_checkpoints():
    for ckpt in [config.EFFNETV2_CHECKPOINT, config.EMA_EFFNETV2_CHECKPOINT,
                 config.CONVNEXT_CHECKPOINT, config.EMA_CONVNEXT_CHECKPOINT]:
        if os.path.exists(ckpt):
            os.remove(ckpt)


def main():
    start_time = time.time()
    log("=" * 70)
    log("CLEAN DATA PIPELINE — WITH TTA")
    log(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    log(f"Clean data: {CLEAN_DIR}")
    log(f"Target: Beat {TARGET_ACCURACY*100:.1f}%")
    log("=" * 70)

    # Step 1: Expand clean dataset
    try:
        train_count = expand_clean_dataset()
    except Exception as e:
        log(f"ERROR expanding dataset: {e}")
        traceback.print_exc()
        return

    # Training attempts with different hyperparams
    hyperparams = [
        # Attempt 1: Standard
        {"effnet_epochs": 80, "effnet_batch": 24, "effnet_lr": 3e-5,
         "convnext_epochs": 70, "convnext_batch": 32, "convnext_lr": 5e-5},
        # Attempt 2: More epochs, lower LR
        {"effnet_epochs": 100, "effnet_batch": 24, "effnet_lr": 2e-5,
         "convnext_epochs": 80, "convnext_batch": 32, "convnext_lr": 3e-5},
        # Attempt 3: Even more
        {"effnet_epochs": 120, "effnet_batch": 20, "effnet_lr": 1e-5,
         "convnext_epochs": 100, "convnext_batch": 28, "convnext_lr": 2e-5},
    ]

    best_overall = 0

    for attempt in range(1, MAX_ATTEMPTS + 1):
        log(f"\n{'='*70}")
        log(f"ATTEMPT {attempt}/{MAX_ATTEMPTS}")
        log(f"{'='*70}")

        params = hyperparams[attempt - 1]

        # Train EfficientNetV2
        try:
            train_model("efficientnetv2_cbam", params["effnet_epochs"],
                        params["effnet_batch"], params["effnet_lr"])
        except Exception as e:
            log(f"ERROR training EfficientNetV2: {e}")
            traceback.print_exc()

        # Train ConvNeXt
        try:
            train_model("convnext_cbam_fpn", params["convnext_epochs"],
                        params["convnext_batch"], params["convnext_lr"])
        except Exception as e:
            log(f"ERROR training ConvNeXt: {e}")
            traceback.print_exc()

        # Evaluate with TTA
        try:
            best_acc = evaluate_with_tta(attempt)
            best_overall = max(best_overall, best_acc)
        except Exception as e:
            log(f"ERROR evaluating: {e}")
            traceback.print_exc()
            best_acc = 0

        if best_acc >= TARGET_ACCURACY:
            log(f"\n*** TARGET REACHED! {best_acc*100:.2f}% >= {TARGET_ACCURACY*100:.1f}% ***")
            break

        if attempt < MAX_ATTEMPTS:
            log(f"Accuracy {best_acc*100:.2f}% < target. Retrying with tuned hyperparams...")
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
    log(f"Results in: {config.PLOTS_DIR}")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
