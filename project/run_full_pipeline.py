"""
Full unattended pipeline: dataset expansion → training → evaluation.

Run this and walk away. It handles everything:
  1. Expand combined dataset with augmentation (~33K images)
  2. Train EfficientNetV2-S + CBAM (60 epochs) — primary model to beat 95.8%
  3. Train ConvNeXt-CBAM-FPN (50 epochs) — ensemble partner
  4. Evaluate all models + ensemble on test set
  5. Generate plots, confusion matrices, classification reports
"""

import os
import sys
import time
import json
import traceback
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# Override config to use combined expanded dataset
config.PROCESSED_DATA_DIR = os.path.join(config.DATA_DIR, "binary_combined_expanded")


def log(msg):
    """Print timestamped log message."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def step1_expand_dataset():
    """Expand the combined dataset with augmentation."""
    log("STEP 1: Expanding combined dataset with augmentation...")
    from scripts.expand_dataset import expand_dataset
    expand_dataset(
        source_dir=os.path.join(config.DATA_DIR, "binary_combined"),
        output_dir=config.PROCESSED_DATA_DIR,
    )
    log("STEP 1 COMPLETE: Dataset expanded.")


def step2_train_efficientnetv2():
    """Train EfficientNetV2-S + CBAM — primary model to beat 95.8%."""
    log("STEP 2: Training EfficientNetV2-S + CBAM (60 epochs, batch=24, AMP)...")
    from phase1_alexnet.train import run_training
    from utils.plot_utils import plot_training_history

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # 384x384 images need smaller batch size; 60 epochs for thorough convergence
    model, history = run_training(
        model_name="efficientnetv2_cbam",
        num_epochs=60,
        batch_size=24,
        learning_rate=3e-5,
    )
    plot_training_history(
        history,
        save_path=os.path.join(config.PLOTS_DIR, "training_history_efficientnetv2.png"),
    )
    with open(os.path.join(config.PLOTS_DIR, "history_efficientnetv2.json"), "w") as f:
        json.dump(history, f)

    del model
    torch.cuda.empty_cache()
    log("STEP 2 COMPLETE: EfficientNetV2 training done.")
    return history


def step3_train_convnext():
    """Train ConvNeXt-CBAM-FPN model — ensemble partner."""
    log("STEP 3: Training ConvNeXt-CBAM-FPN (50 epochs, batch=32, AMP)...")
    from phase1_alexnet.train import run_training
    from utils.plot_utils import plot_training_history

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model, history = run_training(
        model_name="convnext_cbam_fpn",
        num_epochs=50,
        batch_size=32,
        learning_rate=5e-5,
    )
    plot_training_history(
        history,
        save_path=os.path.join(config.PLOTS_DIR, "training_history_convnext.png"),
    )
    with open(os.path.join(config.PLOTS_DIR, "history_convnext.json"), "w") as f:
        json.dump(history, f)

    del model
    torch.cuda.empty_cache()
    log("STEP 3 COMPLETE: ConvNeXt training done.")
    return history


def step4_evaluate():
    """Evaluate all models and ensemble on test set."""
    log("STEP 4: Evaluating models on test set...")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

    # Evaluate each model
    for model_name, ckpt_path, ema_ckpt_path, get_model_fn, img_size in [
        ("EfficientNetV2-CBAM", config.EFFNETV2_CHECKPOINT, config.EMA_EFFNETV2_CHECKPOINT, get_efficientnetv2_cbam, config.EFFNETV2_IMG_SIZE),
        ("ConvNeXt-CBAM-FPN", config.CONVNEXT_CHECKPOINT, config.EMA_CONVNEXT_CHECKPOINT, get_convnext_cbam_fpn, config.CONVNEXT_IMG_SIZE),
    ]:
        # Create test loader with correct image size
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

            all_labels = []
            all_preds = []
            all_probs = []

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

            # Classification report
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

    # Ensemble evaluation (EfficientNetV2 0.55 + ConvNeXt 0.45)
    log("\n  Ensemble evaluation (EfficientNetV2 0.55 + ConvNeXt 0.45)...")
    effnet_key = "EfficientNetV2-CBAM_ema"
    convnext_key = "ConvNeXt-CBAM-FPN_ema"

    # Fallback to raw if EMA not available
    if effnet_key not in results:
        effnet_key = "EfficientNetV2-CBAM_raw"
    if convnext_key not in results:
        convnext_key = "ConvNeXt-CBAM-FPN_raw"

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

        # Confusion matrix for ensemble
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
        ax.set_title("ROC Curve — Ensemble Model")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(config.PLOTS_DIR, "roc_curve_ensemble.png"), dpi=150)
        plt.close()

        results["ensemble"] = {"accuracy": ens_acc, "auc": roc_auc}

    # Save all results
    summary = {}
    for key, val in results.items():
        summary[key] = {
            "accuracy": val["accuracy"],
            "auc": val.get("auc"),
        }
    with open(os.path.join(config.PLOTS_DIR, "evaluation_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log("STEP 4 COMPLETE: Evaluation done.")
    return results


def main():
    start_time = time.time()
    log("=" * 70)
    log("FULL PIPELINE — UNATTENDED MODE (v2)")
    log(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    log(f"Data: {config.PROCESSED_DATA_DIR}")
    log("Target: Beat 95.8% (Attention-Driven VGG16 paper)")
    log("=" * 70)

    try:
        step1_expand_dataset()
    except Exception as e:
        log(f"ERROR in step 1: {e}")
        traceback.print_exc()
        return

    try:
        step2_train_efficientnetv2()
    except Exception as e:
        log(f"ERROR in step 2: {e}")
        traceback.print_exc()

    try:
        step3_train_convnext()
    except Exception as e:
        log(f"ERROR in step 3: {e}")
        traceback.print_exc()

    try:
        step4_evaluate()
    except Exception as e:
        log(f"ERROR in step 4: {e}")
        traceback.print_exc()

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    log(f"\n{'=' * 70}")
    log(f"PIPELINE COMPLETE — Total time: {hours}h {minutes}m")
    log(f"Results saved to: {config.PLOTS_DIR}")
    log(f"{'=' * 70}")


if __name__ == "__main__":
    main()
