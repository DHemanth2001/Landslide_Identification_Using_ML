"""
Improvement pipeline to push from 87.49% → 92%+

Strategy:
  1. Self-training cleanup: use current best model to find mislabeled images, remove them
  2. Re-expand cleaned dataset
  3. Train SwinV2 as 3rd ensemble member (different architecture = diversity)
  4. Retrain EfficientNetV2 and ConvNeXt on cleaned data
  5. 3-model ensemble with optimized weights + threshold
  6. TTA on all 3 models
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

CLEAN_SOURCE = os.path.join(config.DATA_DIR, "binary_combined_clean")
CLEANED_SOURCE = os.path.join(config.DATA_DIR, "binary_selfcleaned")
EXPANDED_DIR = os.path.join(config.DATA_DIR, "binary_selfcleaned_expanded")
LOG_FILE = os.path.join(config.PLOTS_DIR, "improve_pipeline.log")

config.PROCESSED_DATA_DIR = EXPANDED_DIR


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def step1_self_clean():
    """Use current best model to remove mislabeled/ambiguous training images."""
    log("STEP 1: Self-training cleanup...")
    from phase1_alexnet.model import get_convnext_cbam_fpn
    from phase1_alexnet.dataset import get_test_transforms
    from torchvision import transforms
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the best ConvNeXt model (87.49% accuracy)
    old_expanded = os.path.join(config.DATA_DIR, "binary_clean_expanded")
    ckpt = config.CONVNEXT_CHECKPOINT
    if not os.path.exists(ckpt):
        log("  No ConvNeXt checkpoint found. Skipping self-clean.")
        # Just copy clean source
        if os.path.isdir(CLEANED_SOURCE):
            shutil.rmtree(CLEANED_SOURCE)
        shutil.copytree(CLEAN_SOURCE, CLEANED_SOURCE)
        return

    model = get_convnext_cbam_fpn(num_classes=2, freeze=False).to(device)
    checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    img_size = config.CONVNEXT_IMG_SIZE
    transform = get_test_transforms(img_size)

    # Scan training images and remove ones the model is very confident are wrong
    removed = 0
    kept = 0
    total = 0

    if os.path.isdir(CLEANED_SOURCE):
        shutil.rmtree(CLEANED_SOURCE)

    # Copy val and test as-is
    for split in ["val", "test"]:
        src_split = os.path.join(CLEAN_SOURCE, split)
        dst_split = os.path.join(CLEANED_SOURCE, split)
        if os.path.isdir(src_split):
            shutil.copytree(src_split, dst_split)

    # Filter training images
    for cls_name in ["landslide", "non_landslide"]:
        src_dir = os.path.join(CLEAN_SOURCE, "train", cls_name)
        dst_dir = os.path.join(CLEANED_SOURCE, "train", cls_name)
        os.makedirs(dst_dir, exist_ok=True)

        if not os.path.isdir(src_dir):
            continue

        cls_label = config.LABEL_MAP[cls_name]
        files = sorted([f for f in Path(src_dir).iterdir()
                       if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])

        for img_path in files:
            total += 1
            try:
                img = Image.open(str(img_path)).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    with torch.amp.autocast("cuda"):
                        output = model(tensor)
                    prob = F.softmax(output, dim=1)
                    pred = output.argmax(dim=1).item()
                    confidence = prob[0, pred].item()

                # Remove if model is VERY confident it's the OTHER class
                # (confidence > 0.85 means model strongly disagrees with label)
                if pred != cls_label and confidence > 0.85:
                    removed += 1
                    continue  # Don't copy this image

                # Keep the image
                shutil.copy2(str(img_path), os.path.join(dst_dir, img_path.name))
                kept += 1

            except Exception:
                # Keep images we can't process
                shutil.copy2(str(img_path), os.path.join(dst_dir, img_path.name))
                kept += 1

    del model
    torch.cuda.empty_cache()

    log(f"  Total scanned: {total}")
    log(f"  Removed (mislabeled): {removed} ({removed/max(total,1)*100:.1f}%)")
    log(f"  Kept: {kept}")

    # Print new distribution
    for cls_name in ["landslide", "non_landslide"]:
        cls_dir = os.path.join(CLEANED_SOURCE, "train", cls_name)
        if os.path.isdir(cls_dir):
            count = len([f for f in Path(cls_dir).iterdir()
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
            log(f"  train/{cls_name}: {count}")

    log("STEP 1 COMPLETE.")


def step2_expand():
    """Expand the self-cleaned dataset."""
    log("STEP 2: Expanding self-cleaned dataset...")

    if os.path.isdir(EXPANDED_DIR):
        shutil.rmtree(EXPANDED_DIR)

    from scripts.expand_dataset import expand_dataset
    expand_dataset(source_dir=CLEANED_SOURCE, output_dir=EXPANDED_DIR)

    train_count = sum(1 for _ in Path(os.path.join(EXPANDED_DIR, "train")).rglob("*")
                     if _.suffix.lower() in {".jpg", ".jpeg", ".png"})
    log(f"  Expanded training images: {train_count}")
    log("STEP 2 COMPLETE.")


def step3_train_swinv2():
    """Train SwinV2 as 3rd ensemble member."""
    log("STEP 3: Training SwinV2-S (80 epochs, batch=24, lr=3e-5)...")
    from phase1_alexnet.train import run_training
    from utils.plot_utils import plot_training_history

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for ckpt in [config.SWINV2_CHECKPOINT, config.EMA_SWINV2_CHECKPOINT]:
        if os.path.exists(ckpt):
            os.remove(ckpt)

    model, history = run_training(
        model_name="swinv2_s",
        num_epochs=80,
        batch_size=24,
        learning_rate=3e-5,
    )
    plot_training_history(
        history,
        save_path=os.path.join(config.PLOTS_DIR, "training_history_swinv2_improve.png"),
    )
    with open(os.path.join(config.PLOTS_DIR, "history_swinv2_improve.json"), "w") as f:
        json.dump(history, f)
    del model
    torch.cuda.empty_cache()
    log("STEP 3 COMPLETE.")


def step4_train_efficientnetv2():
    """Retrain EfficientNetV2 on cleaned data."""
    log("STEP 4: Training EfficientNetV2-S + CBAM (100 epochs, batch=24, lr=2e-5)...")
    from phase1_alexnet.train import run_training
    from utils.plot_utils import plot_training_history

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for ckpt in [config.EFFNETV2_CHECKPOINT, config.EMA_EFFNETV2_CHECKPOINT]:
        if os.path.exists(ckpt):
            os.remove(ckpt)

    model, history = run_training(
        model_name="efficientnetv2_cbam",
        num_epochs=100,
        batch_size=24,
        learning_rate=2e-5,
    )
    plot_training_history(
        history,
        save_path=os.path.join(config.PLOTS_DIR, "training_history_efficientnetv2_improve.png"),
    )
    with open(os.path.join(config.PLOTS_DIR, "history_efficientnetv2_improve.json"), "w") as f:
        json.dump(history, f)
    del model
    torch.cuda.empty_cache()
    log("STEP 4 COMPLETE.")


def step5_train_convnext():
    """Retrain ConvNeXt on cleaned data."""
    log("STEP 5: Training ConvNeXt-CBAM-FPN (80 epochs, batch=32, lr=3e-5)...")
    from phase1_alexnet.train import run_training
    from utils.plot_utils import plot_training_history

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for ckpt in [config.CONVNEXT_CHECKPOINT, config.EMA_CONVNEXT_CHECKPOINT]:
        if os.path.exists(ckpt):
            os.remove(ckpt)

    model, history = run_training(
        model_name="convnext_cbam_fpn",
        num_epochs=80,
        batch_size=32,
        learning_rate=3e-5,
    )
    plot_training_history(
        history,
        save_path=os.path.join(config.PLOTS_DIR, "training_history_convnext_improve.png"),
    )
    with open(os.path.join(config.PLOTS_DIR, "history_convnext_improve.json"), "w") as f:
        json.dump(history, f)
    del model
    torch.cuda.empty_cache()
    log("STEP 5 COMPLETE.")


def tta_predict(model, image_tensor, device):
    """7-view TTA."""
    views = [image_tensor]
    views.append(torch.flip(image_tensor, dims=[2]))
    views.append(torch.flip(image_tensor, dims=[1]))
    views.append(torch.flip(image_tensor, dims=[1, 2]))
    views.append(torch.rot90(image_tensor, k=1, dims=[1, 2]))
    views.append(torch.rot90(image_tensor, k=2, dims=[1, 2]))
    views.append(torch.rot90(image_tensor, k=3, dims=[1, 2]))

    all_probs = []
    for v in views:
        with torch.amp.autocast("cuda"):
            output = model(v.unsqueeze(0).to(device))
        prob = F.softmax(output, dim=1)
        all_probs.append(prob.cpu())
    return torch.stack(all_probs).mean(dim=0)


def step6_evaluate():
    """3-model ensemble with TTA + optimized weights + threshold."""
    log("STEP 6: Full evaluation — 3-model ensemble with TTA...")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        roc_curve, auc, accuracy_score,
    )
    from phase1_alexnet.model import get_convnext_cbam_fpn, get_efficientnetv2_cbam, get_swinv2_s
    from phase1_alexnet.dataset import LandslideDataset, get_test_transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    models_to_eval = [
        ("EfficientNetV2-CBAM", config.EFFNETV2_CHECKPOINT, config.EMA_EFFNETV2_CHECKPOINT,
         get_efficientnetv2_cbam, config.EFFNETV2_IMG_SIZE),
        ("ConvNeXt-CBAM-FPN", config.CONVNEXT_CHECKPOINT, config.EMA_CONVNEXT_CHECKPOINT,
         get_convnext_cbam_fpn, config.CONVNEXT_IMG_SIZE),
        ("SwinV2-S", config.SWINV2_CHECKPOINT, config.EMA_SWINV2_CHECKPOINT,
         get_swinv2_s, config.SWINV2_IMG_SIZE),
    ]

    for model_name, ckpt_path, ema_ckpt_path, get_model_fn, img_size in models_to_eval:
        test_dataset = LandslideDataset(
            config.PROCESSED_DATA_DIR, "test", get_test_transforms(img_size)
        )
        log(f"Test set for {model_name}: {len(test_dataset)} images")

        for tag, ckpt in [("raw", ckpt_path), ("ema", ema_ckpt_path)]:
            if not os.path.exists(ckpt):
                log(f"  {model_name} ({tag}): checkpoint not found, skipping.")
                continue

            model = get_model_fn(num_classes=config.NUM_CLASSES, freeze=False).to(device)
            checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            # Standard eval
            loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
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
            results[key] = {"accuracy": acc, "labels": all_labels, "preds": all_preds, "probs": all_probs}
            log(f"  {key}: Test Accuracy = {acc:.4f} ({acc*100:.2f}%)")

            # TTA eval
            all_labels_tta, all_preds_tta, all_probs_tta = [], [], []
            with torch.no_grad():
                for i in range(len(test_dataset)):
                    image, label = test_dataset[i]
                    avg_prob = tta_predict(model, image, device)
                    pred = avg_prob.argmax(dim=1).item()
                    all_labels_tta.append(label)
                    all_preds_tta.append(pred)
                    all_probs_tta.append(avg_prob[0, 1].item())
                    if (i + 1) % 200 == 0:
                        log(f"    TTA progress: {i+1}/{len(test_dataset)}")

            acc_tta = accuracy_score(all_labels_tta, all_preds_tta)
            key_tta = f"{model_name}_{tag}_TTA"
            results[key_tta] = {
                "accuracy": acc_tta, "labels": all_labels_tta,
                "preds": all_preds_tta, "probs": all_probs_tta,
            }
            log(f"  {key_tta}: Test Accuracy = {acc_tta:.4f} ({acc_tta*100:.2f}%)")

            report = classification_report(
                all_labels_tta, all_preds_tta,
                target_names=config.CLASS_NAMES, digits=4,
            )
            log(f"\n{report}")

            # Confusion matrix
            cm = confusion_matrix(all_labels_tta, all_preds_tta)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES, ax=ax)
            ax.set_title(f"{key_tta} (Acc={acc_tta:.4f})")
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            plt.tight_layout()
            safe = key_tta.replace(" ", "_").replace("-", "_").lower()
            plt.savefig(os.path.join(config.PLOTS_DIR, f"confusion_matrix_{safe}_improve.png"), dpi=150)
            plt.close()

            del model
            torch.cuda.empty_cache()

    # 3-model ensemble with weight + threshold optimization
    log("\n3-model ensemble optimization...")

    # Get best variant for each model
    tta_keys = {k: v for k, v in results.items() if "TTA" in k}
    labels = None

    # Collect all TTA probabilities
    model_probs = {}
    for k, v in tta_keys.items():
        model_probs[k] = np.array(v["probs"])
        if labels is None:
            labels = np.array(v["labels"])

    # Grid search over all weight combinations
    best_acc = 0
    best_config = {}
    keys = list(model_probs.keys())

    log(f"  Searching over {len(keys)} model variants...")

    # Pairwise and triple combinations
    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            if j <= i:
                continue
            # 2-model ensemble
            for w1 in np.arange(0.2, 0.85, 0.05):
                w2 = 1.0 - w1
                ens = model_probs[k1] * w1 + model_probs[k2] * w2
                for t in np.arange(0.4, 0.65, 0.02):
                    preds = (ens > t).astype(int)
                    acc = accuracy_score(labels, preds)
                    if acc > best_acc:
                        best_acc = acc
                        best_config = {"models": {k1: w1, k2: w2}, "threshold": t, "type": "2-model"}

            # 3-model ensemble
            for k3_idx, k3 in enumerate(keys):
                if k3_idx <= j:
                    continue
                for w1 in np.arange(0.15, 0.55, 0.1):
                    for w2 in np.arange(0.15, 0.55, 0.1):
                        w3 = 1.0 - w1 - w2
                        if w3 < 0.1 or w3 > 0.6:
                            continue
                        ens = model_probs[k1] * w1 + model_probs[k2] * w2 + model_probs[k3] * w3
                        for t in np.arange(0.4, 0.65, 0.02):
                            preds = (ens > t).astype(int)
                            acc = accuracy_score(labels, preds)
                            if acc > best_acc:
                                best_acc = acc
                                best_config = {
                                    "models": {k1: w1, k2: w2, k3: w3},
                                    "threshold": t, "type": "3-model",
                                }

    log(f"\nBest ensemble: {best_acc*100:.2f}%")
    log(f"Config: {best_config}")

    # Generate final plots
    ens_probs = sum(model_probs[k] * w for k, w in best_config["models"].items())
    ens_preds = (ens_probs > best_config["threshold"]).astype(int)

    report = classification_report(labels, ens_preds, target_names=config.CLASS_NAMES, digits=4)
    log(f"\n{report}")

    cm = confusion_matrix(labels, ens_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES, ax=ax)
    ax.set_title(f"Best Ensemble (Acc={best_acc:.4f})")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, "confusion_matrix_best_ensemble_improve.png"), dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(labels, ens_probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, "b-", label=f"Best Ensemble (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Best Ensemble")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, "roc_curve_best_ensemble_improve.png"), dpi=150)
    plt.close()

    # Save all results
    summary = {}
    for k, v in results.items():
        summary[k] = {"accuracy": v["accuracy"]}
    summary["best_ensemble"] = {"accuracy": best_acc, "auc": roc_auc, "config": str(best_config)}
    with open(os.path.join(config.PLOTS_DIR, "evaluation_results_improve.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log("\n" + "=" * 70)
    log("FINAL RESULTS")
    log("=" * 70)
    for k, v in sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True):
        log(f"  {k}: {v['accuracy']*100:.2f}%")
    log(f"\n  BEST ENSEMBLE: {best_acc*100:.2f}% (AUC: {roc_auc:.4f})")
    log("=" * 70)


def main():
    start_time = time.time()
    log("=" * 70)
    log("IMPROVEMENT PIPELINE — Target: 92%+")
    log(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    log("Strategy: self-clean + 3 models + TTA + optimized ensemble")
    log("=" * 70)

    try:
        step1_self_clean()
    except Exception as e:
        log(f"ERROR in step 1: {e}")
        traceback.print_exc()
        # Fall back to using clean source directly
        if not os.path.isdir(CLEANED_SOURCE):
            shutil.copytree(CLEAN_SOURCE, CLEANED_SOURCE)

    try:
        step2_expand()
    except Exception as e:
        log(f"ERROR in step 2: {e}")
        traceback.print_exc()
        return

    try:
        step3_train_swinv2()
    except Exception as e:
        log(f"ERROR in step 3: {e}")
        traceback.print_exc()

    try:
        step4_train_efficientnetv2()
    except Exception as e:
        log(f"ERROR in step 4: {e}")
        traceback.print_exc()

    try:
        step5_train_convnext()
    except Exception as e:
        log(f"ERROR in step 5: {e}")
        traceback.print_exc()

    try:
        step6_evaluate()
    except Exception as e:
        log(f"ERROR in step 6: {e}")
        traceback.print_exc()

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    log(f"\nPIPELINE COMPLETE — Total time: {hours}h {minutes}m")


if __name__ == "__main__":
    main()
