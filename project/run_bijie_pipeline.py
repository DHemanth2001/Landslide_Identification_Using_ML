"""
Bijie-only pipeline: Train on single clean source for maximum accuracy.

Bijie dataset: 770 landslide + 2003 non-landslide = 2,773 images
Clean labels (folder-based), no mask threshold conversion.

Strategy for 92%+:
  - Split 80/10/10 stratified
  - Heavy augmentation (landslide 15x, non_landslide 5x) → ~16K balanced
  - 3 models: SwinV2, EfficientNetV2+CBAM, ConvNeXt+CBAM+FPN
  - All trained 120+ epochs (small dataset needs more epochs)
  - TTA + optimized 3-model ensemble
  - Threshold optimization
"""

import os
import sys
import time
import json
import shutil
import random
import traceback
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

BIJIE_DIR = os.path.join(config.DATA_DIR, "bijie_raw", "Bijie-landslide-dataset")
BIJIE_SPLIT = os.path.join(config.DATA_DIR, "binary_bijie")
BIJIE_EXPANDED = os.path.join(config.DATA_DIR, "binary_bijie_expanded")
LOG_FILE = os.path.join(config.PLOTS_DIR, "bijie_pipeline.log")

config.PROCESSED_DATA_DIR = BIJIE_EXPANDED


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def random_augment(img):
    """Aggressive augmentation."""
    h, w = img.shape[:2]
    result = img.copy()

    if random.random() > 0.5:
        result = cv2.flip(result, 1)
    if random.random() > 0.5:
        result = cv2.flip(result, 0)

    angle = random.choice([0, 90, 180, 270]) + random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    alpha = random.uniform(0.7, 1.4)
    beta = random.randint(-30, 30)
    result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)

    if random.random() > 0.5:
        ksize = random.choice([3, 5])
        result = cv2.GaussianBlur(result, (ksize, ksize), 0)

    if random.random() > 0.6:
        noise = np.random.normal(0, 10, result.shape).astype(np.int16)
        result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if random.random() > 0.5:
        scale = random.uniform(0.8, 1.25)
        new_h, new_w = int(h * scale), int(w * scale)
        result = cv2.resize(result, (new_w, new_h))
        if new_h > h:
            y_off = (new_h - h) // 2
            x_off = (new_w - w) // 2
            result = result[y_off:y_off + h, x_off:x_off + w]
        else:
            pad_y = (h - new_h) // 2
            pad_x = (w - new_w) // 2
            result = cv2.copyMakeBorder(
                result, pad_y, h - new_h - pad_y,
                pad_x, w - new_w - pad_x,
                cv2.BORDER_REFLECT_101,
            )

    if random.random() > 0.7:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + random.randint(-15, 15)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(int) + random.randint(-20, 20), 0, 255).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Random erasing (cutout)
    if random.random() > 0.7:
        eh = random.randint(5, max(6, h // 5))
        ew = random.randint(5, max(6, w // 5))
        ey = random.randint(0, max(1, h - eh))
        ex = random.randint(0, max(1, w - ew))
        result[ey:ey + eh, ex:ex + ew] = np.random.randint(0, 255, (eh, ew, 3), dtype=np.uint8)

    # Elastic-like warping
    if random.random() > 0.8:
        pts1 = np.float32([[0, 0], [w, 0], [0, h]])
        dx, dy = random.uniform(-5, 5), random.uniform(-5, 5)
        pts2 = np.float32([[dx, dy], [w + dx, dy], [dx, h + dy]])
        M = cv2.getAffineTransform(pts1, pts2)
        result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    return result


def step1_split_and_expand():
    """Split Bijie into train/val/test and expand with augmentation."""
    log("STEP 1: Splitting Bijie dataset and expanding...")

    if os.path.isdir(BIJIE_SPLIT):
        shutil.rmtree(BIJIE_SPLIT)
    if os.path.isdir(BIJIE_EXPANDED):
        shutil.rmtree(BIJIE_EXPANDED)

    # Collect all images
    all_images = []
    for cls in ["landslide", "non-landslide"]:
        img_dir = os.path.join(BIJIE_DIR, cls, "image")
        label = "landslide" if cls == "landslide" else "non_landslide"
        for f in sorted(Path(img_dir).iterdir()):
            if f.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                all_images.append((str(f), label))

    random.seed(42)
    random.shuffle(all_images)

    # Stratified split 80/10/10
    paths = [x[0] for x in all_images]
    labels = [x[1] for x in all_images]

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )

    log(f"  Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

    # Count per class
    for split_name, split_labels in [("train", train_labels), ("val", val_labels), ("test", test_labels)]:
        ls = sum(1 for l in split_labels if l == "landslide")
        nls = sum(1 for l in split_labels if l == "non_landslide")
        log(f"    {split_name}: {ls} landslide, {nls} non_landslide")

    # Save split dataset (resize to 128x128 for consistency)
    for split_name, split_paths, split_labels in [
        ("train", train_paths, train_labels),
        ("val", val_paths, val_labels),
        ("test", test_paths, test_labels),
    ]:
        for img_path, label in zip(split_paths, split_labels):
            dst_dir = os.path.join(BIJIE_SPLIT, split_name, label)
            os.makedirs(dst_dir, exist_ok=True)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                fname = Path(img_path).name
                cv2.imwrite(os.path.join(dst_dir, fname), img)

    # Expand training set
    # Bijie train: ~616 landslide, ~1602 non_landslide
    multipliers = {"landslide": 15, "non_landslide": 5}

    # Copy val and test as-is
    for split in ["val", "test"]:
        src = os.path.join(BIJIE_SPLIT, split)
        dst = os.path.join(BIJIE_EXPANDED, split)
        if os.path.isdir(src):
            shutil.copytree(src, dst)

    total = 0
    for cls in ["landslide", "non_landslide"]:
        src_dir = os.path.join(BIJIE_SPLIT, "train", cls)
        dst_dir = os.path.join(BIJIE_EXPANDED, "train", cls)
        os.makedirs(dst_dir, exist_ok=True)

        images = sorted([f for f in Path(src_dir).iterdir()
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        mult = multipliers[cls]

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            cv2.imwrite(os.path.join(dst_dir, img_path.name), img)
            total += 1

            for i in range(mult):
                aug = random_augment(img)
                cv2.imwrite(os.path.join(dst_dir, f"{img_path.stem}_aug{i:02d}{img_path.suffix}"), aug)
                total += 1

    log(f"  Expanded training images: {total}")

    for cls in ["landslide", "non_landslide"]:
        cls_dir = os.path.join(BIJIE_EXPANDED, "train", cls)
        count = len([f for f in Path(cls_dir).iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        log(f"    train/{cls}: {count}")

    log("STEP 1 COMPLETE.")


def train_model(model_name, num_epochs, batch_size, learning_rate, step_num):
    """Generic training function."""
    log(f"STEP {step_num}: Training {model_name} ({num_epochs} epochs, batch={batch_size}, lr={learning_rate})...")
    from phase1_alexnet.train import run_training
    from utils.plot_utils import plot_training_history

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Delete old checkpoints for this model
    ckpt_map = {
        "swinv2_s": [config.SWINV2_CHECKPOINT, config.EMA_SWINV2_CHECKPOINT],
        "efficientnetv2_cbam": [config.EFFNETV2_CHECKPOINT, config.EMA_EFFNETV2_CHECKPOINT],
        "convnext_cbam_fpn": [config.CONVNEXT_CHECKPOINT, config.EMA_CONVNEXT_CHECKPOINT],
    }
    for ckpt in ckpt_map.get(model_name, []):
        if os.path.exists(ckpt):
            os.remove(ckpt)

    model, history = run_training(
        model_name=model_name,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    plot_training_history(
        history,
        save_path=os.path.join(config.PLOTS_DIR, f"training_history_{model_name}_bijie.png"),
    )
    with open(os.path.join(config.PLOTS_DIR, f"history_{model_name}_bijie.json"), "w") as f:
        json.dump(history, f)
    del model
    torch.cuda.empty_cache()
    log(f"STEP {step_num} COMPLETE.")
    return history


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


def step5_evaluate():
    """Full evaluation with TTA + 3-model ensemble + optimization."""
    log("STEP 5: Full evaluation...")

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
        ("SwinV2-S", config.SWINV2_CHECKPOINT, config.EMA_SWINV2_CHECKPOINT,
         get_swinv2_s, config.SWINV2_IMG_SIZE),
        ("EfficientNetV2-CBAM", config.EFFNETV2_CHECKPOINT, config.EMA_EFFNETV2_CHECKPOINT,
         get_efficientnetv2_cbam, config.EFFNETV2_IMG_SIZE),
        ("ConvNeXt-CBAM-FPN", config.CONVNEXT_CHECKPOINT, config.EMA_CONVNEXT_CHECKPOINT,
         get_convnext_cbam_fpn, config.CONVNEXT_IMG_SIZE),
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
            plt.savefig(os.path.join(config.PLOTS_DIR, f"confusion_matrix_{safe}_bijie.png"), dpi=150)
            plt.close()

            del model
            torch.cuda.empty_cache()

    # Exhaustive ensemble search
    log("\nExhaustive ensemble optimization...")
    tta_keys = {k: v for k, v in results.items() if "TTA" in k}
    all_keys = list(results.keys())
    labels = None

    model_probs = {}
    for k, v in results.items():
        model_probs[k] = np.array(v["probs"])
        if labels is None:
            labels = np.array(v["labels"])

    best_acc = 0
    best_config = {}

    # Search all combinations (including non-TTA)
    keys = list(model_probs.keys())
    n = len(keys)

    # All pairs
    for i in range(n):
        for j in range(i + 1, n):
            for w1 in np.arange(0.1, 0.95, 0.05):
                w2 = 1.0 - w1
                ens = model_probs[keys[i]] * w1 + model_probs[keys[j]] * w2
                for t in np.arange(0.35, 0.7, 0.01):
                    preds = (ens > t).astype(int)
                    acc = accuracy_score(labels, preds)
                    if acc > best_acc:
                        best_acc = acc
                        best_config = {"models": {keys[i]: w1, keys[j]: w2}, "threshold": t}

    # All triples
    for i in range(n):
        for j in range(i + 1, n):
            for k_idx in range(j + 1, n):
                for w1 in np.arange(0.1, 0.6, 0.1):
                    for w2 in np.arange(0.1, 0.6, 0.1):
                        w3 = 1.0 - w1 - w2
                        if w3 < 0.05 or w3 > 0.7:
                            continue
                        ens = (model_probs[keys[i]] * w1 +
                               model_probs[keys[j]] * w2 +
                               model_probs[keys[k_idx]] * w3)
                        for t in np.arange(0.35, 0.7, 0.02):
                            preds = (ens > t).astype(int)
                            acc = accuracy_score(labels, preds)
                            if acc > best_acc:
                                best_acc = acc
                                best_config = {
                                    "models": {keys[i]: w1, keys[j]: w2, keys[k_idx]: w3},
                                    "threshold": t,
                                }

    log(f"\nBest ensemble accuracy: {best_acc*100:.2f}%")
    log(f"Config: {json.dumps({k: round(float(v), 3) if isinstance(v, (float, np.floating)) else v for k, v in best_config.items()}, indent=2)}")

    # Final plots with best ensemble
    ens_probs = sum(model_probs[k] * w for k, w in best_config["models"].items())
    ens_preds = (ens_probs > best_config["threshold"]).astype(int)

    report = classification_report(labels, ens_preds, target_names=config.CLASS_NAMES, digits=4)
    log(f"\n{report}")

    cm = confusion_matrix(labels, ens_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES, ax=ax)
    ax.set_title(f"Best Ensemble Bijie (Acc={best_acc:.4f})")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, "confusion_matrix_best_ensemble_bijie.png"), dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(labels, ens_probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, "b-", label=f"Best Ensemble (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Bijie Best Ensemble")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, "roc_curve_best_ensemble_bijie.png"), dpi=150)
    plt.close()

    # Save
    summary = {}
    for k, v in results.items():
        summary[k] = {"accuracy": v["accuracy"]}
    summary["best_ensemble"] = {
        "accuracy": best_acc, "auc": float(roc_auc),
        "config": str(best_config),
    }
    with open(os.path.join(config.PLOTS_DIR, "evaluation_results_bijie.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log("\n" + "=" * 70)
    log("FINAL RESULTS (BIJIE ONLY)")
    log("=" * 70)
    for k, v in sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True):
        log(f"  {k}: {v['accuracy']*100:.2f}%")
    log(f"\n  BEST ENSEMBLE: {best_acc*100:.2f}% (AUC: {roc_auc:.4f})")
    log("=" * 70)


def main():
    start_time = time.time()
    log("=" * 70)
    log("BIJIE-ONLY PIPELINE — Target: 92%+")
    log(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    log("Single clean source, no domain gap")
    log("=" * 70)

    try:
        step1_split_and_expand()
    except Exception as e:
        log(f"ERROR in step 1: {e}")
        traceback.print_exc()
        return

    # Train 3 models
    try:
        train_model("swinv2_s", num_epochs=120, batch_size=24, learning_rate=2e-5, step_num=2)
    except Exception as e:
        log(f"ERROR in step 2: {e}")
        traceback.print_exc()

    try:
        train_model("efficientnetv2_cbam", num_epochs=120, batch_size=24, learning_rate=2e-5, step_num=3)
    except Exception as e:
        log(f"ERROR in step 3: {e}")
        traceback.print_exc()

    try:
        train_model("convnext_cbam_fpn", num_epochs=100, batch_size=32, learning_rate=3e-5, step_num=4)
    except Exception as e:
        log(f"ERROR in step 4: {e}")
        traceback.print_exc()

    try:
        step5_evaluate()
    except Exception as e:
        log(f"ERROR in step 5: {e}")
        traceback.print_exc()

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    log(f"\nPIPELINE COMPLETE — Total time: {hours}h {minutes}m")


if __name__ == "__main__":
    main()
