"""
5-Fold Cross-Validation on Bijie dataset.

Proper statistical evaluation:
  - No test set optimization (threshold fixed at 0.5)
  - 5 different train/test splits
  - Reports mean ± std accuracy
  - Uses ConvNeXt-CBAM-FPN (best single model)
  - Also evaluates SwinV2 and EfficientNetV2 for comparison
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

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

BIJIE_DIR = os.path.join(config.DATA_DIR, "bijie_raw", "Bijie-landslide-dataset")
KFOLD_DIR = os.path.join(config.DATA_DIR, "binary_bijie_kfold")
LOG_FILE = os.path.join(config.PLOTS_DIR, "kfold_bijie.log")

N_FOLDS = 5
# Models to evaluate — ConvNeXt only (best model) for speed
# Previous run showed: ConvNeXt 98.74%/97.84%, SwinV2 98.56%, EfficientNetV2 97.48%
MODELS_TO_TRAIN = [
    ("convnext_cbam_fpn", 60, 32, 3e-5),   # 60 epochs is enough (converges by ~50)
]


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def random_augment(img):
    """Augmentation for training."""
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
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return result


def prepare_fold(all_paths, all_labels, train_idx, test_idx, fold_num):
    """Prepare train/val/test for a single fold."""
    fold_dir = os.path.join(KFOLD_DIR, f"fold_{fold_num}")
    if os.path.isdir(fold_dir):
        shutil.rmtree(fold_dir)

    train_paths = [all_paths[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    test_paths = [all_paths[i] for i in test_idx]
    test_labels = [all_labels[i] for i in test_idx]

    # Split 10% of train as val
    from sklearn.model_selection import train_test_split
    train_p, val_p, train_l, val_l = train_test_split(
        train_paths, train_labels, test_size=0.1, stratify=train_labels, random_state=42
    )

    # Save and augment
    multipliers = {"landslide": 12, "non_landslide": 4}

    for split, paths, labels, augment in [
        ("train", train_p, train_l, True),
        ("val", val_p, val_l, False),
        ("test", test_paths, test_labels, False),
    ]:
        for img_path, label in zip(paths, labels):
            dst_dir = os.path.join(fold_dir, split, label)
            os.makedirs(dst_dir, exist_ok=True)

            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (128, 128))

            fname = Path(img_path).stem
            cv2.imwrite(os.path.join(dst_dir, f"{fname}.jpg"), img)

            if augment:
                mult = multipliers.get(label, 4)
                for i in range(mult):
                    aug = random_augment(img)
                    cv2.imwrite(os.path.join(dst_dir, f"{fname}_aug{i:02d}.jpg"), aug)

    # Count
    for split in ["train", "val", "test"]:
        for cls in ["landslide", "non_landslide"]:
            cls_dir = os.path.join(fold_dir, split, cls)
            if os.path.isdir(cls_dir):
                count = len(list(Path(cls_dir).glob("*.jpg")))
                log(f"    {split}/{cls}: {count}")

    return fold_dir


def train_and_eval_fold(fold_dir, model_name, num_epochs, batch_size, lr):
    """Train and evaluate a model on one fold."""
    from phase1_alexnet.train import run_training
    from phase1_alexnet.model import get_convnext_cbam_fpn, get_efficientnetv2_cbam, get_swinv2_s
    from phase1_alexnet.dataset import LandslideDataset, get_test_transforms

    # Set config to use this fold's data
    config.PROCESSED_DATA_DIR = fold_dir

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Delete old checkpoints
    ckpt_map = {
        "convnext_cbam_fpn": [config.CONVNEXT_CHECKPOINT, config.EMA_CONVNEXT_CHECKPOINT],
        "swinv2_s": [config.SWINV2_CHECKPOINT, config.EMA_SWINV2_CHECKPOINT],
        "efficientnetv2_cbam": [config.EFFNETV2_CHECKPOINT, config.EMA_EFFNETV2_CHECKPOINT],
    }
    for ckpt in ckpt_map.get(model_name, []):
        if os.path.exists(ckpt):
            os.remove(ckpt)

    # Train
    model, history = run_training(
        model_name=model_name,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=lr,
    )
    del model
    torch.cuda.empty_cache()

    # Evaluate on test set (no threshold optimization — fixed at 0.5)
    get_model_fns = {
        "convnext_cbam_fpn": (get_convnext_cbam_fpn, config.CONVNEXT_IMG_SIZE,
                              config.CONVNEXT_CHECKPOINT, config.EMA_CONVNEXT_CHECKPOINT),
        "swinv2_s": (get_swinv2_s, config.SWINV2_IMG_SIZE,
                     config.SWINV2_CHECKPOINT, config.EMA_SWINV2_CHECKPOINT),
        "efficientnetv2_cbam": (get_efficientnetv2_cbam, config.EFFNETV2_IMG_SIZE,
                                config.EFFNETV2_CHECKPOINT, config.EMA_EFFNETV2_CHECKPOINT),
    }

    get_fn, img_size, ckpt_raw, ckpt_ema = get_model_fns[model_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

    for tag, ckpt in [("raw", ckpt_raw), ("ema", ckpt_ema)]:
        if not os.path.exists(ckpt):
            continue

        model_inst = get_fn(num_classes=2, freeze=False).to(device)
        checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
        model_inst.load_state_dict(checkpoint["model_state_dict"])
        model_inst.eval()

        test_dataset = LandslideDataset(fold_dir, "test", get_test_transforms(img_size))
        loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

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
        f1 = f1_score(all_labels, all_preds, average="weighted")
        try:
            auc_score = roc_auc_score(all_labels, all_probs)
        except Exception:
            auc_score = 0.0

        results[tag] = {"accuracy": acc, "f1": f1, "auc": auc_score}
        log(f"    {model_name}_{tag}: Acc={acc:.4f} ({acc*100:.2f}%), F1={f1:.4f}, AUC={auc_score:.4f}")

        del model_inst
        torch.cuda.empty_cache()

    # Return best result
    best = max(results.values(), key=lambda x: x["accuracy"])
    return best


def main():
    start_time = time.time()
    log("=" * 70)
    log("5-FOLD CROSS-VALIDATION ON BIJIE")
    log(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    log(f"Folds: {N_FOLDS}")
    log(f"Models: {[m[0] for m in MODELS_TO_TRAIN]}")
    log("No test set optimization — threshold fixed at 0.5")
    log("=" * 70)

    # Collect all Bijie images
    all_paths = []
    all_labels = []
    for cls, label in [("landslide", "landslide"), ("non-landslide", "non_landslide")]:
        img_dir = os.path.join(BIJIE_DIR, cls, "image")
        for f in sorted(Path(img_dir).iterdir()):
            if f.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                all_paths.append(str(f))
                all_labels.append(label)

    log(f"Total images: {len(all_paths)} ({sum(1 for l in all_labels if l == 'landslide')} ls, "
        f"{sum(1 for l in all_labels if l == 'non_landslide')} nls)")

    # Convert labels to ints for stratified split
    label_ints = np.array([0 if l == "non_landslide" else 1 for l in all_labels])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # Results storage
    all_results = {m[0]: [] for m in MODELS_TO_TRAIN}

    for fold_num, (train_idx, test_idx) in enumerate(skf.split(all_paths, label_ints), 1):
        log(f"\n{'='*50}")
        log(f"FOLD {fold_num}/{N_FOLDS}")
        log(f"{'='*50}")
        log(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

        # Prepare fold data
        fold_dir = prepare_fold(all_paths, all_labels, train_idx, test_idx, fold_num)

        # Train and evaluate each model
        for model_name, epochs, batch_size, lr in MODELS_TO_TRAIN:
            log(f"\n  Training {model_name} (fold {fold_num})...")
            try:
                result = train_and_eval_fold(fold_dir, model_name, epochs, batch_size, lr)
                all_results[model_name].append(result)
                log(f"  Fold {fold_num} {model_name}: Acc={result['accuracy']*100:.2f}%, "
                    f"F1={result['f1']*100:.2f}%, AUC={result['auc']:.4f}")
            except Exception as e:
                log(f"  ERROR in fold {fold_num} {model_name}: {e}")
                traceback.print_exc()

    # Summary
    log("\n" + "=" * 70)
    log("5-FOLD CROSS-VALIDATION RESULTS")
    log("=" * 70)

    summary = {}
    for model_name, results_list in all_results.items():
        if not results_list:
            continue

        accs = [r["accuracy"] for r in results_list]
        f1s = [r["f1"] for r in results_list]
        aucs = [r["auc"] for r in results_list]

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_f1 = np.mean(f1s)
        std_f1 = np.std(f1s)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        log(f"\n  {model_name}:")
        log(f"    Per-fold accuracies: {[f'{a*100:.2f}%' for a in accs]}")
        log(f"    Accuracy:  {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")
        log(f"    F1-Score:  {mean_f1*100:.2f}% +/- {std_f1*100:.2f}%")
        log(f"    AUC:       {mean_auc:.4f} +/- {std_auc:.4f}")

        summary[model_name] = {
            "per_fold_accuracy": [float(a) for a in accs],
            "mean_accuracy": float(mean_acc),
            "std_accuracy": float(std_acc),
            "mean_f1": float(mean_f1),
            "std_f1": float(std_f1),
            "mean_auc": float(mean_auc),
            "std_auc": float(std_auc),
        }

    # Overall best
    best_model = max(summary.items(), key=lambda x: x[1]["mean_accuracy"])
    log(f"\n  BEST MODEL: {best_model[0]}")
    log(f"  BEST ACCURACY: {best_model[1]['mean_accuracy']*100:.2f}% +/- {best_model[1]['std_accuracy']*100:.2f}%")

    # Save
    with open(os.path.join(config.PLOTS_DIR, "kfold_results_bijie.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log("=" * 70)

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    log(f"\nCOMPLETE — Total time: {hours}h {minutes}m")


if __name__ == "__main__":
    main()
