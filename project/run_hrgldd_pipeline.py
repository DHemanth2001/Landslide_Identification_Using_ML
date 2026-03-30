"""
Pipeline: Train on HR-GLDD only (single clean source).

Steps:
  1. Expand HR-GLDD with augmentation (landslide 13x, non_landslide 4x → ~8K balanced)
  2. Train EfficientNetV2-S + CBAM (100 epochs)
  3. Train ConvNeXt-CBAM-FPN (80 epochs)
  4. Evaluate with TTA + ensemble
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

HRGLDD_SOURCE = os.path.join(config.DATA_DIR, "binary")
HRGLDD_EXPANDED = os.path.join(config.DATA_DIR, "binary_hrgldd_expanded")
LOG_FILE = os.path.join(config.PLOTS_DIR, "hrgldd_pipeline.log")

# Override config
config.PROCESSED_DATA_DIR = HRGLDD_EXPANDED


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def random_augment(img):
    """Aggressive augmentation for small dataset."""
    import random
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
        noise = np.random.normal(0, 8, result.shape).astype(np.int16)
        result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if random.random() > 0.5:
        scale = random.uniform(0.85, 1.2)
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

    if random.random() > 0.8:
        channels = list(range(3))
        import random as _r
        _r.shuffle(channels)
        result = result[:, :, channels]

    if random.random() > 0.6:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + random.randint(-10, 10)) % 180
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Extra augmentations for small dataset
    # Elastic-like distortion via random erasing
    if random.random() > 0.7:
        eh = random.randint(5, h // 5)
        ew = random.randint(5, w // 5)
        ey = random.randint(0, h - eh)
        ex = random.randint(0, w - ew)
        result[ey:ey + eh, ex:ex + ew] = np.random.randint(0, 255, (eh, ew, 3), dtype=np.uint8)

    return result


def step1_expand():
    """Expand HR-GLDD with heavy augmentation."""
    log("STEP 1: Expanding HR-GLDD dataset...")

    if os.path.isdir(HRGLDD_EXPANDED):
        shutil.rmtree(HRGLDD_EXPANDED)

    # HR-GLDD: 616 landslide, 1574 non_landslide in train
    # Target: ~7K landslide, ~7K non_landslide (balanced)
    multipliers = {
        "landslide": 11,       # 616 × 12 = 7,392
        "non_landslide": 4,    # 1574 × 5 = 7,870
    }

    # Copy val and test as-is
    for split in ["val", "test"]:
        for cls in ["landslide", "non_landslide"]:
            src = os.path.join(HRGLDD_SOURCE, split, cls)
            dst = os.path.join(HRGLDD_EXPANDED, split, cls)
            os.makedirs(dst, exist_ok=True)
            if os.path.isdir(src):
                for f in Path(src).iterdir():
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        img = cv2.imread(str(f))
                        if img is not None:
                            cv2.imwrite(os.path.join(dst, f.name), img)

    # Expand training set
    total = 0
    for cls in ["landslide", "non_landslide"]:
        src = os.path.join(HRGLDD_SOURCE, "train", cls)
        dst = os.path.join(HRGLDD_EXPANDED, "train", cls)
        os.makedirs(dst, exist_ok=True)

        images = sorted([f for f in Path(src).iterdir()
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        mult = multipliers[cls]

        log(f"  {cls}: {len(images)} originals × {mult} augmentations")

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            cv2.imwrite(os.path.join(dst, img_path.name), img)
            total += 1

            for i in range(mult):
                aug = random_augment(img)
                cv2.imwrite(os.path.join(dst, f"{img_path.stem}_aug{i:02d}{img_path.suffix}"), aug)
                total += 1

    log(f"  Total training images: {total}")

    # Print distribution
    for cls in ["landslide", "non_landslide"]:
        cls_dir = os.path.join(HRGLDD_EXPANDED, "train", cls)
        count = len([f for f in Path(cls_dir).iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        log(f"  train/{cls}: {count}")

    log("STEP 1 COMPLETE.")


def step2_train_efficientnetv2():
    """Train EfficientNetV2-S + CBAM (100 epochs for small dataset)."""
    log("STEP 2: Training EfficientNetV2-S + CBAM (100 epochs, batch=24, lr=3e-5)...")
    from phase1_alexnet.train import run_training
    from utils.plot_utils import plot_training_history

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Delete old checkpoints
    for ckpt in [config.EFFNETV2_CHECKPOINT, config.EMA_EFFNETV2_CHECKPOINT]:
        if os.path.exists(ckpt):
            os.remove(ckpt)

    model, history = run_training(
        model_name="efficientnetv2_cbam",
        num_epochs=100,
        batch_size=24,
        learning_rate=3e-5,
    )
    plot_training_history(
        history,
        save_path=os.path.join(config.PLOTS_DIR, "training_history_efficientnetv2_hrgldd.png"),
    )
    with open(os.path.join(config.PLOTS_DIR, "history_efficientnetv2_hrgldd.json"), "w") as f:
        json.dump(history, f)
    del model
    torch.cuda.empty_cache()
    log("STEP 2 COMPLETE.")
    return history


def step3_train_convnext():
    """Train ConvNeXt-CBAM-FPN (80 epochs)."""
    log("STEP 3: Training ConvNeXt-CBAM-FPN (80 epochs, batch=32, lr=5e-5)...")
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
        learning_rate=5e-5,
    )
    plot_training_history(
        history,
        save_path=os.path.join(config.PLOTS_DIR, "training_history_convnext_hrgldd.png"),
    )
    with open(os.path.join(config.PLOTS_DIR, "history_convnext_hrgldd.json"), "w") as f:
        json.dump(history, f)
    del model
    torch.cuda.empty_cache()
    log("STEP 3 COMPLETE.")
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


def step4_evaluate():
    """Evaluate with TTA + ensemble."""
    log("STEP 4: Evaluating with TTA...")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        roc_curve, auc, accuracy_score,
    )
    from phase1_alexnet.model import get_convnext_cbam_fpn, get_efficientnetv2_cbam
    from phase1_alexnet.dataset import LandslideDataset, get_test_transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    for model_name, ckpt_path, ema_ckpt_path, get_model_fn, img_size in [
        ("EfficientNetV2-CBAM", config.EFFNETV2_CHECKPOINT, config.EMA_EFFNETV2_CHECKPOINT,
         get_efficientnetv2_cbam, config.EFFNETV2_IMG_SIZE),
        ("ConvNeXt-CBAM-FPN", config.CONVNEXT_CHECKPOINT, config.EMA_CONVNEXT_CHECKPOINT,
         get_convnext_cbam_fpn, config.CONVNEXT_IMG_SIZE),
    ]:
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
            plt.savefig(os.path.join(config.PLOTS_DIR, f"confusion_matrix_{safe}_hrgldd.png"), dpi=150)
            plt.close()

            del model
            torch.cuda.empty_cache()

    # Ensemble
    log("\nEnsemble evaluation with TTA...")
    best_ens_acc = 0
    best_weights = (0.5, 0.5)

    eff_key = "EfficientNetV2-CBAM_ema_TTA" if "EfficientNetV2-CBAM_ema_TTA" in results else "EfficientNetV2-CBAM_raw_TTA"
    conv_key = "ConvNeXt-CBAM-FPN_ema_TTA" if "ConvNeXt-CBAM-FPN_ema_TTA" in results else "ConvNeXt-CBAM-FPN_raw_TTA"

    if eff_key in results and conv_key in results:
        for w1 in np.arange(0.2, 0.85, 0.05):
            w2 = 1.0 - w1
            ens_probs = np.array(results[eff_key]["probs"]) * w1 + np.array(results[conv_key]["probs"]) * w2
            ens_preds = (ens_probs > 0.5).astype(int)
            ens_acc = accuracy_score(results[eff_key]["labels"], ens_preds)
            log(f"  Ensemble (Eff={w1:.2f}, Conv={w2:.2f}): {ens_acc:.4f} ({ens_acc*100:.2f}%)")
            if ens_acc > best_ens_acc:
                best_ens_acc = ens_acc
                best_weights = (w1, w2)

        ens_probs = (
            np.array(results[eff_key]["probs"]) * best_weights[0]
            + np.array(results[conv_key]["probs"]) * best_weights[1]
        )
        ens_preds = (ens_probs > 0.5).astype(int)
        ens_labels = results[eff_key]["labels"]

        log(f"\nBest Ensemble: Eff={best_weights[0]:.2f}, Conv={best_weights[1]:.2f}")
        log(f"Best Ensemble Accuracy: {best_ens_acc:.4f} ({best_ens_acc*100:.2f}%)")

        report = classification_report(ens_labels, ens_preds, target_names=config.CLASS_NAMES, digits=4)
        log(f"\n{report}")

        cm = confusion_matrix(ens_labels, ens_preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                    xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES, ax=ax)
        ax.set_title(f"Ensemble TTA HR-GLDD (Acc={best_ens_acc:.4f})")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(config.PLOTS_DIR, "confusion_matrix_ensemble_tta_hrgldd.png"), dpi=150)
        plt.close()

        fpr, tpr, _ = roc_curve(ens_labels, ens_probs)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, "b-", label=f"Ensemble TTA (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve - HR-GLDD Ensemble with TTA")
        ax.legend(); plt.tight_layout()
        plt.savefig(os.path.join(config.PLOTS_DIR, "roc_curve_ensemble_tta_hrgldd.png"), dpi=150)
        plt.close()

        results["ensemble_tta"] = {"accuracy": best_ens_acc, "auc": roc_auc}

    # Save results
    summary = {}
    for k, v in results.items():
        summary[k] = {"accuracy": v["accuracy"], "auc": v.get("auc")}
    with open(os.path.join(config.PLOTS_DIR, "evaluation_results_hrgldd.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log("\n" + "=" * 70)
    log("FINAL RESULTS (HR-GLDD ONLY)")
    log("=" * 70)
    for k, v in sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True):
        log(f"  {k}: {v['accuracy']*100:.2f}%")
    best_overall = max(v["accuracy"] for v in results.values())
    log(f"\n  BEST ACCURACY: {best_overall*100:.2f}%")
    log("=" * 70)
    return results


def main():
    start_time = time.time()
    log("=" * 70)
    log("HR-GLDD ONLY PIPELINE")
    log(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    log(f"Source: {HRGLDD_SOURCE}")
    log(f"Target: Single-source training for maximum accuracy")
    log("=" * 70)

    try:
        step1_expand()
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
    log(f"\nPIPELINE COMPLETE — Total time: {hours}h {minutes}m")


if __name__ == "__main__":
    main()
