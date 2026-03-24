"""
Evaluation functions for landslide classifier (6-class).
Produces accuracy, per-class precision/recall/F1, confusion matrix, and ROC curves.

Supports the new MSAFusionNet (ConvNeXt-CBAM-FPN) and SwinV2-Small models,
as well as legacy models (AlexNet, EfficientNet-B3, ViT-B/16).
"""

import os
import sys

import numpy as np
import torch
from sklearn.metrics import classification_report
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from phase1_alexnet.dataset import LandslideDataset, get_dataloaders, get_test_transforms
from torch.utils.data import DataLoader
from phase1_alexnet.model import (
    get_convnext_cbam_fpn,
    get_swinv2_s,
    get_model,
    get_efficientnet_b3,
    get_vit_b_16,
)
from phase1_alexnet.train import load_checkpoint
from utils.metrics import compute_metrics, print_metrics
from utils.plot_utils import plot_confusion_matrix, plot_roc_curve, plot_training_history
from utils.temperature_scaling import fit_temperature, save_temperature
from utils.tta import get_tta_transforms


def _create_model(model_name: str, num_classes: int):
    """Create a model by name (no checkpoint loading)."""
    if model_name == "convnext_cbam_fpn":
        return get_convnext_cbam_fpn(num_classes=num_classes, freeze=False)
    elif model_name == "swinv2_s":
        return get_swinv2_s(num_classes=num_classes, freeze=False)
    elif model_name == "efficientnet_b3":
        return get_efficientnet_b3(num_classes=num_classes)
    elif model_name == "vit_b_16":
        return get_vit_b_16(num_classes=num_classes)
    else:
        return get_model(pretrained=False, num_classes=num_classes)


def _get_checkpoint_path(model_name: str, use_ema: bool = True):
    """Get checkpoint path for a model, preferring EMA checkpoints."""
    if model_name == "convnext_cbam_fpn":
        ema_path = config.EMA_CONVNEXT_CHECKPOINT
        raw_path = config.CONVNEXT_CHECKPOINT
    elif model_name == "swinv2_s":
        ema_path = config.EMA_SWINV2_CHECKPOINT
        raw_path = config.SWINV2_CHECKPOINT
    elif model_name == "efficientnet_b3":
        return config.EFFICIENTNET_CHECKPOINT
    elif model_name == "vit_b_16":
        return config.VIT_CHECKPOINT
    else:
        return config.ALEXNET_CHECKPOINT

    # Prefer EMA checkpoint if it exists
    if use_ema and os.path.exists(ema_path):
        print(f"Using EMA checkpoint: {ema_path}")
        return ema_path
    return raw_path


def _get_img_size(model_name: str):
    """Get image size for a model."""
    if model_name == "swinv2_s":
        return config.SWINV2_IMG_SIZE
    return config.CONVNEXT_IMG_SIZE


def evaluate_model(model, test_loader, device: torch.device) -> dict:
    """
    Run full inference on the test set and compute all metrics.

    Returns:
        dict with keys: accuracy, precision, recall, f1, roc_auc,
                        confusion_matrix, all_preds, all_labels, all_probs
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            preds = probs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    results = compute_metrics(all_labels, all_preds, all_probs, num_classes=config.NUM_CLASSES)
    results["all_preds"] = all_preds
    results["all_labels"] = all_labels
    results["all_probs"] = all_probs
    return results


def evaluate_model_with_tta(model, test_dataset, device: torch.device,
                            img_size=None, n_augments: int = 5) -> dict:
    """
    Run TTA inference on the test set: for each image, apply N augmentations
    and average the softmax probabilities before taking argmax.
    """
    import cv2
    if img_size is None:
        img_size = _get_img_size(config.ACTIVE_MODEL)

    tta_transforms = get_tta_transforms(img_size)[:n_augments]
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    for idx in tqdm(range(len(test_dataset)), desc=f"Evaluating (TTA x{n_augments})"):
        img_path, label = test_dataset.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        aug_probs = []
        with torch.no_grad():
            for t in tta_transforms:
                tensor = t(img).unsqueeze(0).to(device)
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                aug_probs.append(probs)

        avg_probs = np.mean(aug_probs, axis=0)
        pred = int(np.argmax(avg_probs))

        all_preds.append(pred)
        all_labels.append(label)
        all_probs.append(avg_probs)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    results = compute_metrics(all_labels, all_preds, all_probs, num_classes=config.NUM_CLASSES)
    results["all_preds"] = all_preds
    results["all_labels"] = all_labels
    results["all_probs"] = all_probs
    results["tta_augments"] = n_augments
    return results


def print_classification_report(results: dict) -> None:
    """Print sklearn classification report from evaluation results."""
    print("\n=== Classification Metrics ===")
    print_metrics(results, class_names=config.CLASS_NAMES)
    target_names = config.CLASS_NAMES[:results.get("num_classes", config.NUM_CLASSES)]
    print(
        classification_report(
            results["all_labels"],
            results["all_preds"],
            target_names=target_names,
            labels=list(range(len(target_names))),
            zero_division=0,
        )
    )


def run_evaluation(
    model_name: str = None,
    checkpoint_path: str = None,
    processed_dir: str = None,
    plots_dir: str = None,
    use_ema: bool = True,
) -> dict:
    """
    Load the best checkpoint and evaluate on the test set (6-class).

    Args:
        model_name:      Model architecture name.
        checkpoint_path: Path to .pth checkpoint file.
        processed_dir:   Path to data/processed/.
        plots_dir:       Directory to save evaluation plots.
        use_ema:         Prefer EMA checkpoint if available.

    Returns:
        Evaluation results dict.
    """
    if model_name is None:
        model_name = config.ACTIVE_MODEL
    if checkpoint_path is None:
        checkpoint_path = _get_checkpoint_path(model_name, use_ema=use_ema)
    if processed_dir is None:
        processed_dir = config.PROCESSED_DATA_DIR
    if plots_dir is None:
        plots_dir = config.PLOTS_DIR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = _get_img_size(model_name)

    # Load model
    model = _create_model(model_name, num_classes=config.NUM_CLASSES)
    model = model.to(device)
    load_checkpoint(checkpoint_path, model)

    # Fit temperature scaling on val set
    val_dataset = LandslideDataset(processed_dir, "val", get_test_transforms(img_size))
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True)
    scaler = fit_temperature(model, val_loader, device)
    save_temperature(scaler.temperature.item())

    # Use dedicated test split for final evaluation
    test_dataset = LandslideDataset(processed_dir, "test", get_test_transforms(img_size))
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    print(f"\n--- Raw (uncalibrated) {model_name} ---")
    results = evaluate_model(model, test_loader, device)
    print_classification_report(results)

    print(f"\n--- Temperature-calibrated {model_name} ---")
    results_cal = evaluate_model(scaler, test_loader, device)
    print_classification_report(results_cal)
    results["calibrated"] = results_cal

    # TTA evaluation
    print(f"\n--- TTA evaluation (5 augmented views) ---")
    results_tta = evaluate_model_with_tta(scaler, test_dataset, device,
                                          img_size=img_size, n_augments=5)
    print_classification_report(results_tta)
    results["tta"] = results_tta

    # Save plots
    suffix = f"_{model_name}"
    plot_confusion_matrix(
        results_tta["confusion_matrix"],
        config.CLASS_NAMES,
        title=f"6-Class Confusion Matrix — {model_name} (TTA)",
        save_path=os.path.join(plots_dir, f"confusion_matrix{suffix}.png"),
    )
    plot_roc_curve(
        results_tta["all_probs"],
        results_tta["all_labels"],
        class_names=config.CLASS_NAMES,
        save_path=os.path.join(plots_dir, f"roc_curve{suffix}.png"),
    )

    return results


if __name__ == "__main__":
    # Evaluate primary model
    print("=" * 60)
    print("Evaluating MSAFusionNet (ConvNeXt-CBAM-FPN)")
    print("=" * 60)
    run_evaluation(model_name="convnext_cbam_fpn")

    # Evaluate ensemble partner
    print("\n" + "=" * 60)
    print("Evaluating SwinV2-Small")
    print("=" * 60)
    run_evaluation(model_name="swinv2_s")
