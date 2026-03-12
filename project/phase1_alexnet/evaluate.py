"""
Evaluation functions for the trained AlexNet landslide classifier.
Produces accuracy, precision, recall, F1, confusion matrix, and ROC curve.
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
from phase1_alexnet.model import get_model, get_efficientnet_b3
from phase1_alexnet.train import load_checkpoint
from utils.metrics import compute_metrics, print_metrics
from utils.plot_utils import plot_confusion_matrix, plot_roc_curve, plot_training_history
from utils.temperature_scaling import fit_temperature, save_temperature


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
            landslide_probs = probs[:, 1].cpu().numpy()  # prob of class 1 (landslide)

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(landslide_probs)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    results = compute_metrics(all_labels, all_preds, all_probs)
    results["all_preds"] = all_preds
    results["all_labels"] = all_labels
    results["all_probs"] = all_probs
    return results


def print_classification_report(results: dict) -> None:
    """Print sklearn classification report from evaluation results."""
    print("\n=== Classification Metrics ===")
    print_metrics(results, class_names=config.CLASS_NAMES)
    print(
        classification_report(
            results["all_labels"],
            results["all_preds"],
            target_names=config.CLASS_NAMES,
        )
    )


def run_evaluation(
    checkpoint_path: str = None,
    processed_dir: str = None,
    plots_dir: str = None,
) -> dict:
    """
    Load the best AlexNet checkpoint and evaluate on the test set.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        processed_dir:   Path to data/processed/.
        plots_dir:       Directory to save evaluation plots.

    Returns:
        Evaluation results dict.
    """
    if checkpoint_path is None:
        checkpoint_path = (
            config.EFFICIENTNET_CHECKPOINT
            if config.ACTIVE_MODEL == "efficientnet_b3"
            else config.ALEXNET_CHECKPOINT
        )
    if processed_dir is None:
        processed_dir = config.PROCESSED_DATA_DIR
    if plots_dir is None:
        plots_dir = config.PLOTS_DIR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model matching the active architecture
    for model_fn in [
        lambda: get_efficientnet_b3(num_classes=config.NUM_CLASSES),
        lambda: get_model(pretrained=True, num_classes=config.NUM_CLASSES),
        lambda: get_model(pretrained=False, num_classes=config.NUM_CLASSES),
    ]:
        try:
            model = model_fn()
            model = model.to(device)
            load_checkpoint(checkpoint_path, model)
            break
        except RuntimeError:
            continue

    # Fit temperature scaling on val set
    val_dataset = LandslideDataset(processed_dir, "val", get_test_transforms())
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    scaler = fit_temperature(model, val_loader, device)
    save_temperature(scaler.temperature.item())

    # Use dedicated test split for final evaluation
    test_dataset = LandslideDataset(processed_dir, "test", get_test_transforms())
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print("\n--- Raw (uncalibrated) model ---")
    results = evaluate_model(model, test_loader, device)
    print_classification_report(results)

    print("\n--- Temperature-calibrated model ---")
    results_cal = evaluate_model(scaler, test_loader, device)
    print_classification_report(results_cal)
    results["calibrated"] = results_cal

    # Save plots
    plot_confusion_matrix(
        results["confusion_matrix"],
        config.CLASS_NAMES,
        title="AlexNet Confusion Matrix",
        save_path=os.path.join(plots_dir, "confusion_matrix.png"),
    )
    plot_roc_curve(
        results["all_probs"],
        results["all_labels"],
        save_path=os.path.join(plots_dir, "roc_curve.png"),
    )

    return results


if __name__ == "__main__":
    run_evaluation()
