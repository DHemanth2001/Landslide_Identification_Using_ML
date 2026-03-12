"""Shared plotting utilities for training curves, confusion matrices, and ROC."""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve


def plot_training_history(history: dict, save_path: str = None) -> None:
    """
    Plot loss and accuracy curves from training history.

    Args:
        history: dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        save_path: If provided, save the figure to this path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train Loss", marker="o", markersize=3)
    axes[0].plot(history["val_loss"], label="Val Loss", marker="s", markersize=3)
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["train_acc"], label="Train Accuracy", marker="o", markersize=3)
    axes[1].plot(history["val_acc"], label="Val Accuracy", marker="s", markersize=3)
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    _save_or_show(save_path)


def plot_confusion_matrix(
    cm, class_names: list, title: str = "Confusion Matrix", save_path: str = None
) -> None:
    """
    Plot a confusion matrix as a seaborn heatmap.

    Args:
        cm:          2D array-like confusion matrix.
        class_names: List of class label strings.
        title:       Figure title.
        save_path:   If provided, save the figure.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    _save_or_show(save_path)


def plot_roc_curve(all_probs, all_labels, save_path: str = None) -> float:
    """
    Plot ROC curve and return AUC score.

    Args:
        all_probs:  Predicted probabilities for the positive (landslide) class.
        all_labels: Ground-truth binary labels.
        save_path:  If provided, save the figure.

    Returns:
        AUC score as float.
    """
    from sklearn.metrics import auc

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc_score = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    _save_or_show(save_path)
    return auc_score


def _save_or_show(save_path: str = None) -> None:
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()
