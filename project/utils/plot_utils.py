"""Shared plotting utilities for training curves, confusion matrices, and ROC.

Supports both binary and multi-class (one-vs-rest) ROC curves.
"""

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
    Automatically scales figure size for multi-class matrices.

    Args:
        cm:          2D array-like confusion matrix.
        class_names: List of class label strings.
        title:       Figure title.
        save_path:   If provided, save the figure.
    """
    n = len(class_names)
    figsize = (max(6, n * 1.5), max(5, n * 1.2))
    plt.figure(figsize=figsize)
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


def plot_roc_curve(all_probs, all_labels, class_names=None, save_path: str = None) -> float:
    """
    Plot ROC curve(s). Supports binary and multi-class (one-vs-rest).

    Args:
        all_probs:   For binary: shape (N,). For multi-class: shape (N, C).
        all_labels:  Ground-truth integer labels.
        class_names: List of class name strings (required for multi-class).
        save_path:   If provided, save the figure.

    Returns:
        Macro-average AUC score as float.
    """
    from sklearn.metrics import auc
    from sklearn.preprocessing import label_binarize

    all_probs = np.asarray(all_probs)
    all_labels = np.asarray(all_labels)

    # Binary case
    if all_probs.ndim == 1 or (all_probs.ndim == 2 and all_probs.shape[1] <= 2):
        prob = all_probs[:, 1] if all_probs.ndim == 2 else all_probs
        fpr, tpr, _ = roc_curve(all_labels, prob)
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

    # Multi-class one-vs-rest
    n_classes = all_probs.shape[1]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    y_bin = label_binarize(all_labels, classes=list(range(n_classes)))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    plt.figure(figsize=(8, 6))
    auc_scores = []
    for i in range(n_classes):
        if y_bin[:, i].sum() == 0:
            continue
        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        auc_i = auc(fpr_i, tpr_i)
        auc_scores.append(auc_i)
        plt.plot(fpr_i, tpr_i, color=colors[i], lw=1.5,
                 label=f"{class_names[i]} (AUC={auc_i:.3f})")

    macro_auc = float(np.mean(auc_scores)) if auc_scores else 0.0
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Multi-Class ROC (One-vs-Rest, Macro AUC={macro_auc:.4f})")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    _save_or_show(save_path)
    return macro_auc


def plot_gradcam_grid(
    images: list, heatmaps: list, overlays: list,
    class_names: list, predicted_classes: list,
    title: str = "Grad-CAM Visualizations",
    save_path: str = None,
) -> None:
    """
    Plot a grid of original images, heatmaps, and overlays.

    Args:
        images:            List of RGB numpy arrays (H, W, 3).
        heatmaps:          List of heatmap numpy arrays (H, W) in [0, 1].
        overlays:          List of overlay RGB numpy arrays (H, W, 3).
        class_names:       List of all class name strings.
        predicted_classes:  List of predicted class indices.
        title:             Figure title.
        save_path:         If provided, save the figure.
    """
    n = len(images)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(heatmaps[i], cmap="jet")
        axes[i, 1].set_title("Grad-CAM Heatmap")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlays[i])
        pred_name = class_names[predicted_classes[i]] if predicted_classes[i] < len(class_names) else "?"
        axes[i, 2].set_title(f"Overlay — Pred: {pred_name}")
        axes[i, 2].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_or_show(save_path)


def _save_or_show(save_path: str = None) -> None:
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()
