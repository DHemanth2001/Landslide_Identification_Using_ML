"""Shared metric computation functions used by both phases."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """
    Compute standard classification metrics.

    Args:
        y_true: Ground-truth labels (array-like of ints).
        y_pred: Predicted labels (array-like of ints).
        y_prob: Predicted probabilities for the positive class (optional).

    Returns:
        dict with accuracy, precision, recall, f1, confusion_matrix,
        and optionally roc_auc.
    """
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    if y_prob is not None:
        try:
            results["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            results["roc_auc"] = None

    return results


def print_metrics(results: dict, class_names=None) -> None:
    """Pretty-print a metrics dict."""
    print(f"  Accuracy : {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall   : {results['recall']:.4f}")
    print(f"  F1-Score : {results['f1']:.4f}")
    if "roc_auc" in results and results["roc_auc"] is not None:
        print(f"  ROC-AUC  : {results['roc_auc']:.4f}")
    print()
    if class_names:
        print("Confusion Matrix:")
        cm = results["confusion_matrix"]
        header = "         " + "  ".join(f"{c:>12}" for c in class_names)
        print(header)
        for i, row in enumerate(cm):
            label = class_names[i] if i < len(class_names) else str(i)
            print(f"  {label:>8}" + "  ".join(f"{v:>12}" for v in row))
        print()
