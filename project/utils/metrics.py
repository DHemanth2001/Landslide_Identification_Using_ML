"""Shared metric computation functions used by both phases.

Supports both binary (2-class) and multi-class evaluation.
Multi-class metrics use 'macro' and 'weighted' averaging.
"""

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


def compute_metrics(y_true, y_pred, y_prob=None, num_classes=None) -> dict:
    """
    Compute classification metrics for binary or multi-class problems.

    Args:
        y_true:      Ground-truth labels (array-like of ints).
        y_pred:      Predicted labels (array-like of ints).
        y_prob:      Predicted probabilities — shape (N,) for binary or (N, C) for multi-class.
        num_classes: Number of classes. Auto-detected if None.

    Returns:
        dict with accuracy, precision, recall, f1 (macro & weighted),
        confusion_matrix, per_class metrics, and optionally roc_auc.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if num_classes is None:
        num_classes = max(len(np.unique(y_true)), len(np.unique(y_pred)))

    is_binary = num_classes <= 2
    avg = "binary" if is_binary else "macro"

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(num_classes))),
        "num_classes": num_classes,
    }

    if not is_binary:
        results["precision_weighted"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        results["recall_weighted"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        results["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        results["precision_per_class"] = precision_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(num_classes)))
        results["recall_per_class"] = recall_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(num_classes)))
        results["f1_per_class"] = f1_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(num_classes)))

    if y_prob is not None:
        try:
            if is_binary:
                prob = np.asarray(y_prob)
                if prob.ndim == 2:
                    prob = prob[:, 1]
                results["roc_auc"] = roc_auc_score(y_true, prob)
            else:
                prob = np.asarray(y_prob)
                if prob.ndim == 2 and prob.shape[1] == num_classes:
                    results["roc_auc"] = roc_auc_score(y_true, prob, multi_class="ovr", average="macro")
                    results["roc_auc_weighted"] = roc_auc_score(y_true, prob, multi_class="ovr", average="weighted")
                else:
                    results["roc_auc"] = None
        except ValueError:
            results["roc_auc"] = None

    return results


def print_metrics(results: dict, class_names=None) -> None:
    """Pretty-print a metrics dict (supports binary and multi-class)."""
    is_multiclass = results.get("num_classes", 2) > 2

    print(f"  Accuracy : {results['accuracy']:.4f}")
    if is_multiclass:
        print(f"  Precision (macro)   : {results['precision']:.4f}")
        print(f"  Recall    (macro)   : {results['recall']:.4f}")
        print(f"  F1-Score  (macro)   : {results['f1']:.4f}")
        print(f"  Precision (weighted): {results.get('precision_weighted', 0):.4f}")
        print(f"  Recall    (weighted): {results.get('recall_weighted', 0):.4f}")
        print(f"  F1-Score  (weighted): {results.get('f1_weighted', 0):.4f}")
    else:
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall   : {results['recall']:.4f}")
        print(f"  F1-Score : {results['f1']:.4f}")
    if "roc_auc" in results and results["roc_auc"] is not None:
        print(f"  ROC-AUC  (macro)   : {results['roc_auc']:.4f}")
    if "roc_auc_weighted" in results and results["roc_auc_weighted"] is not None:
        print(f"  ROC-AUC  (weighted): {results['roc_auc_weighted']:.4f}")
    print()

    if class_names and is_multiclass:
        print("Per-class metrics:")
        print(f"  {'Class':>20s}  {'Precision':>9s}  {'Recall':>9s}  {'F1-Score':>9s}  {'Support':>7s}")
        cm = results["confusion_matrix"]
        for i, name in enumerate(class_names):
            p = results["precision_per_class"][i] if i < len(results.get("precision_per_class", [])) else 0
            r = results["recall_per_class"][i] if i < len(results.get("recall_per_class", [])) else 0
            f = results["f1_per_class"][i] if i < len(results.get("f1_per_class", [])) else 0
            support = int(cm[i].sum()) if i < len(cm) else 0
            print(f"  {name:>20s}  {p:>9.4f}  {r:>9.4f}  {f:>9.4f}  {support:>7d}")
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
