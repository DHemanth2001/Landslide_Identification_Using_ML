"""
K-Fold Cross Validation Training for MSAFusionNet.

Instead of a fixed train/val split, this uses 5-fold CV to:
  1. Train 5 separate models, each on a different 80/20 split
  2. Each fold's model sees different data → learns different patterns
  3. At test time, ensemble all 5 fold models → much stronger predictions

This is the key strategy to maximize accuracy on a small dataset.

Expected improvement: +2-4% over single-split training.
"""

import os
import sys
import json
import random
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Subset
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from phase1_alexnet.dataset import (
    LandslideDataset,
    get_train_transforms,
    get_test_transforms,
    mixup_data,
    cutmix_data,
    mixup_criterion,
)
from phase1_alexnet.model import get_convnext_cbam_fpn, get_swinv2_s, EMAModel
from phase1_alexnet.train import (
    train_one_epoch,
    validate,
    save_checkpoint,
    load_checkpoint,
    set_seed,
    get_cosine_schedule_with_warmup,
    unfreeze_backbone,
)
from utils.focal_loss import FocalLoss
from utils.metrics import compute_metrics, print_metrics


class IndexedDataset(Dataset):
    """Wrapper that allows applying different transforms to a subset."""

    def __init__(self, base_dataset, indices, transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
        self.samples = [base_dataset.samples[i] for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        import cv2
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((*config.CONVNEXT_IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        return img, label


def run_kfold_training(
    n_folds: int = 5,
    model_name: str = "convnext_cbam_fpn",
    num_epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    processed_dir: str = None,
    use_ema: bool = True,
):
    """
    Train models using K-Fold Cross Validation.

    Args:
        n_folds:       Number of CV folds (default 5).
        model_name:    Model architecture.
        num_epochs:    Epochs per fold.
        batch_size:    Batch size.
        learning_rate: Learning rate.
        processed_dir: Path to data/processed/ or data/processed_expanded/.
        use_ema:       Use EMA model averaging.

    Returns:
        dict with per-fold results and overall metrics.
    """
    set_seed(config.RANDOM_SEED)

    if num_epochs is None:
        num_epochs = config.NUM_EPOCHS
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE
    if processed_dir is None:
        # Prefer expanded dataset if it exists
        expanded = os.path.join(config.DATA_DIR, "processed_expanded")
        processed_dir = expanded if os.path.isdir(os.path.join(expanded, "train")) else config.PROCESSED_DATA_DIR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"K-Fold: {n_folds} folds, {num_epochs} epochs each")
    print(f"Data: {processed_dir}")

    if model_name == "swinv2_s":
        img_size = config.SWINV2_IMG_SIZE
    else:
        img_size = config.CONVNEXT_IMG_SIZE

    # Load all training data (without transforms, we'll apply per-fold)
    full_dataset = LandslideDataset(processed_dir, "train", transform=None)
    all_labels = [full_dataset.samples[i][1] for i in range(len(full_dataset))]

    print(f"Total training images: {len(full_dataset)}")
    for c in range(config.NUM_CLASSES):
        count = all_labels.count(c)
        print(f"  {config.CLASS_NAMES[c]:>20s}: {count}")

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_SEED)

    fold_results = []
    fold_checkpoints = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(full_dataset)), all_labels)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'='*60}")
        print(f"  Train: {len(train_idx)} images, Val: {len(val_idx)} images")

        # Create fold-specific datasets with appropriate transforms
        train_subset = IndexedDataset(full_dataset, train_idx, get_train_transforms(img_size))
        val_subset = IndexedDataset(full_dataset, val_idx, get_test_transforms(img_size))

        # Weighted sampler for training
        train_labels = [all_labels[i] for i in train_idx]
        class_counts = [max(train_labels.count(c), 1) for c in range(config.NUM_CLASSES)]
        weights = [1.0 / class_counts[lbl] for lbl in train_labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        train_loader = DataLoader(
            train_subset, batch_size=batch_size, sampler=sampler,
            num_workers=4, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
        )

        # Create fresh model for this fold
        if model_name == "convnext_cbam_fpn":
            model = get_convnext_cbam_fpn(num_classes=config.NUM_CLASSES, freeze=True)
        else:
            model = get_swinv2_s(num_classes=config.NUM_CLASSES, freeze=True)
        model = model.to(device)

        # EMA
        ema = EMAModel(model, decay=config.EMA_DECAY) if use_ema else None

        # Optimizer
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate, weight_decay=config.WEIGHT_DECAY,
        )

        # Loss
        criterion = FocalLoss(gamma=2.0, label_smoothing=config.LABEL_SMOOTHING)

        # Scheduler
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_epochs=config.WARMUP_EPOCHS, total_epochs=num_epochs,
        )

        # Checkpoint path for this fold
        ckpt_path = os.path.join(
            config.CHECKPOINTS_DIR,
            f"{model_name}_fold{fold + 1}_best.pth",
        )
        ema_ckpt_path = os.path.join(
            config.CHECKPOINTS_DIR,
            f"{model_name}_fold{fold + 1}_ema_best.pth",
        )

        best_val_acc = 0.0
        best_ema_val_acc = 0.0

        for epoch in range(1, num_epochs + 1):
            # Unfreeze backbone
            if epoch == config.UNFREEZE_EPOCH + 1:
                optimizer = unfreeze_backbone(model, model_name, learning_rate, optimizer)
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer, warmup_epochs=2, total_epochs=max(1, num_epochs - epoch),
                )
                if use_ema:
                    ema = EMAModel(model, decay=config.EMA_DECAY)
                print(f"  Epoch {epoch}: Backbone unfrozen")

            # Train
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                ema=ema, use_mixup=True, epoch=epoch,
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step()

            if epoch % 10 == 0 or epoch == num_epochs:
                print(
                    f"  Fold {fold+1} Epoch [{epoch:>3}/{num_epochs}]  "
                    f"Train: {train_acc:.4f}  Val: {val_acc:.4f}"
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(model, optimizer, epoch, val_acc, ckpt_path)

            # EMA validation
            if ema is not None:
                ema.apply(model)
                _, ema_val_acc = validate(model, val_loader, criterion, device)
                ema.restore(model)
                if ema_val_acc > best_ema_val_acc:
                    best_ema_val_acc = ema_val_acc
                    ema.apply(model)
                    save_checkpoint(model, optimizer, epoch, ema_val_acc, ema_ckpt_path)
                    ema.restore(model)

        print(f"  Fold {fold+1} best val_acc: {best_val_acc:.4f} (EMA: {best_ema_val_acc:.4f})")

        fold_results.append({
            "fold": fold + 1,
            "best_val_acc": best_val_acc,
            "best_ema_val_acc": best_ema_val_acc,
        })
        fold_checkpoints.append(ema_ckpt_path if use_ema else ckpt_path)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("K-FOLD CROSS VALIDATION SUMMARY")
    print(f"{'='*60}")
    accs = [r["best_ema_val_acc" if use_ema else "best_val_acc"] for r in fold_results]
    for r in fold_results:
        print(f"  Fold {r['fold']}: val_acc={r['best_val_acc']:.4f}, ema_val_acc={r['best_ema_val_acc']:.4f}")
    print(f"\n  Mean accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Checkpoints saved: {fold_checkpoints}")

    return {
        "fold_results": fold_results,
        "fold_checkpoints": fold_checkpoints,
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
    }


def evaluate_kfold_ensemble(
    model_name: str = "convnext_cbam_fpn",
    n_folds: int = 5,
    processed_dir: str = None,
):
    """
    Evaluate the K-Fold ensemble on the test set.
    Loads all fold models, averages their predictions.
    """
    if processed_dir is None:
        expanded = os.path.join(config.DATA_DIR, "processed_expanded")
        processed_dir = expanded if os.path.isdir(os.path.join(expanded, "test")) else config.PROCESSED_DATA_DIR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "swinv2_s":
        img_size = config.SWINV2_IMG_SIZE
    else:
        img_size = config.CONVNEXT_IMG_SIZE

    # Load test set
    test_dataset = LandslideDataset(processed_dir, "test", get_test_transforms(img_size))
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    print(f"Test set: {len(test_dataset)} images")

    # Load all fold models
    models = []
    for fold in range(1, n_folds + 1):
        ema_path = os.path.join(config.CHECKPOINTS_DIR, f"{model_name}_fold{fold}_ema_best.pth")
        raw_path = os.path.join(config.CHECKPOINTS_DIR, f"{model_name}_fold{fold}_best.pth")
        ckpt_path = ema_path if os.path.exists(ema_path) else raw_path

        if not os.path.exists(ckpt_path):
            print(f"  Fold {fold} checkpoint not found, skipping.")
            continue

        if model_name == "convnext_cbam_fpn":
            model = get_convnext_cbam_fpn(num_classes=config.NUM_CLASSES, freeze=False)
        else:
            model = get_swinv2_s(num_classes=config.NUM_CLASSES, freeze=False)
        model = model.to(device)
        load_checkpoint(ckpt_path, model)
        model.eval()
        models.append(model)
        print(f"  Loaded fold {fold} from {ckpt_path}")

    if not models:
        print("ERROR: No fold checkpoints found!")
        return None

    print(f"\nEvaluating {len(models)}-fold ensemble on test set...")

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="K-Fold Ensemble Eval"):
            images = images.to(device)

            # Average probabilities across all fold models
            fold_probs = []
            for model in models:
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                fold_probs.append(probs)

            # Mean ensemble
            avg_probs = torch.stack(fold_probs).mean(dim=0)
            preds = avg_probs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(avg_probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    results = compute_metrics(all_labels, all_preds, all_probs, num_classes=config.NUM_CLASSES)
    results["all_preds"] = all_preds
    results["all_labels"] = all_labels
    results["all_probs"] = all_probs

    print(f"\n{'='*60}")
    print(f"K-FOLD ENSEMBLE RESULTS ({model_name}, {len(models)} folds)")
    print(f"{'='*60}")
    print_metrics(results, class_names=config.CLASS_NAMES)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train")
    parser.add_argument("--model", default="convnext_cbam_fpn")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--data", default=None, help="Path to processed data")
    args = parser.parse_args()

    if args.mode == "train":
        results = run_kfold_training(
            n_folds=args.folds,
            model_name=args.model,
            processed_dir=args.data,
        )
        # Save results
        results_path = os.path.join(config.CHECKPOINTS_DIR, f"kfold_results_{args.model}.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")

    elif args.mode == "evaluate":
        evaluate_kfold_ensemble(
            model_name=args.model,
            n_folds=args.folds,
            processed_dir=args.data,
        )
