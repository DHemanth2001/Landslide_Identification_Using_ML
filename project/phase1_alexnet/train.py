"""
Training loop for MSAFusionNet (ConvNeXt-CBAM-FPN) and SwinV2 landslide classifier.

Advanced training strategies:
  - Mixup + CutMix augmentation (applied at batch level)
  - Label smoothing + Focal Loss
  - EMA (Exponential Moving Average) model
  - AdamW optimizer with cosine annealing + warm-up
  - Gradient clipping for stability
  - Progressive unfreezing of backbone
"""

import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from phase1_alexnet.dataset import (
    get_dataloaders,
    mixup_data,
    cutmix_data,
    mixup_criterion,
)
from phase1_alexnet.model import (
    get_convnext_cbam_fpn,
    get_swinv2_s,
    get_efficientnetv2_cbam,
    EMAModel,
    # Legacy
    get_model,
    get_efficientnet_b3,
    get_vit_b_16,
)
from utils.focal_loss import FocalLoss


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # Faster for fixed input sizes


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=1e-7):
    """
    Linear warmup for warmup_epochs, then cosine decay to min_lr.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(min_lr / optimizer.defaults["lr"], 0.5 * (1.0 + np.cos(np.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    ema: EMAModel = None,
    use_mixup: bool = True,
    epoch: int = 0,
    scaler: torch.amp.GradScaler = None,
) -> tuple:
    """
    Run one training epoch with optional Mixup/CutMix and AMP.

    Returns:
        (avg_loss, accuracy) for the epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        # Apply Mixup or CutMix (50/50 chance) during training
        use_mix = use_mixup and epoch >= 3  # Skip mixup for first 3 epochs (warm-up)
        if use_mix and random.random() < 0.8:  # 80% chance of applying mix
            if random.random() < config.MIXUP_PROB:
                images, targets_a, targets_b, lam = mixup_data(
                    images, labels, alpha=config.MIXUP_ALPHA
                )
            else:
                images, targets_a, targets_b, lam = cutmix_data(
                    images, labels, alpha=config.CUTMIX_ALPHA
                )
            mixed = True
        else:
            mixed = False

        optimizer.zero_grad(set_to_none=True)

        # AMP forward pass
        with torch.amp.autocast("cuda"):
            outputs = model(images)
            if mixed:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_NORM)
            optimizer.step()

        # Update EMA after each step
        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * images.size(0)

        # For accuracy, use original labels (not mixed)
        preds = outputs.argmax(dim=1)
        if mixed:
            correct += (lam * (preds == targets_a).float() +
                       (1.0 - lam) * (preds == targets_b).float()).sum().item()
        else:
            correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """
    Run validation epoch without mixup/cutmix.

    Returns:
        (avg_loss, accuracy) for the epoch.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Val  ", leave=False):
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def unfreeze_backbone(model, model_name, learning_rate, optimizer):
    """
    Unfreeze all backbone parameters and create a new optimizer
    with differential learning rates (backbone gets lower LR).
    """
    for param in model.parameters():
        param.requires_grad = True

    if model_name == "convnext_cbam_fpn":
        # Backbone stages get 10x lower LR, everything else gets full LR
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if any(s in name for s in ["stage1", "stage2", "stage3", "stage4"]):
                backbone_params.append(param)
            else:
                head_params.append(param)
    elif model_name == "swinv2_s":
        head_params = []
        backbone_params = []
        for name, param in model.named_parameters():
            if "head" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
    elif model_name == "efficientnetv2_cbam":
        head_params = []
        backbone_params = []
        for name, param in model.named_parameters():
            if "classifier" in name or "cbam" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
    else:
        head_params = list(model.parameters())
        backbone_params = []

    param_groups = [{"params": head_params, "lr": learning_rate}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": learning_rate * 0.1})

    new_optimizer = optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
    return new_optimizer


def run_training(
    num_epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    pretrained: bool = True,
    processed_dir: str = None,
    model_name: str = None,
    use_focal_loss: bool = True,
    focal_gamma: float = 2.0,
    use_mixup: bool = True,
    use_ema: bool = True,
) -> tuple:
    """
    Full training pipeline with advanced strategies.

    Args:
        num_epochs:    Number of training epochs.
        batch_size:    Batch size.
        learning_rate: AdamW learning rate.
        pretrained:    Whether to use pretrained weights (always True for new models).
        processed_dir: Path to data/processed/.
        model_name:    'convnext_cbam_fpn' or 'swinv2_s'.
        use_focal_loss: Use Focal Loss (True) or Cross-Entropy (False).
        focal_gamma:   Focal loss gamma parameter.
        use_mixup:     Enable Mixup/CutMix augmentation.
        use_ema:       Enable EMA model averaging.

    Returns:
        (trained_model, history_dict)
    """
    set_seed(config.RANDOM_SEED)

    if num_epochs is None:
        num_epochs = config.NUM_EPOCHS
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE
    if processed_dir is None:
        processed_dir = config.PROCESSED_DATA_DIR
    if model_name is None:
        model_name = config.ACTIVE_MODEL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model: {model_name}")
    print(f"Training config: epochs={num_epochs}, batch={batch_size}, lr={learning_rate}")
    print(f"Mixup: {use_mixup}, EMA: {use_ema}, Focal Loss: {use_focal_loss}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(
        processed_dir=processed_dir, batch_size=batch_size, model_name=model_name
    )
    if len(train_loader.dataset) == 0:
        raise RuntimeError(
            "Training dataset is empty. "
            "Please run data_utils.split_dataset() to populate data/processed/."
        )

    # ── Model ─────────────────────────────────────────────────────────────
    if model_name == "convnext_cbam_fpn":
        model = get_convnext_cbam_fpn(num_classes=config.NUM_CLASSES, freeze=True)
        checkpoint_path = config.CONVNEXT_CHECKPOINT
        ema_checkpoint_path = config.EMA_CONVNEXT_CHECKPOINT
    elif model_name == "swinv2_s":
        model = get_swinv2_s(num_classes=config.NUM_CLASSES, freeze=True)
        checkpoint_path = config.SWINV2_CHECKPOINT
        ema_checkpoint_path = config.EMA_SWINV2_CHECKPOINT
    elif model_name == "efficientnetv2_cbam":
        model = get_efficientnetv2_cbam(num_classes=config.NUM_CLASSES, freeze=True)
        checkpoint_path = config.EFFNETV2_CHECKPOINT
        ema_checkpoint_path = config.EMA_EFFNETV2_CHECKPOINT
    elif model_name == "efficientnet_b3":
        model = get_efficientnet_b3(num_classes=config.NUM_CLASSES)
        checkpoint_path = config.EFFICIENTNET_CHECKPOINT
        ema_checkpoint_path = None
    elif model_name == "vit_b_16":
        model = get_vit_b_16(num_classes=config.NUM_CLASSES)
        checkpoint_path = config.VIT_CHECKPOINT
        ema_checkpoint_path = None
    else:
        model = get_model(pretrained=pretrained, num_classes=config.NUM_CLASSES)
        checkpoint_path = config.ALEXNET_CHECKPOINT
        ema_checkpoint_path = None
    model = model.to(device)

    # ── EMA ───────────────────────────────────────────────────────────────
    ema = EMAModel(model, decay=config.EMA_DECAY) if use_ema else None

    # ── Optimizer ─────────────────────────────────────────────────────────
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=config.WEIGHT_DECAY,
    )

    # ── Loss ──────────────────────────────────────────────────────────────
    if use_focal_loss:
        criterion = FocalLoss(gamma=focal_gamma, label_smoothing=config.LABEL_SMOOTHING)
        print(f"Using Focal Loss (gamma={focal_gamma}, label_smoothing={config.LABEL_SMOOTHING})")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
        print(f"Using Cross-Entropy Loss (label_smoothing={config.LABEL_SMOOTHING})")

    # ── AMP GradScaler for mixed precision ────────────────────────────────
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # ── Scheduler: warmup + cosine decay ──────────────────────────────────
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=config.WARMUP_EPOCHS,
        total_epochs=num_epochs,
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_ema_val_acc = 0.0
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)

    print(f"\nStarting training for {num_epochs} epochs...\n")
    for epoch in range(1, num_epochs + 1):
        # ── Progressive unfreezing ────────────────────────────────────────
        if epoch == config.UNFREEZE_EPOCH + 1:
            optimizer = unfreeze_backbone(model, model_name, learning_rate, optimizer)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                warmup_epochs=2,
                total_epochs=max(1, num_epochs - epoch),
            )
            # Reinitialize EMA with all parameters now trainable
            if use_ema:
                ema = EMAModel(model, decay=config.EMA_DECAY)
            print(f"  --> Epoch {epoch}: Unfroze backbone for full fine-tuning")

        # ── Train ─────────────────────────────────────────────────────────
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            ema=ema, use_mixup=use_mixup, epoch=epoch, scaler=scaler,
        )

        # ── Validate (raw model) ─────────────────────────────────────────
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        lr_current = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch:>3}/{num_epochs}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}  "
            f"LR: {lr_current:.2e}"
        )

        # Save best raw model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_path)
            print(f"  --> Best model saved (val_acc={val_acc:.4f})")

        # ── Validate EMA model ────────────────────────────────────────────
        if ema is not None:
            ema.apply(model)
            ema_val_loss, ema_val_acc = validate(model, val_loader, criterion, device)
            ema.restore(model)

            if ema_val_acc > best_ema_val_acc and ema_checkpoint_path:
                best_ema_val_acc = ema_val_acc
                # Save EMA weights
                ema.apply(model)
                save_checkpoint(model, optimizer, epoch, ema_val_acc, ema_checkpoint_path)
                ema.restore(model)
                print(f"  --> Best EMA model saved (ema_val_acc={ema_val_acc:.4f})")

            if epoch % 5 == 0 or epoch == num_epochs:
                print(f"       EMA val_acc: {ema_val_acc:.4f}")

    print(f"\nTraining complete.")
    print(f"  Best raw validation accuracy: {best_val_acc:.4f}")
    if ema is not None:
        print(f"  Best EMA validation accuracy: {best_ema_val_acc:.4f}")
    return model, history


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    val_acc: float,
    path: str,
) -> None:
    """Save model and optimizer state dicts."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
        },
        path,
    )


def load_checkpoint(path: str, model: nn.Module, optimizer=None) -> int:
    """
    Load model (and optionally optimizer) state from a checkpoint.

    Returns:
        epoch number of the saved checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    val_acc = checkpoint.get("val_acc", 0.0)
    print(f"Checkpoint loaded: epoch={epoch}, val_acc={val_acc:.4f}")
    return epoch


if __name__ == "__main__":
    # Train ConvNeXt-CBAM-FPN (primary model)
    print("=" * 60)
    print("Training MSAFusionNet (ConvNeXt-Base + CBAM + FPN)")
    print("=" * 60)
    model, history = run_training(model_name="convnext_cbam_fpn")

    from utils.plot_utils import plot_training_history
    plot_training_history(
        history,
        save_path=os.path.join(config.PLOTS_DIR, "training_history_convnext.png"),
    )

    # Train SwinV2-Small (ensemble partner)
    print("\n" + "=" * 60)
    print("Training SwinV2-Small (ensemble partner)")
    print("=" * 60)
    model2, history2 = run_training(model_name="swinv2_s")

    plot_training_history(
        history2,
        save_path=os.path.join(config.PLOTS_DIR, "training_history_swinv2.png"),
    )
