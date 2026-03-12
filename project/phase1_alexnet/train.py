"""
Training loop for AlexNet landslide classifier.
Saves the best checkpoint based on validation accuracy.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from phase1_alexnet.dataset import get_dataloaders
from phase1_alexnet.model import get_model, get_efficientnet_b3


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """
    Run one training epoch.

    Returns:
        (avg_loss, accuracy) for the epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
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
    Run validation / test epoch without updating parameters.

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
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def run_training(
    num_epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    pretrained: bool = False,
    processed_dir: str = None,
    model_name: str = None,
) -> tuple:
    """
    Full training pipeline.

    Args:
        num_epochs:    Number of training epochs.
        batch_size:    Batch size.
        learning_rate: Adam learning rate.
        pretrained:    Whether to use ImageNet pretrained weights.
        processed_dir: Path to data/processed/ directory.
        model_name:    'alexnet' or 'efficientnet_b3'. Defaults to config.ACTIVE_MODEL.

    Returns:
        (trained_model, history_dict)
        history_dict keys: 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """
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

    # Data
    train_loader, val_loader = get_dataloaders(
        processed_dir=processed_dir, batch_size=batch_size
    )
    if len(train_loader.dataset) == 0:
        raise RuntimeError(
            "Training dataset is empty. "
            "Please run data_utils.split_dataset() to populate data/processed/."
        )

    # Model
    if model_name == "efficientnet_b3":
        model = get_efficientnet_b3(num_classes=config.NUM_CLASSES)
        checkpoint_path = config.EFFICIENTNET_CHECKPOINT
    else:
        model = get_model(pretrained=pretrained, num_classes=config.NUM_CLASSES)
        checkpoint_path = config.ALEXNET_CHECKPOINT
    model = model.to(device)

    # Optimizer, loss, scheduler
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)

    print(f"\nStarting training for {num_epochs} epochs...\n")
    for epoch in range(1, num_epochs + 1):
        # Unfreeze feature layers after UNFREEZE_EPOCH for full fine-tuning
        if epoch == config.UNFREEZE_EPOCH + 1:
            for param in model.parameters():
                param.requires_grad = True
            # Features get 10x lower LR than the head
            if model_name == "efficientnet_b3":
                head_params = [p for n, p in model.named_parameters() if "classifier" in n]
                feat_params = [p for n, p in model.named_parameters() if "classifier" not in n]
            else:
                head_params = [p for n, p in model.named_parameters() if "classifier" in n]
                feat_params = [p for n, p in model.named_parameters() if "classifier" not in n]
            optimizer = optim.Adam(
                [
                    {"params": feat_params, "lr": learning_rate * 0.1},
                    {"params": head_params, "lr": learning_rate},
                ],
                weight_decay=1e-4,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs - epoch, eta_min=1e-7
            )
            print(f"  --> Epoch {epoch}: Unfroze all layers for full fine-tuning")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch:>3}/{num_epochs}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_path)
            print(f"  --> Best model saved (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")
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
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    val_acc = checkpoint.get("val_acc", 0.0)
    print(f"Checkpoint loaded: epoch={epoch}, val_acc={val_acc:.4f}")
    return epoch


if __name__ == "__main__":
    model, history = run_training()
