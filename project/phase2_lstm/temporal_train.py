"""
Training loop for Bi-LSTM + Attention temporal forecasting model.

Multi-task training with three losses:
  1. Type classification loss (CrossEntropy) — predict next landslide type
  2. Occurrence probability loss (BCE) — predict if event will occur
  3. Forecast loss (CrossEntropy) — predict types for next N steps

Uses early stopping on validation loss.
"""

import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from phase2_lstm.temporal_model import BiLSTMAttentionModel
from phase2_lstm.data_preprocessing import (
    load_glc_data,
    encode_features,
    build_sequences,
    LandslideTemporalDataset,
    collate_fn,
    FEATURE_DIM,
    N_TYPES,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_temporal_model(
    num_epochs: int = None,
    learning_rate: float = None,
    batch_size: int = 32,
    hidden_dim: int = None,
    num_layers: int = None,
    num_heads: int = None,
    n_forecast_steps: int = None,
    dropout: float = None,
    patience: int = 15,
):
    """
    Train the Bi-LSTM + Attention model on NASA GLC data.

    Returns:
        (model, history, type_encoder, trigger_encoder)
    """
    set_seed(config.RANDOM_SEED)

    if num_epochs is None:
        num_epochs = config.LSTM_NUM_EPOCHS
    if learning_rate is None:
        learning_rate = config.LSTM_LEARNING_RATE
    if hidden_dim is None:
        hidden_dim = config.LSTM_HIDDEN_DIM
    if num_layers is None:
        num_layers = config.LSTM_NUM_LAYERS
    if num_heads is None:
        num_heads = config.LSTM_NUM_HEADS
    if n_forecast_steps is None:
        n_forecast_steps = config.LSTM_FORECAST_STEPS
    if dropout is None:
        dropout = config.LSTM_DROPOUT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load & preprocess data ────────────────────────────────────────────
    print("Loading NASA GLC data...")
    df = load_glc_data()
    df_enc, type_encoder, trigger_encoder = encode_features(df)

    print("Building temporal sequences...")
    sequences = build_sequences(df_enc, type_encoder, trigger_encoder)

    dataset = LandslideTemporalDataset(sequences, n_forecast_steps=n_forecast_steps)

    # Split: 80% train, 20% validation
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )
    print(f"Train: {n_train} sequences, Val: {n_val} sequences")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = BiLSTMAttentionModel(
        input_dim=FEATURE_DIM,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        n_types=N_TYPES,
        n_forecast_steps=n_forecast_steps,
        dropout=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total_params:,} params ({trainable:,} trainable)")

    # ── Loss functions ────────────────────────────────────────────────────
    type_criterion = nn.CrossEntropyLoss()
    occ_criterion = nn.BCELoss()
    forecast_criterion = nn.CrossEntropyLoss()

    # Loss weights (type classification is most important)
    w_type = 1.0
    w_occ = 0.5
    w_forecast = 0.3

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    # ── Training loop ─────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_no_improve = 0

    checkpoint_path = config.LSTM_MODEL_PATH
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    print(f"\nTraining for {num_epochs} epochs (patience={patience})...\n")

    for epoch in range(1, num_epochs + 1):
        # ── Train ─────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"  Train E{epoch}", leave=False):
            features = batch["features"].to(device)
            target_type = batch["target_type"].to(device)
            target_occ = batch["target_occ"].to(device)
            forecast_targets = batch["forecast_targets"].to(device)
            lengths = batch["lengths"].to(device)

            optimizer.zero_grad()

            type_logits, occ_prob, forecast_logits, _ = model(features, lengths)

            # Multi-task loss
            loss_type = type_criterion(type_logits, target_type)
            loss_occ = occ_criterion(occ_prob.squeeze(-1), target_occ)

            # Forecast loss: average across steps
            loss_forecast = 0
            for step in range(n_forecast_steps):
                loss_forecast += forecast_criterion(
                    forecast_logits[:, step, :], forecast_targets[:, step]
                )
            loss_forecast /= n_forecast_steps

            loss = w_type * loss_type + w_occ * loss_occ + w_forecast * loss_forecast
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * features.size(0)
            preds = type_logits.argmax(dim=1)
            correct += (preds == target_type).sum().item()
            total += features.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # ── Validate ──────────────────────────────────────────────────
        model.eval()
        val_total_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"  Val   E{epoch}", leave=False):
                features = batch["features"].to(device)
                target_type = batch["target_type"].to(device)
                target_occ = batch["target_occ"].to(device)
                forecast_targets = batch["forecast_targets"].to(device)
                lengths = batch["lengths"].to(device)

                type_logits, occ_prob, forecast_logits, _ = model(features, lengths)

                loss_type = type_criterion(type_logits, target_type)
                loss_occ = occ_criterion(occ_prob.squeeze(-1), target_occ)
                loss_forecast = 0
                for step in range(n_forecast_steps):
                    loss_forecast += forecast_criterion(
                        forecast_logits[:, step, :], forecast_targets[:, step]
                    )
                loss_forecast /= n_forecast_steps
                loss = w_type * loss_type + w_occ * loss_occ + w_forecast * loss_forecast

                val_total_loss += loss.item() * features.size(0)
                preds = type_logits.argmax(dim=1)
                val_correct += (preds == target_type).sum().item()
                val_total += features.size(0)

        val_loss = val_total_loss / val_total
        val_acc = val_correct / val_total

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        lr_current = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch:>3}/{num_epochs}]  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  "
            f"LR: {lr_current:.2e}"
        )

        # ── Checkpoint & Early Stopping ───────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "model_config": {
                    "input_dim": FEATURE_DIM,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "n_types": N_TYPES,
                    "n_forecast_steps": n_forecast_steps,
                    "dropout": dropout,
                },
            }, checkpoint_path)
            print(f"  --> Best model saved (val_loss={val_loss:.4f}, val_acc={val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}, Best val_acc: {best_val_acc:.4f}")

    # Save encoders
    import joblib
    joblib.dump(type_encoder, config.LSTM_TYPE_ENCODER_PATH)
    joblib.dump(trigger_encoder, config.LSTM_TRIGGER_ENCODER_PATH)
    print(f"Encoders saved.")

    return model, history, type_encoder, trigger_encoder


def load_trained_model(device=None):
    """Load a trained Bi-LSTM + Attention model from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(config.LSTM_MODEL_PATH, map_location=device, weights_only=False)
    cfg = checkpoint["model_config"]

    model = BiLSTMAttentionModel(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        n_types=cfg["n_types"],
        n_forecast_steps=cfg["n_forecast_steps"],
        dropout=cfg.get("dropout", 0.3),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_acc = checkpoint.get("val_acc", 0)
    print(f"Loaded Bi-LSTM+Attention model (val_acc={val_acc:.4f})")
    return model


if __name__ == "__main__":
    model, history, type_enc, trig_enc = train_temporal_model()

    from utils.plot_utils import plot_training_history
    plot_training_history(
        history,
        save_path=os.path.join(config.PLOTS_DIR, "training_history_lstm.png"),
    )
