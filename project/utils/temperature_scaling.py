"""
Temperature Scaling — post-hoc confidence calibration for Phase 1.

Learned temperature T scales the logits: p = softmax(logits / T)
T > 1 → softer (less confident), T < 1 → sharper.
Optimal T minimises NLL on the validation set.

Reference: Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017)
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class TemperatureScaler(nn.Module):
    """Wraps a trained model and adds a single learnable temperature parameter."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._scale(self.model(x))

    def _scale(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return calibrated probabilities."""
        return torch.softmax(self._scale(self.model(x)), dim=1)


def fit_temperature(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    lr: float = 0.01,
    max_iter: int = 50,
) -> TemperatureScaler:
    """
    Fit temperature T on validation set logits by minimising NLL.

    Args:
        model:      Trained classifier (already in eval mode).
        val_loader: Validation DataLoader.
        device:     torch.device.
        lr:         Learning rate for L-BFGS.
        max_iter:   Max L-BFGS iterations.

    Returns:
        Fitted TemperatureScaler wrapping the original model.
    """
    model.eval()
    scaler = TemperatureScaler(model).to(device)

    # Collect all logits and labels from the validation set
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits).to(device)
    all_labels = torch.cat(all_labels).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.LBFGS(
        [scaler.temperature], lr=lr, max_iter=max_iter
    )

    def eval_step():
        optimizer.zero_grad()
        loss = criterion(all_logits / scaler.temperature, all_labels)
        loss.backward()
        return loss

    optimizer.step(eval_step)

    T = scaler.temperature.item()
    print(f"Temperature scaling fitted: T = {T:.4f}")
    return scaler


def save_temperature(T: float, path: str = None) -> None:
    if path is None:
        path = os.path.join(config.CHECKPOINTS_DIR, "temperature.pt")
    torch.save({"temperature": T}, path)
    print(f"Temperature saved → {path}")


def load_temperature(path: str = None) -> float:
    if path is None:
        path = os.path.join(config.CHECKPOINTS_DIR, "temperature.pt")
    if not os.path.exists(path):
        return 1.0  # no calibration → identity
    data = torch.load(path, map_location="cpu")
    T = float(data["temperature"])
    print(f"Temperature loaded: T = {T:.4f}")
    return T
