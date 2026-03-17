"""
Focal Loss for handling class imbalance in landslide classification.

Focal Loss down-weights easy (well-classified) examples and focuses training
on hard (misclassified) examples. This is especially useful when the majority
class (non_landslide ~72%) dominates the standard Cross-Entropy gradient.

Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha:     Per-class weight tensor of shape (C,), or a scalar applied
                   uniformly. None = no class weighting.
        gamma:     Focusing parameter. gamma=0 recovers standard CE.
                   Higher gamma → more focus on hard examples.
        reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  Raw model outputs, shape (N, C).
            targets: Ground-truth class indices, shape (N,).

        Returns:
            Scalar loss (if reduction='mean' or 'sum'), else (N,) tensor.
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")  # (N,)
        p_t = torch.exp(-ce_loss)  # probability of the true class

        focal_weight = (1.0 - p_t) ** self.gamma  # (N,)

        loss = focal_weight * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]  # (N,)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
