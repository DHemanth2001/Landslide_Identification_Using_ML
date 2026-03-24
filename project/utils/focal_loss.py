"""
Focal Loss with Label Smoothing for handling class imbalance.

Focal Loss down-weights easy (well-classified) examples and focuses training
on hard (misclassified) examples. Combined with label smoothing for better
calibration and generalization.

References:
  - Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
  - Muller et al., "When Does Label Smoothing Help?", NeurIPS 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    With optional label smoothing: targets become (1 - eps) * one_hot + eps / C

    Args:
        alpha:           Per-class weight tensor of shape (C,), or None.
        gamma:           Focusing parameter. gamma=0 recovers standard CE.
        reduction:       'mean' | 'sum' | 'none'.
        label_smoothing: Smoothing factor (0 = no smoothing, 0.1 = recommended).
    """

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean",
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

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
        ce_loss = F.cross_entropy(
            logits, targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        p_t = torch.exp(-ce_loss)

        focal_weight = (1.0 - p_t) ** self.gamma

        loss = focal_weight * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
