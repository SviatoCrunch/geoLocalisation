"""Weighted CE + soft-Dice for imbalanced multi-class segmentation (ignore_index aware)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .taxonomy import IGNORE_INDEX, NUM_CLASSES


class DiceLoss(nn.Module):
    def __init__(self, ignore_index: int = IGNORE_INDEX, eps: float = 1.0):
        super().__init__()
        self.ignore_index, self.eps = ignore_index, eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        valid = (target != self.ignore_index)
        tgt = target.clone(); tgt[~valid] = 0
        onehot = F.one_hot(tgt, NUM_CLASSES).permute(0, 3, 1, 2).float() * valid.unsqueeze(1)
        probs = probs * valid.unsqueeze(1)
        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)
        card = probs.sum(dims) + onehot.sum(dims)
        present = onehot.sum(dims) > 0
        dice = (2 * inter + self.eps) / (card + self.eps)
        return 1.0 - dice[present].mean() if present.any() else logits.sum() * 0.0


class CombinedLoss(nn.Module):
    """``ce_w * weighted_CE + dice_w * Dice``."""

    def __init__(self, class_weights=None, ce_w: float = 1.0, dice_w: float = 1.0,
                 ignore_index: int = IGNORE_INDEX):
        super().__init__()
        w = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.ce = nn.CrossEntropyLoss(weight=w, ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.ce_w, self.dice_w = ce_w, dice_w

    def forward(self, logits, target):
        loss = self.ce_w * self.ce(logits, target)
        if self.dice_w:
            loss = loss + self.dice_w * self.dice(logits, target)
        return loss
