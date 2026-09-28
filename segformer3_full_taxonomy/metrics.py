"""Streaming confusion matrix -> per-class IoU + mIoU (ignore_index aware)."""
from __future__ import annotations

import numpy as np

from .taxonomy import CLASSES, IGNORE_INDEX, NUM_CLASSES


class ConfusionMatrix:
    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX,
                 names: list[str] | None = None):
        self.n = num_classes
        self.ignore = ignore_index
        self.names = names or CLASSES
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, target) -> None:
        pred = np.asarray(pred).ravel()
        target = np.asarray(target).ravel()
        keep = (target != self.ignore) & (target < self.n)
        k = self.n * target[keep] + pred[keep]
        self.mat += np.bincount(k, minlength=self.n ** 2).reshape(self.n, self.n)

    def iou(self) -> np.ndarray:
        tp = np.diag(self.mat).astype(np.float64)
        denom = self.mat.sum(0) + self.mat.sum(1) - tp
        return np.divide(tp, denom, out=np.full(self.n, np.nan), where=denom > 0)

    def summary(self) -> dict:
        iou = self.iou()
        return {"mIoU": float(np.nanmean(iou)),
                "per_class_iou": {c: (None if np.isnan(v) else float(v))
                                  for c, v in zip(self.names, iou)}}
