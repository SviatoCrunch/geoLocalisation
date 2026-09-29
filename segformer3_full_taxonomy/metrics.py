"""Streaming confusion matrix -> per-class IoU + mIoU (ignore_index aware, active taxonomy)."""
from __future__ import annotations

import numpy as np

from . import taxonomy as tx


class ConfusionMatrix:
    def __init__(self, num_classes: int | None = None, ignore_index: int | None = None,
                 names: list[str] | None = None):
        self.n = tx.NUM_CLASSES if num_classes is None else num_classes
        self.ignore = tx.IGNORE_INDEX if ignore_index is None else ignore_index
        self.names = names or list(tx.CLASSES)
        self.mat = np.zeros((self.n, self.n), dtype=np.int64)

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
