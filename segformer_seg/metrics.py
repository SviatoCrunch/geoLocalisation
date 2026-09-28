"""Streaming confusion matrix -> per-class IoU + mIoU (ignore_index aware)."""
from __future__ import annotations

import numpy as np

from .config import CLASSES, IGNORE_INDEX, NUM_CLASSES


class ConfusionMatrix:
    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX):
        self.n = num_classes
        self.ignore = ignore_index
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, target) -> None:
        """pred/target: int arrays (any shape), same shape."""
        pred = np.asarray(pred).ravel()
        target = np.asarray(target).ravel()
        keep = target != self.ignore
        pred, target = pred[keep], target[keep]
        k = self.n * target + pred
        self.mat += np.bincount(k, minlength=self.n ** 2).reshape(self.n, self.n)

    def iou(self) -> np.ndarray:
        tp = np.diag(self.mat).astype(np.float64)
        fp = self.mat.sum(0) - tp
        fn = self.mat.sum(1) - tp
        denom = tp + fp + fn
        return np.divide(tp, denom, out=np.full(self.n, np.nan), where=denom > 0)

    def summary(self) -> dict:
        iou = self.iou()
        miou = float(np.nanmean(iou))
        return {"mIoU": miou,
                "per_class_iou": {c: (None if np.isnan(v) else float(v))
                                  for c, v in zip(CLASSES, iou)}}
