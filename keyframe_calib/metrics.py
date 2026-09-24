"""Segmentation metrics for the calibration sweep — mean IoU and mean F1 over a
set of frames vs dense ground truth, ignoring the ``ignore`` label. numpy only.

Uses a global confusion matrix accumulated across all scored (pred, gt) frame
pairs so the reported mean-F1 / mean-IoU match the dataset-level definition used
by SegProp / Ruralscapes (per-class, then averaged over classes present in GT).
"""
from __future__ import annotations

import numpy as np


def confusion(pred: np.ndarray, gt: np.ndarray, num_classes: int, ignore: int = 255) -> np.ndarray:
    """(num_classes, num_classes) confusion for one frame; rows=GT, cols=pred.
    Pixels where GT or pred is ``ignore`` (or out of range) are dropped."""
    m = (gt != ignore) & (pred != ignore) & (gt < num_classes) & (pred < num_classes) \
        & (gt >= 0) & (pred >= 0)
    g = gt[m].astype(np.int64)
    p = pred[m].astype(np.int64)
    k = g * num_classes + p
    return np.bincount(k, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def iou_f1_from_confusion(conf: np.ndarray):
    """(mean_iou, mean_f1) averaged over classes that appear in GT (rows with
    support), the convention used for Ruralscapes-style reporting."""
    tp = np.diag(conf).astype(np.float64)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    present = conf.sum(axis=1) > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (tp + fp + fn)
        f1 = 2.0 * tp / (2.0 * tp + fp + fn)
    iou = iou[present]
    f1 = f1[present]
    return (float(np.nanmean(iou)) if iou.size else 0.0,
            float(np.nanmean(f1)) if f1.size else 0.0)


def evaluate(preds, gts, num_classes: int, ignore: int = 255):
    """Accumulate a global confusion over aligned (pred, gt) frame pairs ->
    (mean_iou, mean_f1)."""
    conf = np.zeros((num_classes, num_classes), np.int64)
    for p, g in zip(preds, gts):
        conf += confusion(p, g, num_classes, ignore)
    return iou_f1_from_confusion(conf)
