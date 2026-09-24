"""Pure-core metrics tests — numpy only."""
from __future__ import annotations

import numpy as np

from keyframe_calib.metrics import confusion, evaluate, iou_f1_from_confusion


def test_perfect_prediction():
    gt = np.array([[0, 1], [2, 1]])
    miou, mf1 = evaluate([gt], [gt], num_classes=3)
    assert miou == 1.0 and mf1 == 1.0


def test_ignore_excluded():
    gt = np.array([[0, 1]])
    pred = np.array([[0, 255]])  # second px ignored on both -> perfect over the rest
    conf = confusion(pred, gt, num_classes=2, ignore=255)
    assert conf[0, 0] == 1 and conf.sum() == 1


def test_half_wrong_binary():
    gt = np.array([[0, 0, 1, 1]])
    pred = np.array([[0, 1, 1, 0]])  # one FP + one FN per class
    miou, mf1 = evaluate([pred], [gt], num_classes=2)
    # each class: tp=1, fp=1, fn=1 -> iou=1/3, f1=0.5
    assert abs(miou - 1 / 3) < 1e-9
    assert abs(mf1 - 0.5) < 1e-9


def test_only_present_classes_averaged():
    # class 2 never appears in GT -> excluded from the mean
    gt = np.array([[0, 1]])
    pred = np.array([[0, 1]])
    conf = confusion(pred, gt, num_classes=3)
    miou, mf1 = iou_f1_from_confusion(conf)
    assert miou == 1.0 and mf1 == 1.0
