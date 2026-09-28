"""Taxonomy, RGB->trainid, town split, losses, metrics."""
import numpy as np
import torch
from PIL import Image

from segformer3_full_taxonomy import taxonomy as tx
from segformer3_full_taxonomy.dataset import discover, rgb_to_trainid, town_split
from segformer3_full_taxonomy.losses import CombinedLoss
from segformer3_full_taxonomy.metrics import ConfusionMatrix


def test_taxonomy_22_classes_void_ignored():
    assert tx.NUM_CLASSES == 22
    assert "unlabeled" not in tx.CLASSES
    c2i = tx.color_to_trainid()
    assert c2i[(128, 64, 128)] == tx.CLASS_TO_ID["road"]
    assert c2i[(0, 0, 0)] == tx.IGNORE_INDEX          # unlabeled/void
    assert tx.CLASS_TO_ID["railtrack"] < 22 and tx.CLASS_TO_ID["water"] < 22


def test_rgb_to_trainid_unknown_to_ignore():
    m = np.zeros((3, 3, 3), np.uint8)
    m[0] = (128, 64, 128)     # road
    m[1] = (45, 60, 150)      # water
    m[2] = (7, 7, 7)          # not in palette -> ignore
    out = rgb_to_trainid(m, tx.color_to_trainid())
    assert (out[0] == tx.CLASS_TO_ID["road"]).all()
    assert (out[1] == tx.CLASS_TO_ID["water"]).all()
    assert (out[2] == tx.IGNORE_INDEX).all()


def test_town_split_no_leakage(tmp_path):
    for kind in ("Images", "Segment"):
        for town in ("Town01", "Town02", "Town03"):
            d = tmp_path / kind / town; d.mkdir(parents=True)
            suff = "_clrnoon.png" if kind == "Images" else "_semsegCarla_clrnoon.png"
            for i in (1, 2):
                Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(d / f"00{i}{suff}")
    tr, va = town_split(tmp_path, val_towns=["Town03"])
    tr_towns = {p["town"] for p in tr}; va_towns = {p["town"] for p in va}
    assert va_towns == {"Town03"} and "Town03" not in tr_towns
    assert not (tr_towns & va_towns)                  # leakage-free
    assert len(tr) == 4 and len(va) == 2


def test_combined_loss_and_confusion():
    logits = torch.randn(2, tx.NUM_CLASSES, 8, 8, requires_grad=True)
    target = torch.randint(0, tx.NUM_CLASSES, (2, 8, 8))
    target[:, 0, :] = tx.IGNORE_INDEX
    loss = CombinedLoss(dice_w=1.0)(logits, target)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad is not None
    cm = ConfusionMatrix()
    gt = np.array([[0, 1], [2, tx.IGNORE_INDEX]])
    cm.update(gt, gt)
    s = cm.summary()
    assert abs(s["per_class_iou"][tx.CLASSES[0]] - 1.0) < 1e-9
