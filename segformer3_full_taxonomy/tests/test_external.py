"""External GT_flat_mask eval: pair discovery + one-vs-rest IoU (sky ignored)."""
import numpy as np
import torch
from PIL import Image

from segformer3_full_taxonomy import taxonomy as tx
from segformer3_full_taxonomy.external import evaluate_external, external_pairs


class _RoadModel:
    """Predicts 'road' everywhere (logits peaked at the road id)."""
    def eval(self):
        return self

    def __call__(self, pixel_values):
        _, _, h, w = pixel_values.shape
        logits = torch.full((1, tx.NUM_CLASSES, h, w), -10.0)
        logits[:, tx.CLASS_TO_ID["road"]] = 10.0
        return type("O", (), {"logits": logits})()


def _mk(gt_root, key="48.5_37.6"):
    flat = gt_root / "GT_flat"; mdir = gt_root / "GT_flat_mask"
    flat.mkdir(parents=True); mdir.mkdir(parents=True)
    Image.fromarray((np.random.rand(10, 10, 3) * 255).astype(np.uint8)).save(flat / f"1_{key}.jpg")
    road = np.zeros((10, 10), np.uint8); road[:5] = 255      # top half = road
    Image.fromarray(road).save(mdir / f"{key}__Road.png")
    Image.fromarray(np.zeros((10, 10), np.uint8)).save(mdir / f"{key}__Sky.png")  # no sky


def test_external_pairs_discovery(tmp_path):
    _mk(tmp_path)
    pairs = external_pairs(tmp_path)
    assert len(pairs) == 1
    assert "Road" in pairs[0]["masks"] and "Sky" in pairs[0]["masks"]


def test_external_road_iou_half(tmp_path):
    _mk(tmp_path)
    pairs = external_pairs(tmp_path)
    out = evaluate_external(_RoadModel(), pairs, device="cpu", crop=8)
    # GT road = top half; pred road = everywhere -> IoU = TP/(TP+FP) = 50/100 = 0.5
    assert abs(out["road"] - 0.5) < 1e-6
    assert out["water"] is None and out["railtrack"] is None      # absent -> N/A
    assert abs(out["macro_target"] - 0.5) < 1e-6                  # macro over present only
