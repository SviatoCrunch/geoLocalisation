"""External validation on gt_cramatorsc/GT_flat_mask — MONITOR ONLY (road/railtrack/water).

Kramatorsk stills (``GT_flat/<N>_<lat>_<lon>.jpg``) with per-class binary masks
(``GT_flat_mask/<lat>_<lon>__<Class>.png``). We score the model's 22-class prediction against
the three targets one-vs-rest, with **sky ignored** and any unlabelled pixel treated as
negative (GT_flat_mask road/rail/water annotations are taken as complete per frame). A target
absent from a frame is skipped (aggregated over frames where it is present).

This is the held-out TEST set: it must NOT drive checkpoint selection. train.py logs these
numbers for visibility but keeps ``best/`` chosen by source-validation mIoU.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch

from . import taxonomy as tx
from .dataset import IMAGENET_MEAN, IMAGENET_STD

_STILL = re.compile(r"^\d+_([0-9.]+)_([0-9.]+)\.jpg$", re.I)
_TARGET_RAW = {"Road": "road", "Railways": "railtrack", "Water": "water"}


def external_pairs(gt_root: str | Path) -> list[dict]:
    """[{image, masks:{RawClass: path}}] for every GT_flat still that has >=1 mask."""
    gt_root = Path(gt_root)
    flat, mdir = gt_root / "GT_flat", gt_root / "GT_flat_mask"
    pairs = []
    for img in sorted(flat.glob("*.jpg")):
        m = _STILL.match(img.name)
        if not m:
            continue
        key = f"{m.group(1)}_{m.group(2)}"
        masks = {cls: str(mdir / f"{key}__{cls}.png")
                 for cls in ("Road", "Railways", "Water", "Sky")
                 if (mdir / f"{key}__{cls}.png").exists()}
        if masks:
            pairs.append({"image": str(img), "masks": masks})
    return pairs


def _preprocess(img_pil, crop: int) -> torch.Tensor:
    im = np.asarray(img_pil.convert("RGB").resize((crop, crop))).astype(np.float32) / 255.0
    im = (im - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    return torch.from_numpy(im.transpose(2, 0, 1)).unsqueeze(0).float()


def _load_bin(path, h, w) -> np.ndarray:
    from PIL import Image
    if not path:
        return np.zeros((h, w), bool)
    m = np.asarray(Image.open(path).convert("L"))
    if m.shape != (h, w):
        m = np.asarray(Image.fromarray(m).resize((w, h), Image.NEAREST))
    return m > 0


@torch.no_grad()
def evaluate_external(model, pairs: list[dict], device, crop: int = 512) -> dict:
    """Per-target IoU (road/railtrack/water) + macro over present. Sky pixels ignored."""
    from PIL import Image

    model.eval()
    stat = {name: {"tp": 0, "fp": 0, "fn": 0} for name in _TARGET_RAW.values()}
    for e in pairs:
        img = Image.open(e["image"]).convert("RGB")
        w, h = img.size
        logits = torch.nn.functional.interpolate(
            model(pixel_values=_preprocess(img, crop).to(device)).logits, size=(h, w),
            mode="bilinear", align_corners=False)
        pred = logits.argmax(1)[0].cpu().numpy()
        valid = ~_load_bin(e["masks"].get("Sky"), h, w)          # sky -> ignore
        for raw, name in _TARGET_RAW.items():
            if raw not in e["masks"]:
                continue
            gt = _load_bin(e["masks"][raw], h, w) & valid
            pr = (pred == tx.CLASS_TO_ID[name]) & valid
            stat[name]["tp"] += int((gt & pr).sum())
            stat[name]["fp"] += int((~gt & pr).sum())
            stat[name]["fn"] += int((gt & ~pr).sum())
    out = {}
    for name, s in stat.items():
        d = s["tp"] + s["fp"] + s["fn"]
        out[name] = (s["tp"] / d) if d > 0 else None
    present = [v for v in out.values() if v is not None]
    out["macro_target"] = float(np.mean(present)) if present else None
    return out
