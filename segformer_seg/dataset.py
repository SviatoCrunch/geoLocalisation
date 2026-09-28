"""Torch Dataset over a manifest JSONL: compose label on the fly + augment.

Path-level — nothing is pre-merged on disk. Each ``__getitem__`` reads the still, composes
its index label map from the entry (server PNGs or COCO polygons via ``compose_label``),
augments the (image, mask) pair jointly, and returns tensors ready for SegFormer.

Augmentation (Albumentations) is deliberately UAV-appropriate: multi-scale + crop, small
bank rotation, horizontal flip and photometric jitter — but **no vertical flip** (sky/road
are orientation-bound). Padding fills the mask with ``IGNORE_INDEX`` so padded borders never
supervise. Val/test only resize + normalise.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .build_manifest import compose_label
from .config import IGNORE_INDEX

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RARE_CLASSES = {"bridge", "water", "railway"}  # for the oversampling sampler


def read_manifest(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def entry_classes(entry: dict) -> set[str]:
    """Canonical classes present in an entry — cheap (no image read), for the sampler."""
    if entry["source"] == "server":
        return {c for c, _ in entry["label_pngs"]}
    return {c for c, _ in entry["label_polys"]}


def sampler_weights(entries: list[dict], rare_boost: float = 5.0) -> list[float]:
    """Per-sample weights: images containing any rare class get ``rare_boost``x. Feed to
    ``torch.utils.data.WeightedRandomSampler`` so rare-class frames (the 66 COCO ones) are
    seen far more often than their 18% share would give."""
    return [rare_boost if entry_classes(e) & RARE_CLASSES else 1.0 for e in entries]


def _fill_kw(cls, fill, fill_mask) -> dict:
    """Border-fill kwargs, robust across albumentations versions (``fill``/``fill_mask`` in
    >=1.4/2.x, ``value``/``mask_value`` before). Mask is filled with IGNORE so padded/rotated
    borders never supervise."""
    import inspect
    params = inspect.signature(cls.__init__).parameters
    if "fill" in params:
        return {"fill": fill, "fill_mask": fill_mask}
    return {"value": fill, "mask_value": fill_mask}


def build_transforms(train: bool, crop: int = 512):
    import albumentations as A
    import cv2
    from albumentations.pytorch import ToTensorV2

    if train:
        return A.Compose([
            A.RandomScale(scale_limit=(-0.5, 1.0), p=1.0),                       # 0.5x .. 2.0x
            A.PadIfNeeded(crop, crop, border_mode=cv2.BORDER_CONSTANT,
                          **_fill_kw(A.PadIfNeeded, 0, IGNORE_INDEX)),
            A.RandomCrop(crop, crop),
            A.HorizontalFlip(p=0.5),                                             # NO vertical flip
            A.Rotate(limit=12, border_mode=cv2.BORDER_CONSTANT,
                     **_fill_kw(A.Rotate, 0, IGNORE_INDEX), p=0.3),              # small UAV bank
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                                 val_shift_limit=10, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(crop, crop),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class SegDataset(Dataset):
    def __init__(self, manifest: str | Path, train: bool, crop: int = 512,
                 background_mode: str = "class"):
        self.entries = read_manifest(manifest)
        self.tf = build_transforms(train, crop)
        self.background_mode = background_mode  # "class" | "ignore"

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        from PIL import Image

        e = self.entries[i]
        img = np.asarray(Image.open(e["image"]).convert("RGB"))
        label = compose_label(e)                       # HxW uint8, ids 0..6
        if img.shape[:2] != label.shape:               # safety: align to label grid
            img = np.asarray(Image.fromarray(img).resize((label.shape[1], label.shape[0])))
        out = self.tf(image=img, mask=label)
        lbl = out["mask"].long()
        if self.background_mode == "ignore":
            lbl[lbl == 0] = IGNORE_INDEX
        return {"pixel_values": out["image"], "labels": lbl}
