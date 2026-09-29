"""SkyScenes semantic-seg dataset: RGB-palette mask -> train-id, leakage-free town split.

Images/masks live in ``<prepared>/Images/<Town>/<id>_clrnoon.png`` and
``<prepared>/Segment/<Town>/<id>_semsegCarla_clrnoon.png`` (paired by numeric id within a
town). Masks are CARLA-palette RGB -> converted to train ids via the palette (unknown/void ->
IGNORE_INDEX). Split is by TOWN so no town appears in both train and val (no scene leakage).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import taxonomy as tx

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_ID = re.compile(r"(\d+)")


def _pairs_for_town(images_root: Path, segment_root: Path, town: str) -> list[dict]:
    imgs = {(_ID.search(p.name).group(1)): p for p in (images_root / town).glob("*.png")}
    msks = {(_ID.search(p.name).group(1)): p for p in (segment_root / town).glob("*.png")}
    return [{"image": str(imgs[k]), "mask": str(msks[k]), "town": town}
            for k in sorted(imgs.keys() & msks.keys())]


def discover(prepared_root: str | Path, towns: list[str]) -> list[dict]:
    prepared_root = Path(prepared_root)
    out = []
    for t in towns:
        out += _pairs_for_town(prepared_root / "Images", prepared_root / "Segment", t)
    return out


def town_split(prepared_root: str | Path, val_towns: list[str]) -> tuple[list[dict], list[dict]]:
    """All towns present -> (train pairs, val pairs); val_towns held out entirely."""
    prepared_root = Path(prepared_root)
    all_towns = sorted(p.name for p in (prepared_root / "Images").iterdir() if p.is_dir())
    train_towns = [t for t in all_towns if t not in val_towns]
    return discover(prepared_root, train_towns), discover(prepared_root, val_towns)


def rgb_to_trainid(mask_rgb: np.ndarray, color2id: dict) -> np.ndarray:
    """HxWx3 RGB palette mask -> HxW uint8 train ids (unknown colour -> IGNORE_INDEX)."""
    a = mask_rgb.astype(np.int64)
    packed = (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]
    uniq, inv = np.unique(packed, return_inverse=True)
    lut = np.empty(len(uniq), np.uint8)
    for i, u in enumerate(uniq.tolist()):
        lut[i] = color2id.get(((u >> 16) & 255, (u >> 8) & 255, u & 255), tx.IGNORE_INDEX)
    return lut[inv].reshape(packed.shape)


def build_transforms(train: bool, crop: int = 512, aug: str = "standard"):
    """train aug: ``standard`` (light photometric) | ``analog`` (heavy analog-FPV domain rand)."""
    import albumentations as A
    import cv2
    from albumentations.pytorch import ToTensorV2

    if not train:
        return A.Compose([A.Resize(crop, crop), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])

    geom = [
        A.RandomResizedCrop(size=(crop, crop), scale=(0.2, 1.0), ratio=(0.75, 1.33),
                            interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),                                     # NO vertical flip (sky/road)
    ]
    if aug == "analog":
        from .aug import analog_fpv_photometric
        photo = analog_fpv_photometric()
    else:
        photo = [
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=8, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ]
    return A.Compose(geom + photo + [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])


class SkyScenesDataset(Dataset):
    def __init__(self, pairs: list[dict], train: bool, crop: int = 512, palette_path=None,
                 aug: str = "standard"):
        self.pairs = pairs
        self.tf = build_transforms(train, crop, aug)
        self.color2id = tx.color_to_trainid(palette_path) if palette_path else tx.color_to_trainid()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        from PIL import Image

        e = self.pairs[i]
        img = np.asarray(Image.open(e["image"]).convert("RGB"))
        mask_rgb = np.asarray(Image.open(e["mask"]).convert("RGB"))
        label = rgb_to_trainid(mask_rgb, self.color2id)
        out = self.tf(image=img, mask=label)
        return {"pixel_values": out["image"], "labels": out["mask"].long()}
