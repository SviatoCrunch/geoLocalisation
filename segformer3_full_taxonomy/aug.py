"""Analog-FPV domain-randomization augmentation (image-only) to close the synthetic->real gap.

The real Kramatorsk frames come from analog 5.8 GHz FPV video, so clean CARLA renders must be
degraded toward that look during training: chroma bleed/crawl + chroma noise (composite
cross-chroma), RF horizontal streaks, low resolution (downscale), transmission compression
blockiness, motion/defocus blur, sensor noise, and colour/tone drift (reduced gamut). Grounded
in domain-randomisation for sim-to-real (heavy random photometric) + analog/VHS artifact
simulation (noise/blur/colour degradation, cross-chroma, low res, compression).

All transforms touch only the IMAGE; masks pass through unchanged.
"""
from __future__ import annotations

import numpy as np


def _chroma_bleed(image, **_):
    """Composite cross-chroma: blur/shift/noise the Cr,Cb channels (luma kept sharp)."""
    import cv2
    ycc = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    kx = int(np.random.choice([3, 5, 7, 9]))                       # horizontal chroma blur (crawl)
    ycc[..., 1] = cv2.blur(ycc[..., 1], (kx, 1))
    ycc[..., 2] = cv2.blur(ycc[..., 2], (kx, 1))
    shift = int(np.random.randint(-3, 4))                          # small chroma horizontal shift
    if shift:
        ycc[..., 1] = np.roll(ycc[..., 1], shift, axis=1)
        ycc[..., 2] = np.roll(ycc[..., 2], -shift, axis=1)
    ycc[..., 1:] += np.random.randn(*ycc[..., 1:].shape).astype(np.float32) * np.random.uniform(2, 8)
    return cv2.cvtColor(np.clip(ycc, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)


def _rf_lines(image, **_):
    """RF interference: a few bright/dark horizontal streaks + occasional row tearing."""
    out = image.copy()
    h, w = out.shape[:2]
    for _ in range(int(np.random.randint(1, 6))):
        y = int(np.random.randint(0, h)); th = int(np.random.randint(1, 4))
        delta = int(np.random.randint(-60, 61))
        out[y:y + th] = np.clip(out[y:y + th].astype(np.int16) + delta, 0, 255).astype(np.uint8)
    if np.random.rand() < 0.3:                                     # horizontal tear of a band
        y = int(np.random.randint(0, h - 8)); bh = int(np.random.randint(4, 16))
        out[y:y + bh] = np.roll(out[y:y + bh], int(np.random.randint(-12, 13)), axis=1)
    return out


def analog_fpv_photometric() -> list:
    """Heavy analog-FPV degradation block (list of Albumentations transforms, image-only)."""
    import albumentations as A

    return [
        A.OneOf([A.MotionBlur(blur_limit=(3, 9)), A.GaussianBlur(blur_limit=(3, 7)),
                 A.Defocus(radius=(2, 5))], p=0.5),
        A.Downscale(scale_range=(0.25, 0.6), p=0.5),               # low analog resolution
        A.Lambda(image=_chroma_bleed, name="chroma_bleed", p=0.6),
        A.Lambda(image=_rf_lines, name="rf_lines", p=0.4),
        A.OneOf([A.ISONoise(intensity=(0.2, 0.6)),
                 A.MultiplicativeNoise(multiplier=(0.85, 1.15), per_channel=True),
                 A.GaussNoise()], p=0.5),
        A.ImageCompression(quality_range=(20, 60), p=0.5),         # transmission blockiness
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.RandomGamma(gamma_limit=(70, 140), p=0.3),
        A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=30, val_shift_limit=12, p=0.5),
        A.RandomToneCurve(scale=0.2, p=0.3),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.2),
        A.RingingOvershoot(p=0.2),                                 # analog sharpening halos
    ]
