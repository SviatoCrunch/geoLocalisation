"""Per-frame day/night tagging by chroma — ported from
``geoLocalisation/src_tif_build_universal/classify_night.py`` (itself a port of
``main_try/chroma_day_night.ipynb``; rule-based, no weights, validated acc 0.986).

    downscale to 256 px long side (INTER_AREA),
    chroma = mean over pixels & channels of |channel - pixel_gray|,
    NIGHT if chroma < THRESHOLD (default 12).

Here it takes an in-memory BGR frame and RETURNS a label (we do not rename any
files — the label goes into the manifest JSON).
"""
from __future__ import annotations

import cv2
import numpy as np

THRESHOLD = 12.0


def chroma(img_bgr: np.ndarray | None, long_side: int = 256) -> float | None:
    """Mean |channel - per-pixel-gray| on a 256px-long-side copy. None if unreadable."""
    img = img_bgr
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(
            img, (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    f = img.astype(np.float32)
    gray = f.mean(axis=2, keepdims=True)          # per-pixel gray = mean of B,G,R
    return float(np.abs(f - gray).mean())         # mean |channel - gray|


def day_night(img_bgr: np.ndarray | None, thr: float = THRESHOLD) -> tuple[str, float | None]:
    """Return ``("day"|"night"|"unknown", chroma_value)``. NIGHT if chroma < thr."""
    c = chroma(img_bgr)
    if c is None:
        return "unknown", None
    return ("night" if c < thr else "day"), c
