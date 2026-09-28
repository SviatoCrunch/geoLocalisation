"""Compose a single-channel index label map (HxW uint8) from either source.

Two annotation sources, one output format:
  * server GT: a list of per-class **binary PNG masks** (mode L, >0 = that class);
  * new COCO: **polygon** annotations grouped by canonical class.

Both are painted onto a zeros (=background) canvas in ``PAINT_ORDER`` so thin/structural
classes win overlaps deterministically (see config). Output pixels are canonical class ids
0..6. Nothing here touches torch — pure numpy/PIL so it is cheap to unit-test and to run
inside a Dataset ``__getitem__`` (path-level, no pre-merged folders on disk).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import BACKGROUND_ID, CLASS_TO_ID, PAINT_ORDER, canonical


def _order_key(cls: str) -> int:
    """Paint rank for a canonical class; classes not in PAINT_ORDER paint first (lowest)."""
    return PAINT_ORDER.index(cls) if cls in PAINT_ORDER else -1


def compose_from_pngs(class_pngs: list[tuple[str, str]], size: tuple[int, int]) -> np.ndarray:
    """Binary per-class PNGs -> index label map.

    ``class_pngs`` = ``[(canonical_class, png_path), ...]``. ``size`` = (H, W) of the still.
    Missing/unreadable PNGs are skipped (logged by the caller, not here). Painted in
    PAINT_ORDER; a pixel >0 in a class PNG becomes that class id.
    """
    from PIL import Image

    h, w = size
    label = np.zeros((h, w), dtype=np.uint8)
    for cls, png in sorted(class_pngs, key=lambda cp: _order_key(cp[0])):
        cid = CLASS_TO_ID.get(cls)
        if cid is None:
            continue
        m = np.asarray(Image.open(png).convert("L"))
        if m.shape != (h, w):  # masks are authored at still resolution; guard anyway
            m = np.asarray(Image.fromarray(m).resize((w, h), Image.NEAREST))
        label[m > 0] = cid
    return label


def compose_from_coco(anns: list[dict], cats_by_id: dict[int, str], size: tuple[int, int]) -> np.ndarray:
    """COCO polygon annotations for ONE image -> index label map.

    ``anns`` = that image's annotation dicts (each with ``category_id`` + polygon
    ``segmentation``); ``cats_by_id`` maps COCO category_id -> raw name. Names are run
    through ``canonical`` (dropping Building/Target etc.). Polygons are rasterised and
    painted in PAINT_ORDER.
    """
    from PIL import Image, ImageDraw

    h, w = size
    label = np.zeros((h, w), dtype=np.uint8)
    # group polygons by canonical class so we can paint class-by-class in order
    by_class: dict[str, list] = {}
    for a in anns:
        cls = canonical(cats_by_id.get(a["category_id"], ""))
        if cls is None:
            continue
        by_class.setdefault(cls, []).extend(_iter_polys(a.get("segmentation")))
    for cls in sorted(by_class, key=_order_key):
        cid = CLASS_TO_ID[cls]
        canvas = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(canvas)
        for poly in by_class[cls]:
            if len(poly) >= 6:  # >=3 (x,y) points
                d.polygon(poly, fill=1)
        label[np.asarray(canvas) > 0] = cid
    return label


def _iter_polys(seg) -> list[list[float]]:
    """Yield flat [x0,y0,x1,y1,...] polygons from a COCO ``segmentation`` (polygon form).

    RLE segmentations (dict) are not expected in this Roboflow export and are skipped;
    the caller can log if any image yields zero polygons.
    """
    if isinstance(seg, list):
        return [p for p in seg if isinstance(p, list)]
    return []


def colorize(label: np.ndarray) -> np.ndarray:
    """Index map -> RGB using the palette (QC only)."""
    from .config import PALETTE

    out = np.zeros((*label.shape, 3), dtype=np.uint8)
    for cid, rgb in enumerate(PALETTE):
        out[label == cid] = rgb
    return out
