"""
COCO ``result.json`` -> per-class binary masks for UAV GT frames.

Reads a COCO-format annotation file (``images`` + ``annotations`` +
``categories``) and rasterises, for every (image, class) pair, a single **binary**
PNG mask (foreground = 255, background = 0 — a "bitwise" mask). Masks land next to
the GT frames, by convention under a ``GT_flat_mask`` directory, so they can be
pushed to S3 with the same :func:`s3_gt_sync.push` as the frames themselves.

Naming mirrors the frame names so a mask stays paired with its frame::

    frame :  N_lat_lon.jpg          (COCO images[i].file_name)
    mask  :  N_lat_lon__<class>.png (one per class present in that image)

Segmentation support:
  * polygon segmentation (COCO default)         -> rasterised with PIL
  * uncompressed RLE ({"counts": [...] })        -> decoded natively
  * compressed RLE  ({"counts": "..."} bytes)    -> pycocotools if installed
  * no segmentation                              -> the bbox rectangle is filled
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

log = logging.getLogger(__name__)

# Default sub-directory the masks are written to (and which pull() skips on S3).
MASK_SUBDIR = "GT_flat_mask"

_SANITIZE = re.compile(r"[^\w.\-]+")
# A lat/lon coordinate pair inside a file name, e.g. 48.597..._37.590... (also
# tolerates the comma screenshot variant lat,lon).
_COORD = re.compile(r"(-?\d+\.\d+)[_,](-?\d+\.\d+)")


def _safe_class(name: str) -> str:
    """Make a category name safe to embed in a filename (no path separators)."""
    return _SANITIZE.sub("_", str(name)).strip("_") or "class"


def _frame_stem(file_name: str) -> str:
    """Derive the pairing stem from a COCO ``file_name``.

    Label-Studio exports carry a full ``..\\..\\label-studio\\...`` Windows path
    plus a hash prefix (``0d0aeb21-102_...``); ``PurePosixPath`` won't split on
    backslashes, so we basename across BOTH separators. When a ``lat_lon``
    coordinate pair is present we keep only that (the stable GT key), dropping the
    upload id / hash — e.g. ``0d0aeb21-102_48.59_37.59`` -> ``48.59_37.59``.
    Falls back to the sanitised basename stem when no coordinates are found.
    """
    base = re.split(r"[\\/]", str(file_name))[-1]
    stem = base.rsplit(".", 1)[0] if "." in base else base
    m = _COORD.search(stem)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return _SANITIZE.sub("_", stem).strip("_") or "frame"


# ── COCO decoding ─────────────────────────────────────────────────────────────

def _load_coco(coco_json: str) -> dict:
    with open(coco_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or "annotations" not in data:
        raise ValueError(
            f"{coco_json!r} is not a COCO dict (need images/annotations/categories)."
        )
    return data


def _rle_decode_uncompressed(counts: list[int], h: int, w: int) -> np.ndarray:
    """Decode an uncompressed COCO RLE (column-major runs, starting with 0)."""
    flat = np.zeros(h * w, dtype=bool)
    idx = 0
    val = False
    for c in counts:
        flat[idx:idx + c] = val
        idx += c
        val = not val
    return flat.reshape((h, w), order="F")


def _decode_rle(seg: dict, h: int, w: int) -> np.ndarray:
    counts = seg.get("counts")
    size = seg.get("size", [h, w])
    hh, ww = int(size[0]), int(size[1])
    if isinstance(counts, list):
        return _rle_decode_uncompressed(counts, hh, ww)
    # Compressed RLE (str/bytes) needs pycocotools.
    try:
        from pycocotools import mask as mask_utils
    except ImportError as e:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "Compressed-RLE segmentation requires pycocotools "
            "(`uv add pycocotools`); only polygon and uncompressed RLE work "
            "without it."
        ) from e
    rle = dict(seg)
    if isinstance(counts, str):
        rle["counts"] = counts.encode("ascii")
    return mask_utils.decode(rle).astype(bool)


def _decode_polygons(polys: list, h: int, w: int) -> np.ndarray:
    """Rasterise one or more COCO polygons into a boolean mask."""
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    # A single polygon may be given as a flat number list; wrap it.
    if polys and isinstance(polys[0], (int, float)):
        polys = [polys]
    for poly in polys:
        if len(poly) < 6:  # need >= 3 points
            continue
        pts = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly) - 1, 2)]
        draw.polygon(pts, fill=1)
    return np.asarray(img, dtype=bool)


def _decode_bbox(bbox: list, h: int, w: int) -> np.ndarray:
    """Fill the COCO bbox rectangle [x, y, bw, bh] as a fallback mask."""
    x, y, bw, bh = bbox
    x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
    x1, y1 = min(w, int(round(x + bw))), min(h, int(round(y + bh)))
    m = np.zeros((h, w), dtype=bool)
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = True
    return m


def _ann_mask(ann: dict, h: int, w: int) -> np.ndarray | None:
    seg = ann.get("segmentation")
    if isinstance(seg, dict):
        return _decode_rle(seg, h, w)
    if isinstance(seg, list) and seg:
        return _decode_polygons(seg, h, w)
    bbox = ann.get("bbox")
    if bbox and len(bbox) == 4:
        return _decode_bbox(bbox, h, w)
    return None


# ── mask generation ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MaskTarget:
    """One planned mask: union of all ``category_id`` annotations of ``image_id``,
    written locally as ``dst_name``."""
    image_id: int
    category_id: int
    dst_name: str
    height: int
    width: int


def masks_from_coco(coco_json: str, dest_dir: str, *,
                    skip_existing: bool = True, dry_run: bool = False) -> dict:
    """Rasterise per-(image, class) binary masks from a COCO ``result.json``.

    One PNG per image-and-class (all annotations of that class in that image are
    unioned into a single bitwise mask). Existing same-name files are left in
    place with ``skip_existing``. Returns a stats dict:
    written / skipped / images / annotations / empty / classes.
    """
    data = _load_coco(coco_json)
    images = {img["id"]: img for img in data.get("images", [])}
    cats = {c["id"]: c["name"] for c in data.get("categories", [])}
    anns = data.get("annotations", [])

    # Group annotations by (image_id, category_id) so one class -> one mask.
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for a in anns:
        grouped[(a["image_id"], a["category_id"])].append(a)

    if not dry_run:
        os.makedirs(dest_dir, exist_ok=True)

    written = skipped = empty = 0
    used_classes: set[str] = set()
    for (image_id, category_id), group in sorted(grouped.items()):
        img = images.get(image_id)
        if img is None:
            log.warning("mask skip  image_id=%s not in images[]", image_id)
            continue
        h, w = int(img["height"]), int(img["width"])
        cls = _safe_class(cats.get(category_id, f"cat{category_id}"))
        used_classes.add(cls)
        stem = _frame_stem(img["file_name"])
        dst_name = f"{stem}__{cls}.png"
        local = os.path.join(dest_dir, dst_name)

        if skip_existing and os.path.exists(local):
            skipped += 1
            log.debug("mask skip (exists)  %s", dst_name)
            continue

        # Union every annotation of this class in this image.
        mask = np.zeros((h, w), dtype=bool)
        for a in group:
            m = _ann_mask(a, h, w)
            if m is not None:
                mask |= m
        if not mask.any():
            empty += 1
            log.warning("mask empty  %s (no decodable segmentation/bbox)", dst_name)
            continue

        if dry_run:
            log.info("mask would write  %s (%dx%d)", dst_name, w, h)
            written += 1
            continue

        Image.fromarray((mask * 255).astype("uint8"), mode="L").save(local)
        log.info("mask write  %s (%dx%d)", dst_name, w, h)
        written += 1

    stats = {
        "written": written,
        "skipped": skipped,
        "images": len(images),
        "annotations": len(anns),
        "empty": empty,
        "classes": sorted(used_classes),
    }
    log.info("masks done  written=%d skipped=%d empty=%d images=%d anns=%d",
             written, skipped, empty, len(images), len(anns))
    return stats
