"""Content-adaptive shot-boundary segmentation (PySceneDetect AdaptiveDetector
logic), ported verbatim from ``quality-cls/tools/subvideo_homography.py`` but
operating on DECODED video frames (BGR numpy arrays) instead of JPEGs on disk.

This is the exact backend (``--backend content``) and recipe that produced the
``frames_good`` tree (see the saved ``structure.json``):
    adaptive_thr=3.0, min_content=15.0, content_window=2, content_reduce=4,
    min_seg=2

The cut rule: a boundary before frame ``i`` fires when the HSV content delta
both exceeds an absolute floor (``min_content``) AND is >= ``adaptive_thr`` x the
average delta of the surrounding +/-``window`` frames. The rolling-average ratio
is what makes it robust to steady fast motion (which raises every delta, so the
ratio stays ~1 and no cut fires); only a real scene change spikes above the
local base.
"""
from __future__ import annotations

import cv2
import numpy as np


def to_feature(frame_bgr: np.ndarray, width: int = 480, reduce: int = 4) -> np.ndarray:
    """BGR frame -> HSV float32 feature, at 1/``reduce`` scale then capped to
    ``width`` px wide (mirrors ContentMatcher.features: decode-reduce + resize)."""
    img = frame_bgr
    if reduce and reduce > 1:
        h, w = img.shape[:2]
        img = cv2.resize(img, (max(1, w // reduce), max(1, h // reduce)),
                         interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)


def feature_diff(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Mean HSV dissimilarity (0..255); hue treated as circular. Verbatim diff()."""
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    dh = np.abs(a[..., 0] - b[..., 0]); dh = np.minimum(dh, 180.0 - dh)   # hue is circular
    ds = np.abs(a[..., 1] - b[..., 1])
    dv = np.abs(a[..., 2] - b[..., 2])
    return float((dh.mean() + ds.mean() + dv.mean()) / 3.0)


def content_deltas(feats: list[np.ndarray]) -> list[float]:
    """Per-frame delta vs the previous frame; deltas[0] == 0."""
    deltas = [0.0]
    for i in range(1, len(feats)):
        deltas.append(feature_diff(feats[i - 1], feats[i]))
    return deltas


def cut_indices(deltas: list[float], adaptive_thr: float = 3.0,
                min_content: float = 15.0, window: int = 2) -> set[int]:
    n = len(deltas)
    cuts: set[int] = set()
    for i in range(1, n):
        lo, hi = max(1, i - window), min(n - 1, i + window)
        neigh = [deltas[j] for j in range(lo, hi + 1) if j != i]
        avg = (sum(neigh) / len(neigh)) if neigh else 0.0
        if deltas[i] >= min_content and deltas[i] >= adaptive_thr * (avg + 1e-6):
            cuts.add(i)
    return cuts


def boundary_context(deltas: list[float], i: int, window: int = 2,
                     min_content: float = 15.0, adaptive_thr: float = 3.0) -> dict:
    """Explain why a cut fired before position ``i``: the content delta there,
    the local average of neighbour deltas (+/-``window``), their ratio, and the
    thresholds in force. Makes each boundary reproducible/auditable from JSON."""
    n = len(deltas)
    lo, hi = max(1, i - window), min(n - 1, i + window)
    neigh = [deltas[j] for j in range(lo, hi + 1) if j != i]
    local_avg = (sum(neigh) / len(neigh)) if neigh else 0.0
    d = deltas[i] if 0 <= i < n else 0.0
    return {
        "content_delta": round(d, 3),
        "local_avg": round(local_avg, 3),
        "ratio": round(d / (local_avg + 1e-6), 3),
        "min_content": min_content,
        "adaptive_thr": adaptive_thr,
    }


def segments_from_cuts(n: int, cuts: set[int]) -> list[list[int]]:
    """Positions 0..n-1 grouped into contiguous segments split at ``cuts``."""
    if n <= 0:
        return []
    segs, cur = [], [0]
    for i in range(1, n):
        if i in cuts:
            segs.append(cur); cur = [i]
        else:
            cur.append(i)
    segs.append(cur)
    return segs


def merge_short(segs: list[list[int]], min_seg: int) -> list[list[int]]:
    """Absorb segments shorter than ``min_seg`` into the previous one. Verbatim."""
    if min_seg <= 1 or not segs:
        return segs
    out = [list(segs[0])]
    for s in segs[1:]:
        (out[-1].extend(s) if len(out[-1]) < min_seg else out.append(list(s)))
    if len(out) > 1 and len(out[-1]) < min_seg:
        out[-2].extend(out[-1]); out.pop()
    return out


def segment_frames(feats: list[np.ndarray], adaptive_thr: float = 3.0,
                   min_content: float = 15.0, window: int = 2, min_seg: int = 2):
    """Split a frame sequence into sub-videos.

    Returns ``(segments, deltas, cuts)`` where ``segments`` is a list of
    position-lists (indices into ``feats``), ``deltas`` the per-frame content
    delta, and ``cuts`` the raw boundary set (pre-merge).
    """
    n = len(feats)
    deltas = content_deltas(feats)
    if n <= 1:
        return ([list(range(n))] if n else []), deltas, set()
    cuts = cut_indices(deltas, adaptive_thr, min_content, window)
    segs = merge_short(segments_from_cuts(n, cuts), min_seg)
    return segs, deltas, cuts
