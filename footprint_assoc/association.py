"""Footprint → gallery-tile association (overlap-aware positives).

Turns a :class:`FrameEstimate` into per-tile CQ/CT/IoU + a pair status/weight. Uses the
EXPECTED overlap over levels (Σ_s p(s)·metric) so a soft/uncertain estimate yields graded
weights instead of a forced hard positive (doc §"Варіант із невизначеним масштабом"). A
REFUSED frame yields only UNDETERMINED/NEGATIVE pairs — never hard labels.
"""
from __future__ import annotations

from typing import Sequence

from .geometry import merc, square_bounds, overlap_metrics
from .schemas import PairAssoc, FrameEstimate, STRONG, PARTIAL, UNDETERMINED, NEGATIVE, REFUSE


def associate(estimate: FrameEstimate, tiles: Sequence, cfg) -> list:
    """``tiles`` = iterable of ``(tile_id, center_lat, center_lon, size_m)``. Returns PairAssoc list."""
    cx0, cy0 = merc(estimate.center_lat, estimate.center_lon)
    scales = estimate.scales
    soft = [estimate.soft.get(float(s), 0.0) for s in scales]
    ssum = sum(soft) or 1.0
    soft = [w / ssum for w in soft]
    fp = [square_bounds(cx0, cy0, s) for s in scales]      # concentric footprint per level

    out = []
    for tile_id, tlat, tlon, tsize in tiles:
        tx, ty = merc(tlat, tlon)
        tb = square_bounds(tx, ty, tsize)
        e_cq = e_ct = e_iou = 0.0
        for i, w in enumerate(soft):
            if w <= 0:
                continue
            cq, ct, iou = overlap_metrics(fp[i], tb)
            e_cq += w * cq; e_ct += w * ct; e_iou += w * iou
        out.append(_status(tile_id, e_cq, e_ct, e_iou, estimate.status, cfg))
    return out


def _status(tile_id, cq, ct, iou, frame_status, cfg) -> PairAssoc:
    if frame_status == REFUSE:
        st = UNDETERMINED if iou > 0 else NEGATIVE
        return PairAssoc(tile_id, cq, ct, iou, weight=0.0, status=st)
    if cq >= cfg.strong_cq and ct >= cfg.strong_ct:
        st, weight = STRONG, max(cq, iou)
    elif iou >= cfg.partial_min_iou:
        st, weight = PARTIAL, iou
    else:
        st, weight = NEGATIVE, 0.0
    return PairAssoc(tile_id, cq, ct, iou, weight=float(weight), status=st)
