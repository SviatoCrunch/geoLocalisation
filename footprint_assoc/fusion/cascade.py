"""cascade — the recommended default (doc §"Об'єднання оцінок").

global → top candidates; patch → orders them; agreement between the two (independent-ish)
methods drives CONFIDENCE, not the pick. Never forces argmax: weak margin or method
disagreement → SOFT; no support at all → REFUSE. (Geometry veto is a later iteration.)
"""
from __future__ import annotations

import numpy as np

from ..schemas import FrameEstimate, ACCEPT, SOFT, REFUSE
from .base import method_scores, softmax_masked


class CascadeFusion:
    name = "cascade"

    def fuse(self, frame_id, center_lat, center_lon, scales, per_method, cfg) -> FrameEstimate:
        n = len(scales)
        g = method_scores(per_method, "global_vlad", scales)
        p = method_scores(per_method, "patch_overlap", scales)
        has_g, has_p = "global_vlad" in per_method, "patch_overlap" in per_method

        # global forms top-K candidates (else all levels are candidates)
        mask = np.zeros(n, bool)
        if has_g:
            k = min(3, n)
            mask[np.argsort(-g)[:k]] = True
        else:
            mask[:] = True

        primary = p if has_p else g                 # patch orders candidates; else global
        soft = softmax_masked(primary, cfg.tau, mask)

        if not np.isfinite(soft).any() or soft.sum() <= 0 or float(primary[mask].max() if mask.any() else 0) <= 0:
            return self._refuse(frame_id, center_lat, center_lon, scales, per_method, soft,
                                "no supporting evidence (zero primary score)")

        order = np.argsort(-soft)
        best = int(order[0])
        gap = float(soft[order[0]] - soft[order[1]]) if n > 1 else float(soft[order[0]])

        agree = True
        if has_g and has_p:
            agree = abs(int(np.argmax(g)) - int(np.argmax(p))) <= cfg.agreement_levels

        if gap >= cfg.accept_margin and agree:
            status, reason = ACCEPT, "margin+agreement"
        else:
            status = SOFT
            reason = "low margin" if gap < cfg.accept_margin else "method disagreement"

        return FrameEstimate(
            frame_id=frame_id, center_lat=center_lat, center_lon=center_lon, scales=list(scales),
            per_method=per_method, soft={float(scales[i]): float(soft[i]) for i in range(n)},
            best_scale=float(scales[best]), confidence=float(soft[best]),
            status=status, reason=reason)

    def _refuse(self, fid, lat, lon, scales, per_method, soft, reason) -> FrameEstimate:
        n = len(scales)
        return FrameEstimate(fid, lat, lon, list(scales), per_method,
                             {float(scales[i]): float(soft[i]) for i in range(n)},
                             None, 0.0, REFUSE, reason)
