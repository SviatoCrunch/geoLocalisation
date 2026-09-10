"""rank_vote — scale-invariant per-level voting (experiment, doc's alternative).

Reciprocal-Rank Fusion across methods: fused(level) = Σ_m 1/(k0 + rank_m(level)). Avoids
the "different score scales" problem (ranks only). Provided for A/B against ``cascade`` —
NOT recommended as the primary selector (the DINOv2-based voters are correlated). Geometry,
when added, remains a veto rather than an equal vote.
"""
from __future__ import annotations

import numpy as np

from ..schemas import FrameEstimate, ACCEPT, SOFT, REFUSE
from .base import method_scores, softmax_masked

_K0 = 60.0  # standard RRF constant


class RankVoteFusion:
    name = "rank_vote"

    def fuse(self, frame_id, center_lat, center_lon, scales, per_method, cfg) -> FrameEstimate:
        n = len(scales)
        names = [m for m in ("global_vlad", "patch_overlap") if m in per_method]
        fused = np.zeros(n, np.float64)
        any_support = False
        for m in names:
            s = method_scores(per_method, m, scales)
            if s.max() <= 0:
                continue
            any_support = True
            ranks = (-s).argsort().argsort()          # 0 = best
            fused += 1.0 / (_K0 + ranks + 1.0)

        mask = np.ones(n, bool)
        soft = softmax_masked(fused, cfg.tau, mask)
        if not any_support or soft.sum() <= 0:
            return FrameEstimate(frame_id, center_lat, center_lon, list(scales), per_method,
                                 {float(scales[i]): float(soft[i]) for i in range(n)},
                                 None, 0.0, REFUSE, "no supporting evidence")

        order = np.argsort(-soft)
        best = int(order[0])
        gap = float(soft[order[0]] - soft[order[1]]) if n > 1 else float(soft[order[0]])
        status = ACCEPT if gap >= cfg.accept_margin else SOFT
        return FrameEstimate(
            frame_id, center_lat, center_lon, list(scales), per_method,
            {float(scales[i]): float(soft[i]) for i in range(n)},
            float(scales[best]), float(soft[best]), status,
            "rrf margin" if status == ACCEPT else "low margin")
