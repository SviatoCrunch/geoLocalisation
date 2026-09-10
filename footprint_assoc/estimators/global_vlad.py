"""global_vlad — Sglobal(s) = cosine(D(Q), D(G(c,s))) (AnyLoc-style whole-image score).

Cheap baseline; keep the WHOLE score(s) curve, not just argmax (a broad/multimodal curve
means the scale is undetermined). Known risk: cosine bias between scales.
"""
from __future__ import annotations

import numpy as np

from ..schemas import LevelScore, LevelFeatures, QueryFeatures


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b / (na * nb))


class GlobalVladEstimator:
    name = "global_vlad"

    def score_level(self, query: QueryFeatures, level: LevelFeatures, cfg) -> LevelScore:
        return LevelScore(scale_m=level.scale_m,
                          score=_cos(query.global_vec, level.global_vec),
                          aux={"metric": "cosine"})
