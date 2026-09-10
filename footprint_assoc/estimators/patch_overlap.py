"""patch_overlap — Spatch(s) = trimmed_mean(match_sims) · √(CQ·CG).

Mutual-nearest-neighbour patch correspondences between the UAV and satellite token grids
(optionally gated by a ratio test), weighted by how much of the QUERY is supported (CQ)
and how much of the satellite crop is covered (CG). Robust to extra context; risk =
repetitive textures. Pure numpy — operates on token grids only.
"""
from __future__ import annotations

import numpy as np

from ..schemas import LevelScore, LevelFeatures, QueryFeatures


def _flat_norm(grid: np.ndarray) -> np.ndarray:
    x = np.asarray(grid, np.float64).reshape(-1, grid.shape[-1])
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def _trimmed_mean(vals: np.ndarray, trim: float) -> float:
    if vals.size == 0:
        return 0.0
    v = np.sort(vals)
    k = int(len(v) * trim)
    v = v[k:len(v) - k] if len(v) - 2 * k > 0 else v
    return float(v.mean())


class PatchOverlapEstimator:
    name = "patch_overlap"

    def score_level(self, query: QueryFeatures, level: LevelFeatures, cfg) -> LevelScore:
        Q = _flat_norm(query.patch_grid)                    # (Nq, D)
        S = _flat_norm(level.patch_grid)                    # (Ns, D)
        if Q.shape[0] == 0 or S.shape[0] == 0:
            return LevelScore(level.scale_m, 0.0, {"n_matches": 0})
        sim = Q @ S.T                                       # (Nq, Ns)
        q2s = sim.argmax(axis=1)                            # best sat for each query
        s2q = sim.argmax(axis=0)                            # best query for each sat
        # mutual nearest neighbours
        qi = np.arange(Q.shape[0])
        mutual = s2q[q2s] == qi
        # optional ratio test on the query side (top1 sufficiently above top2)
        if cfg.patch_ratio_test < 1.0 and S.shape[0] >= 2:
            part = np.partition(sim, -2, axis=1)
            top1, top2 = part[:, -1], part[:, -2]
            ok = top2 <= cfg.patch_ratio_test * np.clip(top1, 1e-9, None)
            mutual = mutual & ok
        m_idx = np.nonzero(mutual)[0]
        if m_idx.size == 0:
            return LevelScore(level.scale_m, 0.0, {"n_matches": 0})
        match_sims = sim[m_idx, q2s[m_idx]]
        cq = m_idx.size / Q.shape[0]                        # fraction of query supported
        cg = np.unique(q2s[m_idx]).size / S.shape[0]        # fraction of sat covered
        score = _trimmed_mean(match_sims, cfg.patch_trim) * float(np.sqrt(cq * cg))
        return LevelScore(level.scale_m, float(score),
                          {"n_matches": int(m_idx.size), "cq": float(cq), "cg": float(cg),
                           "mean_sim": float(match_sims.mean())})
