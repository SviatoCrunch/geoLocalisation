"""Canonical query→tile pairs (one per train query).

Ported verbatim from ``siam_model_stage4_full_gallery.batching.dss``. The canonical
tile for a query is the positive whose centre is closest to the query GT. Queries with
no positive are skipped (a preflight/geometry error surfaced upstream).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CanonicalPair:
    pair_index: int
    query_index: int             # row into the relevance table / query list
    query_id: str
    canonical_tile_row: int      # gallery row of the canonical positive tile


def build_pair_pool(relevance_table, query_gt_xy, tile_centers) -> list:
    """One canonical pair per query with ≥1 positive (canonical = argmin centre distance)."""
    gt = np.asarray(query_gt_xy, np.float64).reshape(-1, 2)
    centers = np.asarray(tile_centers, np.float64).reshape(-1, 2)
    pairs = []
    for i in range(relevance_table.n_queries):
        pos = np.asarray(relevance_table.pos_of(i))
        if pos.size == 0:
            continue
        d = np.linalg.norm(centers[pos] - gt[i], axis=1)
        canonical = int(pos[int(np.argmin(d))])
        pairs.append(CanonicalPair(pair_index=len(pairs), query_index=i,
                                   query_id=str(relevance_table.query_ids[i]),
                                   canonical_tile_row=canonical))
    return pairs


def pair_pool_to_json(pairs) -> dict:
    return {"n_pairs": len(pairs),
            "pairs": [{"pair_index": p.pair_index, "query_id": p.query_id,
                       "query_index": p.query_index, "canonical_tile_row": p.canonical_tile_row}
                      for p in pairs]}
