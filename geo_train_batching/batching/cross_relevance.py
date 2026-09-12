"""Cross-relevance masks Y[B,B] for the DSS symmetric multi-positive CE.

Ported verbatim from ``siam_model_stage4_full_gallery.batching.dss``. Cell (a,b) is
positive iff pair b's canonical tile ∈ P_pos(query a); safe ⇒ candidate-only; ambiguous
⇒ ignore. Every anchor must retain ≥1 positive (else the canonical pairing is broken).
"""
from __future__ import annotations

import numpy as np
import torch


def build_cross_relevance(pairs_in_batch, relevance_table):
    """Return (R_pos, R_cand) bool torch tensors ``[B,B]`` (R_cand ⊇ R_pos)."""
    B = len(pairs_in_batch)
    R_pos = np.zeros((B, B), dtype=bool)
    R_cand = np.zeros((B, B), dtype=bool)
    pos_sets = [set(int(x) for x in relevance_table.pos_of(p.query_index)) for p in pairs_in_batch]
    safe_sets = [set(int(x) for x in relevance_table.safe_of(p.query_index)) for p in pairs_in_batch]
    tiles = [int(p.canonical_tile_row) for p in pairs_in_batch]
    for a in range(B):
        for b in range(B):
            t = tiles[b]
            if t in pos_sets[a]:
                R_pos[a, b] = True
                R_cand[a, b] = True
            elif t in safe_sets[a]:
                R_cand[a, b] = True
            # else ambiguous/ignore -> neither
    if not R_pos.any(axis=1).all():
        bad = np.nonzero(~R_pos.any(axis=1))[0].tolist()
        raise ValueError(f"DSS anchors with no positive in the batch (rows {bad}) — check canonical pairs")
    return torch.from_numpy(R_pos), torch.from_numpy(R_cand)


def build_cross_weights(pairs_in_batch, relevance_table, R_pos):
    """Soft-target weight matrix ``W[B,B]`` (float) for the weighted CE: ``W[a,b]`` = the
    positive-selection weight of pair ``b``'s canonical tile for query ``a`` (0 where not a
    positive). Requires a relevance table that carries weights (``has_weights``)."""
    if not getattr(relevance_table, "has_weights", False):
        raise ValueError("relevance_table has no positive weights — use a weighting positive "
                         "selector (e.g. overlap_weighted / tile_iou_1000)")
    rp = R_pos.cpu().numpy() if hasattr(R_pos, "cpu") else np.asarray(R_pos)
    B = len(pairs_in_batch)
    tiles = [int(p.canonical_tile_row) for p in pairs_in_batch]
    W = np.zeros((B, B), dtype=np.float64)
    for a in range(B):
        wa = relevance_table.weight_lookup(pairs_in_batch[a].query_index)
        for b in range(B):
            if rp[a, b]:
                W[a, b] = wa.get(tiles[b], 0.0)
    return torch.from_numpy(W)
