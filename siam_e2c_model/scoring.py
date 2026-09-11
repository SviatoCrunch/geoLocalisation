"""Detailed query-conditioned scoring — the SAME formula as the vendored
``score_queries_against_tiles``, but also returning the per-level contributions.

Works for BOTH cores (cell + concentric) because it only uses the shared interface
``core.scale_gate``, ``core.tau(n)`` and ``core.scales_cells`` + the map descriptor dict ``V``.
The returned ``tile_scores`` equal ``core.score_queries_against_tiles(q, V)`` exactly (asserted
in tests) — this module does not invent a new formula, it exposes the intermediate level scores.

Final tile score (both modes):  ``score(q, tile) = Σ_ℓ β_ℓ(q) · S_ℓ(q, tile)``
  β = softmax(ScaleGate(q))                                   (query-conditioned scale weights)
  S_ℓ = ⟨ L2( Σ_cells softmax(⟨V[ℓ],q⟩/τ_ℓ) · V[ℓ] ) , q ⟩   (per-level, over that level's cells)
Cell mode: level ℓ = an n×n grid (n² cells, τ_ℓ trainable for n>1). Concentric mode: level ℓ = one
central crop (1 cell ⇒ softmax is identity, τ≡1), so S_ℓ = ⟨V[ℓ], q⟩.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def score_tile_and_levels(core, q: torch.Tensor, V: dict, *, tile_chunk: int | None = None):
    """Return (tile_scores (B,M), level_scores (B,M,L), beta (B,L)). Grad flows through V/q."""
    scales = list(core.scales_cells)
    M = V[scales[0]].shape[0]
    beta = core.scale_gate(q)                                   # (B, L)
    cs = int(tile_chunk) if tile_chunk else M
    tile_cols, level_cols = [], []
    for s0 in range(0, M, cs):
        S_cols = []
        for n in scales:
            Vn = V[n][s0:s0 + cs]                               # (m, r, d)
            e = torch.einsum("mrd,bd->bmr", Vn, q)             # (B, m, r)
            A = F.softmax(e / core.tau(n), dim=2)              # softmax over that level's cells
            m = F.normalize((A.unsqueeze(-1) * Vn.unsqueeze(0)).sum(2), dim=2)   # (B, m, d)
            S_cols.append(torch.einsum("bmd,bd->bm", m, q))   # (B, m)
        S_stack = torch.stack(S_cols, dim=2)                   # (B, m, L)
        tile_cols.append(torch.einsum("bs,bms->bm", beta, S_stack))
        level_cols.append(S_stack)
    return torch.cat(tile_cols, dim=1), torch.cat(level_cols, dim=1), beta


def best_levels(level_scores: torch.Tensor, beta: torch.Tensor, level_values: torch.Tensor) -> dict:
    """Three DISTINCT 'best level' notions (they can disagree — see the docstring below).

    * similarity  — per (query,tile): argmax of the raw per-level similarity S (which crop of THIS
                    tile is most similar to the query).
    * gate        — per query only: argmax of β=ScaleGate(q) (the scale the gate prefers from the
                    UAV query alone, independent of any tile).
    * contribution— per (query,tile): argmax of β_ℓ·S_ℓ (the level that contributes most to the
                    final tile score = the mixture actually used).
    ``level_scores`` (B,M,L), ``beta`` (B,L), ``level_values`` (L,)."""
    contributions = level_scores * beta[:, None, :]           # (B,M,L)
    sim_idx = level_scores.argmax(dim=2)                       # (B,M)
    con_idx = contributions.argmax(dim=2)                      # (B,M)
    gate_idx = beta.argmax(dim=1)                              # (B,)  query-only
    return {
        "level_contributions": contributions,
        "similarity_best_level_index": sim_idx,
        "similarity_best_level_value": level_values[sim_idx],
        "gate_best_level_index": gate_idx,
        "gate_best_level_value": level_values[gate_idx],
        "contribution_best_level_index": con_idx,
        "contribution_best_level_value": level_values[con_idx],
    }


def score_with_details(core, q: torch.Tensor, V: dict, level_values, pyramid_mode: str, *,
                       tile_chunk: int | None = None) -> dict:
    """Diagnostic bundle. ``level_values`` (L,) = cell counts (cell) or physical metres (concentric).

    The final tile score is a SOFT MIXTURE of scales ``Σ_ℓ β_ℓ·S_ℓ``, not a hard selection of one
    footprint — hence THREE separate 'best level' notions are returned (similarity / gate /
    contribution) rather than a single ambiguous one. Invariant (asserted in tests):
    ``level_contributions.sum(dim=2) == tile_scores == core.score_queries_against_tiles(q, V)``.

    NOTE: ``level_probabilities`` = softmax over raw per-level similarities — a *normalized level
    score*, NOT a calibrated footprint probability (no scale supervision). ``*_best_level_value`` are
    pseudo-footprint estimates, not metric-verified footprints. ``best_level_{index,value}`` is a
    documented BACKWARD-COMPAT ALIAS of ``similarity_best_level_*``.
    """
    tile_scores, level_scores, beta = score_tile_and_levels(core, q, V, tile_chunk=tile_chunk)
    B, M, L = level_scores.shape
    lv = torch.as_tensor(level_values, dtype=level_scores.dtype, device=level_scores.device)  # (L,)

    bl = best_levels(level_scores, beta, lv)
    level_probabilities = F.softmax(level_scores, dim=2)       # normalized level score (NOT footprint prob)
    if L >= 2:
        top2 = level_scores.topk(2, dim=2).values
        confidence_margin = top2[..., 0] - top2[..., 1]
    else:
        confidence_margin = torch.zeros(B, M, dtype=level_scores.dtype, device=level_scores.device)
    entropy = -(level_probabilities.clamp_min(1e-9).log() * level_probabilities).sum(2)  # (B, M)

    out = {
        "tile_scores": tile_scores,
        "level_scores": level_scores,
        "beta": beta,
        "level_values": lv,
        "pyramid_mode": pyramid_mode,
        "level_probabilities": level_probabilities,
        "confidence_margin": confidence_margin,
        "entropy": entropy,
        **bl,
    }
    # documented backward-compat alias == similarity_best_level_*
    out["best_level_index"] = bl["similarity_best_level_index"]
    out["best_level_value"] = bl["similarity_best_level_value"]
    return out
