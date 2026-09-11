"""Concentric physical pyramid — nested CENTRAL crops of a token grid (separate from cell).

Different concept from the cell pyramid: instead of splitting a tile into n×n spatial cells,
each level ``s`` is ONE central region of physical side ``s`` metres (a fraction ``s/tile_size_m``
of the grid), aggregated to a single regional descriptor ``V[s] = (M, 1, d_out)``. Levels are
nested, share the geometric centre, apex is 250 m.

Reuses the VENDORED Stage-2 building blocks verbatim (``CellSuperVLAD``, ``PerGroupProjection``,
``ProjectionHead``, ``ScaleGate``, ``super_global``, assignment loaders) and SUBCLASSES
``Stage2QueryConditionedModel`` so the query side + the scorer are literally the same code — only
the map-side region pooling (mask instead of n×n cells) and the level set differ. The vendored
cell core file is NOT edited, so cell parity is untouched.

Rounding rule (deterministic, symmetric): for physical level ``s`` and a token axis of length
``L``, the central side is ``round((s/tile_size_m) * L)`` tokens, taken as the central block
``start = (L - side)//2``. For a 60×60 grid and levels (1000,840,710,600,500,420,350,300,250) m
this gives sides (60,50,43,36,30,25,21,18,15). This ``round(ratio·L)`` rule is chosen to match the
intended physical side; it can differ by one token from a naive token-centre threshold (e.g. 710 m
→ 43 here vs 42 by the strict centre test) — see ``tests/test_concentric_masks.py``.

Level score (documented): 1 region per level ⇒ the per-cell softmax in the reused scorer is over a
single element (identity), τ≡1, so ``S_ℓ = ⟨V[ℓ], q⟩`` and the tile score is ``Σ_ℓ β_ℓ·S_ℓ`` with
``β = softmax(ScaleGate(q))`` — the SAME cross-scale formula as cell mode, with spatial pooling
(masked mean inside the crop) separated from cross-scale pooling (β). A concentric region is NOT
treated as n² cells.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vendored.stage2_core import (CellSuperVLAD, PerGroupProjection, Stage2QueryConditionedModel,
                                   apply_assignment_init, assignment_from_centroids)


# ── central-crop token masks ────────────────────────────────────────────────────────
def _central_bounds(axis_len: int, ratio: float):
    """(start, stop, side) of the central block of ``round(ratio*axis_len)`` tokens."""
    side = int(round(float(ratio) * int(axis_len)))
    side = max(1, min(int(axis_len), side))
    start = (int(axis_len) - side) // 2
    return start, start + side, side


def concentric_token_masks(H: int, W: int, sizes_m, tile_size_m: float):
    """Per level: dict with mask (H*W,) bool, token bounds, side_h/side_w, count, ratio, size_m.

    Levels are taken in the given order (caller passes canonical descending). Same deterministic
    rounding for even/odd H/W; centre never drifts by a whole token between levels."""
    out = []
    for s in sizes_m:
        ratio = float(s) / float(tile_size_m)
        h0, h1, side_h = _central_bounds(H, ratio)
        w0, w1, side_w = _central_bounds(W, ratio)
        mask = np.zeros((int(H), int(W)), dtype=bool)
        mask[h0:h1, w0:w1] = True
        out.append({"size_m": float(s), "ratio": ratio, "bounds": (h0, h1, w0, w1),
                    "side_h": int(side_h), "side_w": int(side_w),
                    "count": int(side_h * side_w), "mask": mask.reshape(-1)})
    return out


class ConcentricStage2Model(Stage2QueryConditionedModel):
    """Concentric-pyramid variant. Subclasses the cell core; overrides only the map-side region
    pooling + level set + temperature. scale_gate is sized to L levels; no per-cell ρ (τ≡1)."""

    def __init__(self, agg: CellSuperVLAD, group_proj: PerGroupProjection, concentric_sizes_m,
                 tile_size_m: float, *, d_out=256, head_hidden=512, dropout=0.1, intra=True,
                 eps=1e-8):
        sizes = tuple(float(s) for s in concentric_sizes_m)
        L = len(sizes)
        # build the shared agg/group_proj/heads + a scale_gate of width L via the base __init__.
        # scales_cells := level indices [0..L-1] (dict keys for V + scorer iteration).
        super().__init__(agg, group_proj, scales_cells=tuple(range(L)), d_out=d_out,
                         head_hidden=head_hidden, dropout=dropout, tau_min=1.0, tau_init=1.0,
                         intra=intra, eps=eps)
        self.rho = nn.ParameterDict()                    # concentric: no per-cell temperature
        self.pyramid_mode = "concentric"
        self.scoring_mode = "concentric_query_conditioned"
        self.concentric_sizes_m = sizes
        self.tile_size_m = float(tile_size_m)
        self.level_values = sizes                         # physical metres per level
        self._mask_cache: dict = {}

    # 1 region per level ⇒ the per-cell softmax is over one element; temperature is irrelevant.
    def tau(self, n) -> torch.Tensor:
        return torch.ones((), device=self._dev())

    def masks(self, H: int, W: int, device) -> torch.Tensor:
        """(L, H*W) bool central-crop masks for this grid size, cached per (H, W, device)."""
        key = (int(H), int(W), str(torch.device(device)))
        t = self._mask_cache.get(key)
        if t is None:
            specs = concentric_token_masks(H, W, self.concentric_sizes_m, self.tile_size_m)
            arr = np.stack([sp["mask"] for sp in specs], axis=0)          # (L, N) bool
            t = torch.from_numpy(arr).to(torch.device(device))
            self._mask_cache[key] = t
        return t

    def mask_specs(self, H: int, W: int):
        """Per-level geometry metadata (bounds/count/ratio) — for tests / diagnostics."""
        return concentric_token_masks(H, W, self.concentric_sizes_m, self.tile_size_m)

    # ── map side: token grids → concentric region descriptors (supervlad arm) ─────────
    def region_from_tiles(self, tok_grids: torch.Tensor) -> dict:
        """(M,H,W,D) → {level: (M,1,d_reduced)} — EXACT _pyramid_per_group_fast math, but pooled
        over the central-crop mask (1 region) instead of index_add over n² cells."""
        M, H, W, D = tok_grids.shape
        dev = tok_grids.device
        eps = self.eps
        flat = F.normalize(tok_grids.reshape(M, H * W, D), dim=-1, eps=eps)
        alpha = self.agg.alpha(flat)                                      # (M, N, K)
        value = self.agg.value(flat)                                     # (M, N, Dv)
        w = self.group_proj.weight
        if w.dtype != value.dtype:
            w = w.to(value.dtype)
        pv = torch.einsum("mnd,kod->mnko", value, w)                     # (M, N, K, dg)
        weighted = (pv * alpha.unsqueeze(-1)).reshape(M, H * W, -1)      # (M, N, K*dg)
        K, dg = alpha.shape[-1], self.group_proj.d_out
        masks = self.masks(H, W, dev).to(weighted.dtype)                 # (L, N)
        pyr = {}
        for li in self.scales_cells:                                     # level indices
            m = masks[li].view(1, -1, 1)                                 # (1, N, 1)
            pooled = (weighted * m).sum(1, keepdim=True)                 # (M, 1, K*dg)
            pooled = F.normalize(pooled.reshape(M, 1, K, dg), dim=3, eps=eps)
            pyr[li] = F.normalize(pooled.reshape(M, 1, K * dg), dim=2, eps=eps)
        return pyr


def build_concentric_model(*, d_token=1536, n_groups=32, n_ghost=0, group_projection_dim=32,
                           group_proj_init="orthogonal",
                           concentric_sizes_m=(1000.0, 840.0, 710.0, 600.0, 500.0, 420.0,
                                               350.0, 300.0, 250.0),
                           tile_size_m=1000.0, d_out=256, head_hidden=512, dropout=0.1,
                           intra=True, assign_weight: torch.Tensor | None = None,
                           centroids: torch.Tensor | None = None,
                           calib_tokens: torch.Tensor | None = None,
                           freeze_assignment=True) -> ConcentricStage2Model:
    """Build the concentric model — same agg/assignment construction as the cell core, only the
    pyramid differs. ``assign_weight`` (K,D) frozen SuperVLAD dictionary, or centroids+calib."""
    agg = CellSuperVLAD(d_token=d_token, n_groups=n_groups, value_proj="identity",
                        phi_proj="identity", n_ghost=n_ghost, assign_bias=False)
    if assign_weight is not None:
        apply_assignment_init(agg, assign_weight, {"method": "file"})
    elif centroids is not None and calib_tokens is not None:
        fit = assignment_from_centroids(centroids, calib_tokens)
        apply_assignment_init(agg, fit["assign_weight"], {"method": "centroids", "alpha": fit["alpha"]})
    if freeze_assignment:
        agg.set_assign_frozen(True)
    red = PerGroupProjection(n_groups, agg.d_value, int(group_projection_dim), group_proj_init)
    return ConcentricStage2Model(agg, red, concentric_sizes_m, tile_size_m, d_out=d_out,
                                 head_hidden=head_hidden, dropout=dropout, intra=intra)
