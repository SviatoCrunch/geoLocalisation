"""Concentric physical pyramid — nested CENTRAL crops of a token grid (separate from cell).

Different concept from the cell pyramid: instead of splitting a tile into n×n spatial cells,
each level ``s`` is ONE central region of physical side ``s`` metres (a fraction ``s/tile_size_m``
of the grid), aggregated to a single regional descriptor ``V[s] = (M, 1, d_out)``. Levels are
nested, share the geometric centre, apex is 250 m.

The levels are PARALLEL regions of the SAME DINO token map (not sequential layers). Adjacent
levels are correlated because they reuse nested tokens. NOTE (DINOv2 context leakage): cropping
already-computed tokens is NOT equivalent to a separate DINOv2 pass on the crop — global
self-attention already mixed context across the whole tile.

Reuses the VENDORED Stage-2 building blocks verbatim and SUBCLASSES ``Stage2QueryConditionedModel``
so the query side + the scorer are literally the same code — only the map-side region pooling
(fractional central mask instead of n×n cells) and the level set differ. The vendored cell core
file is NOT edited, so cell parity is untouched.

Fractional mask (exact symmetric, no half-token shift). For a token axis of length ``L`` token ``i``
spans ``[i/L, (i+1)/L]``; level ``s`` (ratio ``r = s/tile_size_m``) spans ``[0.5-r/2, 0.5+r/2]``.
The 1-D weight is the overlap length scaled by ``L``::

    w_i = L * max(0, min((i+1)/L, 0.5+r/2) - max(i/L, 0.5-r/2))     # in [0, 1]

and the 2-D mask is the outer product ``w_y ⊗ w_x`` (float, inner tokens 1, outer 0, boundary
fractional). ``Σ w_i = r·L`` (the nominal continuous side); the centre of mass is exactly the grid
centre for even AND odd ``L`` (fixes the previous ``round``-crop half-token shift). This fractional
spatial weight multiplies each token's contribution BEFORE the VLAD sum.

Level score (documented): 1 region per level ⇒ the per-cell softmax in the reused scorer is over a
single element (identity), τ≡1, so ``S_ℓ = ⟨V[ℓ], q⟩`` and the tile score is ``Σ_ℓ β_ℓ·S_ℓ`` with
``β = softmax(ScaleGate(q))`` — the SAME cross-scale (soft-mixture) formula as cell mode. This is a
soft mixture of scales, NOT a hard selection of one footprint.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vendored.stage2_core import (CellSuperVLAD, PerGroupProjection, Stage2QueryConditionedModel,
                                   apply_assignment_init, assignment_from_centroids)


# ── fractional central-crop token weights ────────────────────────────────────────────
def _axis_weights(L: int, ratio: float) -> np.ndarray:
    """(L,) float weights in [0,1]: overlap of each token [i/L,(i+1)/L] with the central crop
    [0.5-r/2, 0.5+r/2], scaled by L. Symmetric ⇒ centre of mass = (L-1)/2 for even and odd L."""
    L = int(L)
    lo, hi = 0.5 - float(ratio) / 2.0, 0.5 + float(ratio) / 2.0
    i = np.arange(L, dtype=np.float64)
    left, right = i / L, (i + 1.0) / L
    w = (np.minimum(right, hi) - np.maximum(left, lo)).clip(min=0.0) * L
    return np.clip(w, 0.0, 1.0)


def concentric_token_masks(H: int, W: int, sizes_m, tile_size_m: float):
    """Per level: fractional mask (H*W,) float + geometry metadata (bounds/weight_sum/effective
    side/centre error). Same algorithm for even/odd H,W; H and W use the same physical fraction."""
    out = []
    for s in sizes_m:
        ratio = float(s) / float(tile_size_m)
        wy, wx = _axis_weights(H, ratio), _axis_weights(W, ratio)
        mask = np.outer(wy, wx)                                      # (H, W) float in [0,1]
        ys, xs = np.nonzero(wy > 0)[0], np.nonzero(wx > 0)[0]
        sb = (int(ys[0]), int(ys[-1]), int(xs[0]), int(xs[-1])) if ys.size and xs.size else (0, -1, 0, -1)
        comy = float((np.arange(H) * wy).sum() / max(wy.sum(), 1e-12))
        comx = float((np.arange(W) * wx).sum() / max(wx.sum(), 1e-12))
        out.append({
            "size_m": float(s), "ratio": ratio, "support_bounds": sb,
            "weight_sum": float(mask.sum()),
            "effective_side_h": float(wy.sum()), "effective_side_w": float(wx.sum()),
            "center_error_y": abs(comy - (H - 1) / 2.0),
            "center_error_x": abs(comx - (W - 1) / 2.0),
            "weight": mask.reshape(-1).astype(np.float64),
        })
    return out


class ConcentricStage2Model(Stage2QueryConditionedModel):
    """Concentric-pyramid variant. Subclasses the cell core; overrides only the map-side region
    pooling (fractional central mask) + level set + temperature. scale_gate is sized to L levels;
    no per-cell ρ (τ≡1)."""

    def __init__(self, agg: CellSuperVLAD, group_proj: PerGroupProjection, concentric_sizes_m,
                 tile_size_m: float, *, d_out=256, head_hidden=512, dropout=0.1, intra=True,
                 eps=1e-8):
        sizes = tuple(float(s) for s in concentric_sizes_m)
        L = len(sizes)
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

    def _apply(self, fn, *args, **kwargs):
        # moving the module (.to/.cuda/.cpu) invalidates cached masks (device/dtype) — drop them
        out = super()._apply(fn, *args, **kwargs)
        self._mask_cache.clear()
        return out

    def tau(self, n) -> torch.Tensor:
        return torch.ones((), device=self._dev())        # 1 region/level ⇒ softmax is identity

    def masks(self, H: int, W: int, device, dtype=torch.float32) -> torch.Tensor:
        """(L, H*W) float fractional masks for this grid, cached per (H, W, device, dtype)."""
        key = (int(H), int(W), str(torch.device(device)), str(dtype))
        t = self._mask_cache.get(key)
        if t is None:
            specs = concentric_token_masks(H, W, self.concentric_sizes_m, self.tile_size_m)
            arr = np.stack([sp["weight"] for sp in specs], axis=0)       # (L, N) float64
            t = torch.from_numpy(arr).to(device=torch.device(device), dtype=dtype)
            self._mask_cache[key] = t
        return t

    def mask_specs(self, H: int, W: int):
        """Per-level geometry metadata (fractional). For tests / diagnostics."""
        return concentric_token_masks(H, W, self.concentric_sizes_m, self.tile_size_m)

    # ── map side: token grids → concentric region descriptors (supervlad arm) ─────────
    def region_from_tiles(self, tok_grids: torch.Tensor) -> dict:
        """(M,H,W,D) → {level: (M,1,d_reduced)} — EXACT _pyramid_per_group_fast math, but pooled
        with the fractional central-crop weight (1 region) instead of index_add over n² cells."""
        M, H, W, D = tok_grids.shape
        dev, eps = tok_grids.device, self.eps
        flat = F.normalize(tok_grids.reshape(M, H * W, D), dim=-1, eps=eps)
        alpha = self.agg.alpha(flat)                                      # (M, N, K)
        value = self.agg.value(flat)                                     # (M, N, Dv)
        w = self.group_proj.weight
        if w.dtype != value.dtype:
            w = w.to(value.dtype)
        pv = torch.einsum("mnd,kod->mnko", value, w)                     # (M, N, K, dg)
        weighted = (pv * alpha.unsqueeze(-1)).reshape(M, H * W, -1)      # (M, N, K*dg)
        K, dg = alpha.shape[-1], self.group_proj.d_out
        masks = self.masks(H, W, dev, weighted.dtype)                    # (L, N) fractional
        pyr = {}
        for li in self.scales_cells:                                     # level indices
            m = masks[li].view(1, -1, 1)                                 # (1, N, 1) fractional weight
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
