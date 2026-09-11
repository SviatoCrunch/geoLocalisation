"""Classic / residual VLAD aggregation — hard argmin assignment, residual ``Σ(x̄ − c_k)``.

This is the e2c "residual" arm ported VERBATIM (parity-tested against
``train_multicity_e2c.residual_*``). Classic VLAD (Jégou 2010) IS exactly this — per
descriptor, residual to its assigned centroid, summed per cell/cluster, intra- and
L2-normalised — so ``vlad`` and ``residual`` are the SAME strategy (both names registered).

Needs the recast ``centroids`` (K, D); it stores them normalised and moves them to the
grids'/tokens' device on use. Shares the model's pyramid / group_proj / heads.
"""
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

from ..adapters import stage3 as S3


class VladAggregation:
    name = "vlad"

    def __init__(self, centroids: torch.Tensor):
        # stored row-normalised (idempotent if already normalised)
        self._C = S3.normalize_centroids(centroids)

    @classmethod
    def build(cls, *, blob=None, centroids=None, cfg=None) -> "VladAggregation":
        if centroids is None:
            if not blob or "centroids" not in blob:
                raise ValueError("vlad/residual aggregation needs centroids — the recast k*.pt must "
                                 "contain 'centroids' (or pass centroids=...)")
            centroids = blob["centroids"]
        return cls(centroids)

    # ── map side: token grids → residual cell pyramid (pre map_head) ──────────────────
    def _region(self, core, grids, eps=1e-8):
        """(M,H,W,D) → residual pyramid {n:(M,n²,K·d_group)}. One scatter_add per scale
        over (cell*K + cluster). Same aggregation/normalisation as the e2c residual arm."""
        C = self._C.to(device=grids.device, dtype=torch.float32)
        M, H, W, D = grids.shape
        dev = grids.device
        flat = F.normalize(grids.reshape(M, H * W, D).float(), dim=-1, eps=eps)     # (M,N,D)
        labels = torch.cdist(flat, C.unsqueeze(0).expand(M, -1, -1)).argmin(-1)     # (M,N) hard assign
        res_tok = flat - C[labels]                                                 # (M,N,D) residual
        K = C.shape[0]
        gp = core.group_proj
        pyr = {}
        for n in core.scales_cells:
            rid = S3.region_ids_cached(H, W, n, dev)                               # (N,) token -> cell
            idx = (rid.view(1, -1) * K + labels).unsqueeze(-1).expand(-1, -1, D)   # cell*K + cluster
            block = torch.zeros(M, n * n * K, D, device=dev, dtype=flat.dtype)
            block.scatter_add_(1, idx, res_tok)
            block = block.view(M, n * n, K, D)
            if core.intra:
                block = F.normalize(block, dim=3, eps=eps)
            block = F.normalize(gp(block), dim=3, eps=eps).reshape(M, n * n, K * gp.d_out)
            pyr[n] = F.normalize(block, dim=2, eps=eps)
        return pyr

    def _region_concentric(self, core, grids, eps=1e-8):
        """(M,H,W,D) → residual concentric pyramid {ℓ:(M,1,K·d_group)}. Same residual/normalise
        math as ``_region`` with a single spatial region per level, but summed ONLY over the
        central-crop tokens (mask). At the full-tile level (ratio 1, mask all) this reduces to the
        cell n=1 region → parity with ``_region``."""
        C = self._C.to(device=grids.device, dtype=torch.float32)
        M, H, W, D = grids.shape
        dev = grids.device
        flat = F.normalize(grids.reshape(M, H * W, D).float(), dim=-1, eps=eps)      # (M,N,D)
        labels = torch.cdist(flat, C.unsqueeze(0).expand(M, -1, -1)).argmin(-1)      # (M,N)
        res_tok = flat - C[labels]                                                   # (M,N,D)
        K = C.shape[0]
        gp = core.group_proj
        masks = core.masks(H, W, dev, res_tok.dtype)                                 # (L,N) fractional
        idx = labels.unsqueeze(-1).expand(-1, -1, D)                                 # (M,N,D) cluster ids
        pyr = {}
        for li in core.scales_cells:                                                 # level indices
            m = masks[li].view(1, -1, 1)                                             # (1,N,1) fractional
            block = torch.zeros(M, K, D, device=dev, dtype=flat.dtype)
            # assignment (hard) -> residual -> fractional spatial weighting -> per-cluster sum
            block.scatter_add_(1, idx, res_tok * m)
            block = block.view(M, 1, K, D)
            if core.intra:
                block = F.normalize(block, dim=3, eps=eps)
            block = F.normalize(gp(block), dim=3, eps=eps).reshape(M, 1, K * gp.d_out)
            pyr[li] = F.normalize(block, dim=2, eps=eps)
        return pyr

    def build_V(self, core, grids):
        if getattr(core, "pyramid_mode", "cell") == "concentric":
            return core.transform_map(self._region_concentric(core, grids, eps=core.eps))
        return core.transform_map(self._region(core, grids, eps=core.eps))

    # ── query side: global residual VLAD over UAV tokens ──────────────────────────────
    def encode_query(self, core, tokens):
        eps = core.eps
        C = self._C.to(device=tokens.device, dtype=torch.float32)
        x = F.normalize(tokens.float(), dim=-1, eps=eps)                            # (N,D)
        labels = torch.cdist(x, C).argmin(-1)                                       # (N,)
        K, D = C.shape
        U = torch.zeros(K, D, device=x.device, dtype=x.dtype).index_add_(0, labels, x)
        counts = torch.zeros(K, device=x.device).index_add_(
            0, labels, torch.ones(x.shape[0], device=x.device))
        U = U - counts.unsqueeze(1) * C                                             # residual −n_k·c_k
        if core.intra:
            U = F.normalize(U, dim=1, eps=eps)
        U = F.normalize(core.group_proj(U), dim=1, eps=eps)
        q = F.normalize(U.reshape(-1), dim=0, eps=eps)
        return core.drone_head(core.reduce(q))

    def resolved_config(self) -> Mapping[str, object]:
        return {"agg": "vlad", "classic_vlad": True, "assign": "argmin(hard)",
                "value": "residual(xbar - c_k)", "n_centroids": int(self._C.shape[0])}
