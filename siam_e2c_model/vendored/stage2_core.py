"""VENDORED Stage-2 model core — the query-conditioned late-interaction scorer.

This is a byte-faithful copy of the SUCCESSFUL Stage-2 model+scoring path (single source of
truth: ``siam_model_stage2/supervlad_core.py`` + ``siam_model_stage2/train.py``), vendored into
Stage-3 so the ``stage2_query_conditioned`` scoring mode reproduces Stage-2 EXACTLY while running
on Stage-3 geometry / gallery-aware split / full-gallery mining / evaluation.

Contract (reconstructed from Stage-2 code, verified by ``tests/test_stage2_parity.py``):

    UAV tokens ──► super_global (shared CellSuperVLAD + shared PerGroupProjection) ──► reduce
                 ──► drone_head ──► q                                    [d_out], L2

    map tokens ──► build_super_pyramid_batched (shared agg + reducer)    {n:(M,n²,head_in)}
                 ──► reduce ──► map_head ──► V                           {n:(M,n²,d_out)}, L2

    score(q, tile j, scale n):
        e   = ⟨V[n]_j , q⟩                       (query-conditioned cell logits)
        A   = softmax(e / τ_n)   over the n² cells
        m   = L2( Σ_cells A · V[n]_j )           (query-conditioned tile vector)
        S_n = ⟨m , q⟩
        β   = ScaleGate(q)       (query-conditioned scale weights)
        score = Σ_n β_n · S_n

    τ_n = τ_min + softplus(ρ_n) for n>1 (trainable per-scale ρ); τ = 1 for the 1×1 scale.

Heads are SEPARATE per domain (Stage-2 ``ProjectionHead`` with LayerNorm). The SuperVLAD
assignment and the per-group reducer are SHARED between UAV and map. Nothing here is the
Stage-3 static ``cell_attn``/``scale_fuse`` — those belong to the ``static_global_descriptor``
baseline only.

DO NOT edit the math without re-running the parity test — the whole point is bit-parity with
the proven Stage-2 model.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── cell geometry (inlined from siam_model_stage*/vlad_extract.cell_edges) ──────────────
def cell_edges(total_tokens: int, n_cells: int) -> np.ndarray:
    """``n_cells+1`` integer token boundaries splitting a token axis into n cells."""
    return np.linspace(0, total_tokens, n_cells + 1).round().astype(int)


_REGION_ID_CACHE: dict = {}


def region_ids(Hm: int, Wm: int, n: int, device) -> torch.Tensor:
    """(N,) region index of each token in an n×n cell grid over an Hm×Wm token grid.

    Reproduces Stage-1/2 ``build_pyramid`` exactly: ``cell_edges`` boundaries +
    ``searchsorted(..., right=True)`` so cell membership is byte-for-byte identical."""
    er, ec = cell_edges(Hm, n), cell_edges(Wm, n)
    if (np.diff(er) == 0).any() or (np.diff(ec) == 0).any():
        raise ValueError(f"token grid {Hm}×{Wm} too small for {n}×{n} cells")
    h_idx = torch.arange(Hm, device=device).repeat_interleave(Wm)
    w_idx = torch.arange(Wm, device=device).repeat(Hm)
    er_t = torch.as_tensor(er[1:], device=device, dtype=torch.long)
    ec_t = torch.as_tensor(ec[1:], device=device, dtype=torch.long)
    row_id = torch.searchsorted(er_t.contiguous(), h_idx.contiguous(), right=True)
    col_id = torch.searchsorted(ec_t.contiguous(), w_idx.contiguous(), right=True)
    return row_id * n + col_id


def _region_ids_cached(Hm: int, Wm: int, n: int, device) -> torch.Tensor:
    dev = torch.device(device)
    key = (int(Hm), int(Wm), int(n), str(dev))
    t = _REGION_ID_CACHE.get(key)
    if t is None or t.device != dev or t.is_inference():
        with torch.inference_mode(False), torch.no_grad():
            t = region_ids(Hm, Wm, n, dev)
        _REGION_ID_CACHE[key] = t
    return t


# ── center-free SuperVLAD aggregator (verbatim from supervlad_core.CellSuperVLAD) ───────
class CellSuperVLAD(nn.Module):
    """Learned center-free soft-group pooling: α_k=softmax(W_a·φ+b)[:K], U=Σ G·α·ψ."""

    def __init__(self, d_token: int, n_groups: int, d_value: int | None = None,
                 value_proj: str = "identity", phi_proj: str = "identity",
                 d_phi: int | None = None, n_ghost: int = 1,
                 assign_bias: bool = False, freeze_assign: bool = False):
        super().__init__()
        self.d_token = int(d_token)
        self.n_groups = int(n_groups)
        self.n_ghost = max(0, int(n_ghost))
        self.value_proj = value_proj
        self.phi_proj = phi_proj
        self.assign_bias = bool(assign_bias)
        self.freeze_assign = bool(freeze_assign)

        if phi_proj == "linear":
            d_phi = int(d_phi or d_token)
            self.phi = nn.Linear(d_token, d_phi)
            assign_in = d_phi
        elif phi_proj == "identity":
            self.phi = None
            assign_in = d_token
        else:
            raise ValueError(f"phi_proj must be identity|linear, got {phi_proj!r}")

        if value_proj == "linear":
            self.d_value = int(d_value or d_token)
            self.psi = nn.Linear(d_token, self.d_value)
        elif value_proj == "identity":
            self.d_value = int(d_token)
            self.psi = None
        else:
            raise ValueError(f"value_proj must be identity|linear, got {value_proj!r}")

        self.n_out = self.n_groups + self.n_ghost
        self.assign = nn.Linear(assign_in, self.n_out, bias=self.assign_bias)
        self.assign_in = assign_in
        self.init_meta: dict = {"method": "torch_default", "seed": None}
        if freeze_assign:
            self.set_assign_frozen(True)

    def set_assign_frozen(self, frozen: bool = True):
        self.freeze_assign = bool(frozen)
        for p in self.assign.parameters():
            p.requires_grad_(not frozen)
        if self.phi is not None:
            for p in self.phi.parameters():
                p.requires_grad_(not frozen)

    @property
    def d_super(self) -> int:
        return self.n_groups * self.d_value

    def alpha_full(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens if self.phi is None else self.phi(tokens)
        return F.softmax(self.assign(x), dim=-1)

    def alpha(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.alpha_full(tokens)[..., :self.n_groups]

    def value(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens if self.psi is None else self.psi(tokens)


# ── per-group post-pooling projection (verbatim from supervlad_core.PerGroupProjection) ─
class PerGroupProjection(nn.Module):
    """Independent bias-free A_k: ℝ^{D_v}→ℝ^{d_group} per output group; shared UAV/map."""

    def __init__(self, n_groups: int, d_in: int, d_out: int, init: str = "orthogonal",
                 eps: float = 1e-8):
        super().__init__()
        self.n_groups = int(n_groups)
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.eps = float(eps)
        self.init = init
        self.weight = nn.Parameter(torch.empty(self.n_groups, self.d_out, self.d_in))
        self.reset_parameters(init)

    @torch.no_grad()
    def reset_parameters(self, init: str):
        if init == "orthogonal":
            for k in range(self.n_groups):
                nn.init.orthogonal_(self.weight[k])
        elif init == "normal":
            self.weight.normal_(0.0, 1.0 / math.sqrt(self.d_in))
        else:
            raise ValueError(f"group_proj_init must be orthogonal|normal, got {init!r}")

    def project_group(self, x: torch.Tensor, k: int) -> torch.Tensor:
        return F.linear(x, self.weight[k])

    def forward(self, groups: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...kd,kod->...ko", groups, self.weight)

    @property
    def d_reduced(self) -> int:
        return self.n_groups * self.d_out

    def num_params(self) -> int:
        return self.n_groups * self.d_out * self.d_in


# ── pyramid (map side) — verbatim from supervlad_core ───────────────────────────────────
def build_super_pyramid_batched(tok_grids: torch.Tensor, agg: CellSuperVLAD,
                                scales_cells: list[int], intra: bool = True,
                                eps: float = 1e-8,
                                group_proj: "PerGroupProjection | None" = None
                                ) -> dict[int, torch.Tensor]:
    """{n: (M, n², d_reduced)} center-free region descriptors for a batch of tiles."""
    M, Hm, Wm, D = tok_grids.shape
    dev = tok_grids.device
    flat = F.normalize(tok_grids.reshape(M, Hm * Wm, D), dim=-1, eps=eps)
    alpha = agg.alpha(flat)
    value = agg.value(flat)
    if group_proj is not None:
        return _pyramid_per_group_fast(alpha, value, group_proj, scales_cells, Hm, Wm, dev, eps)
    return _pyramid_reference(alpha, value, scales_cells, Hm, Wm, dev, intra, eps, None)


def _pyramid_reference(alpha, value, scales_cells, Hm, Wm, dev, intra, eps, group_proj):
    M = value.shape[0]
    K, Dv = alpha.shape[-1], value.shape[-1]
    pyr: dict[int, torch.Tensor] = {}
    for n in scales_cells:
        rid = _region_ids_cached(Hm, Wm, n, dev)
        blocks = []
        for k in range(K):
            contrib = alpha[..., k:k + 1] * value
            block = torch.zeros(M, n * n, Dv, device=dev, dtype=value.dtype)
            block.index_add_(1, rid, contrib)
            if intra:
                block = F.normalize(block, dim=2, eps=eps)
            if group_proj is not None:
                block = group_proj.project_group(block, k)
                block = F.normalize(block, dim=2, eps=eps)
            blocks.append(block)
        out = torch.cat(blocks, dim=2)
        pyr[n] = F.normalize(out, dim=2, eps=eps)
    return pyr


def _pyramid_per_group_fast(alpha, value, group_proj, scales_cells, Hm, Wm, dev, eps):
    """EXACT fast center-free pyramid for reducer=per_group (norm₂(A_k(norm₂(x)))=norm₂(A_k(x)),
    A_k bias-free linear → project each token first, then pool in K·d_group space)."""
    M, N, Dv = value.shape
    K = alpha.shape[-1]
    dg = group_proj.d_out
    w = group_proj.weight
    if w.dtype != value.dtype:
        w = w.to(value.dtype)
    pv = torch.einsum("mnd,kod->mnko", value, w)
    weighted = (pv * alpha.unsqueeze(-1)).reshape(M, N, K * dg)
    pyr: dict[int, torch.Tensor] = {}
    for n in scales_cells:
        rid = _region_ids_cached(Hm, Wm, n, dev)
        pooled = torch.zeros(M, n * n, K * dg, device=dev, dtype=weighted.dtype)
        pooled.index_add_(1, rid, weighted)
        pooled = F.normalize(pooled.reshape(M, n * n, K, dg), dim=3, eps=eps)
        pyr[n] = F.normalize(pooled.reshape(M, n * n, K * dg), dim=2, eps=eps)
    return pyr


def super_global(tok_keep: torch.Tensor, agg: CellSuperVLAD, intra: bool = True,
                 eps: float = 1e-8,
                 group_proj: "PerGroupProjection | None" = None) -> torch.Tensor:
    """(d_reduced,) global center-free SuperVLAD over (N, D) UAV tokens (Stage-2 q⁰)."""
    d_reduced = group_proj.d_reduced if group_proj is not None else agg.d_super
    if tok_keep.numel() == 0:
        return torch.zeros(d_reduced, device=tok_keep.device)
    x = F.normalize(tok_keep, dim=1, eps=eps)
    a = agg.alpha(x)
    v = agg.value(x)
    U = a.transpose(0, 1) @ v
    if intra:
        U = F.normalize(U, dim=1, eps=eps)
    if group_proj is not None:
        U = F.normalize(group_proj(U), dim=1, eps=eps)
    u = U.reshape(-1)
    return F.normalize(u, dim=0, eps=eps)


# ── heads / scale gate (verbatim from siam_model_stage2/train.py) ───────────────────────
class ProjectionHead(nn.Module):
    """Linear→LN→GELU→Dropout→Linear→L2. Separate weights per domain."""

    def __init__(self, d_in: int, d_out: int, d_hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class ScaleGate(nn.Module):
    """g(q): UAV-only → softmax over scales. Sees the query, never the map."""

    def __init__(self, d_out: int, n_scales: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_out, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, n_scales),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(q), dim=-1)


# ── assignment initialisation from pre-existing VLAD centroids (verbatim) ───────────────
@torch.no_grad()
def assignment_from_centroids(centroids: torch.Tensor, tokens: torch.Tensor,
                              init_prob: float = 0.01, eps: float = 1e-8) -> dict:
    """Build a bias-free assignment init from PRE-EXISTING VLAD centroids (no k-means).

    ``centroids`` (K,D) legacy VLAD centres in the SAME feature space as ``tokens``. γ is
    calibrated on ``tokens`` from the mean top1−top2 cosine gap. G=0 (legacy VLAD, no ghost)."""
    C = F.normalize(centroids.float(), dim=1, eps=eps)
    x = F.normalize(tokens.float(), dim=1, eps=eps)
    K, D = C.shape
    sims = x @ C.t()
    if sims.shape[1] >= 2:
        top = sims.topk(2, dim=1).values
        gap = float((top[:, 0] - top[:, 1]).clamp_min(eps).mean())
    else:
        gap = float(sims.max(1).values.mean().clamp_min(eps))
    gamma = float(-math.log(init_prob) / max(gap, eps))
    return {"assign_weight": (gamma * C).cpu(), "centroids": C.cpu(), "alpha": gamma,
            "mean_top1_top2_gap": gap, "n_groups": int(K), "n_ghost": 0,
            "n_out": int(K), "d_assign": int(D)}


@torch.no_grad()
def apply_assignment_init(agg: CellSuperVLAD, assign_weight: torch.Tensor,
                          meta: dict | None = None):
    """Copy a fitted (K+G, D_assign) assignment weight into ``agg`` (bias zeroed)."""
    if agg.phi is not None:
        raise ValueError("assignment init requires phi_proj='identity'.")
    exp = (agg.n_out, agg.assign_in)
    if tuple(assign_weight.shape) != exp:
        raise ValueError(f"assign_weight shape {tuple(assign_weight.shape)} != {exp} (K+G, d_assign)")
    agg.assign.weight.copy_(assign_weight.to(agg.assign.weight.dtype).to(agg.assign.weight.device))
    if agg.assign.bias is not None:
        agg.assign.bias.zero_()
    agg.init_meta = dict(meta) if meta else {"method": "applied"}
    return agg.init_meta


# ── the query-conditioned model (mirrors siam_model_stage2/train.MultiScaleCellSuperVLAD) ─
class Stage2QueryConditionedModel(nn.Module):
    """Stage-2 model contract for the ``stage2_query_conditioned`` mode, reducer=per_group.

    Holds the shared aggregation (``agg`` + ``group_proj``), the SEPARATE ``drone_head`` /
    ``map_head`` (LayerNorm ProjectionHead), the per-scale temperatures ``rho`` and the
    query-conditioned ``scale_gate``. ``reduce`` is identity for per_group (no P₀).

    The map descriptor is a CELL PYRAMID ``{n:(M,n²,d_out)}`` — never a single vector — and
    scoring is query-conditioned (``score_from_V`` / ``score_queries_against_tiles``)."""

    def __init__(self, agg: CellSuperVLAD, group_proj: PerGroupProjection,
                 scales_cells=(8, 4, 2, 1), *, d_out=256, head_hidden=512, dropout=0.1,
                 tau_min=0.01, tau_init=0.1, intra=True, eps=1e-8):
        super().__init__()
        self.agg = agg
        self.group_proj = group_proj
        self.scales_cells = list(scales_cells)
        self.scales = tuple(scales_cells)            # alias (static model exposes .scales)
        self.tau_min = float(tau_min)
        self.intra = bool(intra)
        self.eps = float(eps)
        # per_group reducer ⇒ identity reduce (no P₀); head_in = K·d_group
        self.use_p0 = False
        self.P0 = None
        head_in = group_proj.d_reduced
        self.head_in = head_in
        self.d_out = int(d_out)
        self.out_dim = int(d_out)                    # alias so downstream code is mode-agnostic
        self.scoring_mode = "stage2_query_conditioned"
        self.drone_head = ProjectionHead(head_in, d_out, head_hidden, dropout)
        self.map_head = ProjectionHead(head_in, d_out, head_hidden, dropout)
        self.scale_gate = ScaleGate(d_out, len(self.scales_cells), dropout)
        rho0 = math.log(math.expm1(max(tau_init - tau_min, 1e-4)))   # inverse-softplus
        self.rho = nn.ParameterDict({
            str(n): nn.Parameter(torch.tensor(rho0)) for n in self.scales_cells if n > 1
        })

    # ── plumbing ────────────────────────────────────────────────────────────────────
    def _dev(self):
        return self.drone_head.net[0].weight.device

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.P0 if self.use_p0 else x

    def tau(self, n: int) -> torch.Tensor:
        if n <= 1:
            return torch.ones((), device=self._dev())
        return self.tau_min + F.softplus(self.rho[str(n)])

    def freeze_assignment(self, frozen=True):
        self.agg.set_assign_frozen(frozen)

    # ── map side: token grids → cell pyramid V ────────────────────────────────────────
    def region_from_tiles(self, tok_grids: torch.Tensor) -> dict:
        pyr = build_super_pyramid_batched(tok_grids, self.agg, self.scales_cells,
                                          self.intra, self.eps, group_proj=self.group_proj)
        return {n: self.reduce(pyr[n]) for n in self.scales_cells}

    def transform_map(self, region: dict) -> dict:
        return {n: self.map_head(region[n]) for n in self.scales_cells}

    def build_V(self, tok_grids: torch.Tensor, *, tile_chunk: int | None = None) -> dict:
        """(M,H,W,D) → V {n:(M,n²,d_out)} (aggregate + reduce + map_head). ``tok_grids`` may live on
        CPU; each chunk is moved to the parameter device so peak GPU memory is one chunk."""
        dev = self._dev()
        if not tile_chunk or tok_grids.shape[0] <= tile_chunk:
            return self.transform_map(self.region_from_tiles(tok_grids.to(dev)))
        parts: dict[int, list] = {n: [] for n in self.scales_cells}
        for s in range(0, tok_grids.shape[0], int(tile_chunk)):
            V = self.transform_map(self.region_from_tiles(tok_grids[s:s + int(tile_chunk)].to(dev)))
            for n in self.scales_cells:
                parts[n].append(V[n])
        return {n: torch.cat(parts[n], 0) for n in self.scales_cells}

    # ── query side: UAV tokens → q ────────────────────────────────────────────────────
    def encode_query_from_tokens(self, tok_keep: torch.Tensor) -> torch.Tensor:
        u = super_global(tok_keep, self.agg, self.intra, self.eps, group_proj=self.group_proj)
        return self.drone_head(self.reduce(u))

    def encode_queries_from_grids(self, grids: torch.Tensor) -> torch.Tensor:
        """(B,H,W,D) UAV token grids → q (B,d_out). Flattens each grid to (H·W,D) tokens.

        NOTE (data-pipeline vs model contract): Stage-2 drops sky/black tokens BEFORE
        super_global; Stage-3 sky-FILLS them upstream, so here every grid cell is a token.
        This is a preprocessing difference, documented — not a model-contract change."""
        dev = self._dev()
        B, D = grids.shape[0], grids.shape[-1]
        qs = [self.encode_query_from_tokens(grids[i].reshape(-1, D).to(dev)) for i in range(B)]
        return torch.stack(qs, 0)

    # ── scoring (VERBATIM Stage-2 single-query score_from_V + batched training variant) ─
    def score_from_V(self, q: torch.Tensor, V: dict):
        """Stage-2 ``score_from_V``: one query q (d,) vs V{n:(M,n²,d)} → scores (M,)."""
        S_cols, ent_cols = [], []
        for n in self.scales_cells:
            e = torch.einsum("mnd,d->mn", V[n], q)
            A = F.softmax(e / self.tau(n), dim=1)
            m = F.normalize((A.unsqueeze(-1) * V[n]).sum(1), dim=1)
            S_cols.append(m @ q)
            ent = -(A.clamp_min(1e-9).log() * A).sum(1).mean()
            ent_cols.append(ent)
        S_stack = torch.stack(S_cols, dim=1)
        beta = self.scale_gate(q.unsqueeze(0))[0]
        scores = (beta.unsqueeze(0) * S_stack).sum(1)
        return scores, S_stack, beta, torch.stack(ent_cols)

    def score_queries_against_tiles(self, q: torch.Tensor, V: dict, *,
                                    tile_chunk: int | None = None) -> torch.Tensor:
        """Vectorised query-conditioned scoring: q (B,d), V{n:(M,n²,d)} → scores (B,M).

        Numerically identical to looping ``score_from_V`` over B queries (see parity test).
        ``tile_chunk`` bounds the (B, m, n², d) intermediate for full-gallery scoring."""
        B = q.shape[0]
        M = V[self.scales_cells[0]].shape[0]
        beta = self.scale_gate(q)                                  # (B, n_scales)
        cs = int(tile_chunk) if tile_chunk else M
        cols = []
        for s0 in range(0, M, cs):
            S_cols = []
            for si, n in enumerate(self.scales_cells):
                Vn = V[n][s0:s0 + cs]                              # (m, n², d)
                e = torch.einsum("mrd,bd->bmr", Vn, q)            # (B, m, n²)
                A = F.softmax(e / self.tau(n), dim=2)             # softmax over cells
                m = F.normalize((A.unsqueeze(-1) * Vn.unsqueeze(0)).sum(2), dim=2)  # (B, m, d)
                S_cols.append(torch.einsum("bmd,bd->bm", m, q))  # (B, m)
            S_stack = torch.stack(S_cols, dim=2)                  # (B, m, n_scales)
            cols.append(torch.einsum("bs,bms->bm", beta, S_stack))
        return torch.cat(cols, dim=1)                             # (B, M)


def build_stage2_query_conditioned_model(*, d_token=1536, n_groups=32, n_ghost=0,
                                         group_projection_dim=32, group_proj_init="orthogonal",
                                         scales_cells=(8, 4, 2, 1), d_out=256, head_hidden=512,
                                         dropout=0.1, tau_min=0.01, tau_init=0.1,
                                         assign_weight: torch.Tensor | None = None,
                                         centroids: torch.Tensor | None = None,
                                         calib_tokens: torch.Tensor | None = None,
                                         freeze_assignment=True) -> Stage2QueryConditionedModel:
    """Build the Stage-2-faithful model (K=32, ghost=0, per_group, identity ψ/φ).

    Assignment init (frozen VLAD, same as the successful Stage-2 run): either pass a ready
    ``assign_weight`` (K,D) [the recast SuperVLAD dictionary, e.g. assign_init_vlad.pt], OR
    ``centroids`` (K,D) + ``calib_tokens`` (N,D) to derive it via ``assignment_from_centroids``."""
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
    return Stage2QueryConditionedModel(agg, red, scales_cells, d_out=d_out, head_hidden=head_hidden,
                                       dropout=dropout, tau_min=tau_min, tau_init=tau_init)
