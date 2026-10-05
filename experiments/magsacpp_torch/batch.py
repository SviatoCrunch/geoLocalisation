"""Cross-pair vectorized MAGSAC++ homography (the GPU throughput path).

``estimate_homography_magsacpp_batch`` processes MANY independent correspondence pairs at once:
the minimal solve, the per-hypothesis forward residual, the marginalized loss scoring and the
sigma-consensus++ IRLS all run as batched tensor ops over (pairs x hypotheses), instead of the
per-pair Python loop in :func:`estimator.estimate_homography_magsacpp`. The MATH is identical --
same :func:`solver.solve_minimal_h` / :func:`solver.solve_weighted_h` / :func:`gamma.GammaMath` --
so on a shared ``hypothesis_indices`` schedule the batch path reproduces the per-pair path
bit-for-bit (asserted by ``tests/test_batch.py``); only the sampling RNG differs in free-run mode.

Memory is bounded by processing pairs in GROUPS of ``g = points_budget // (S*N)`` at a time, so the
largest work tensor is ``(g*S, N)`` regardless of how many pairs are passed.
"""
from __future__ import annotations

from typing import List, Optional

import torch

from .config import MagsacppConfig
from .estimator import HomographyResult, _collinear_guard, _gamma_for
from .solver import forward_sq_residual, solve_minimal, solve_weighted_h


@torch.inference_mode()
def estimate_homography_magsacpp_batch(
    points1: torch.Tensor,
    points2: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    *,
    config: Optional[MagsacppConfig] = None,
    generator: Optional[torch.Generator] = None,
    hypothesis_indices: Optional[torch.Tensor] = None,
) -> List[HomographyResult]:
    """Vectorized batch estimate. ``points1``/``points2`` = (P, N, 2) padded tensors on one device,
    ``valid_mask`` = (P, N) bool. Returns a list of P :class:`HomographyResult` (trace is always None).
    ``hypothesis_indices`` optional (P, S, 4) recorded schedule (replay/parity)."""
    cfg = config or MagsacppConfig()
    if points1.dim() != 3 or points1.shape[-1] != 2 or points1.shape != points2.shape:
        raise ValueError("points1/points2 must be matching (P,N,2) tensors")
    device = points1.device
    P, N, _ = points1.shape
    p1 = points1.to(cfg.dtype)
    p2 = points2.to(cfg.dtype)
    if valid_mask is None:
        valid = torch.ones(P, N, dtype=torch.bool, device=device)
    else:
        valid = valid_mask.to(device=device, dtype=torch.bool)
    p1 = torch.where(valid.unsqueeze(-1), p1, torch.zeros_like(p1))
    p2 = torch.where(valid.unsqueeze(-1), p2, torch.zeros_like(p2))
    gm = _gamma_for(cfg, device)
    S = cfg.max_hypotheses if hypothesis_indices is None else hypothesis_indices.shape[1]
    ms = cfg.min_sample
    INF = float("inf")

    nan_H = torch.full((3, 3), float("nan"), dtype=cfg.dtype, device=device)
    results: List[Optional[HomographyResult]] = [None] * P
    nvalid = valid.sum(1)
    todo = [i for i in range(P) if int(nvalid[i]) >= ms]
    for i in range(P):
        if i not in set(todo):
            st = "all_invalid" if int(nvalid[i]) == 0 else "n_lt_4"
            results[i] = HomographyResult(nan_H.clone(), False, st, 0.0, INF,
                                          torch.zeros(N, dtype=torch.bool, device=device), 0, 0, 0,
                                          cfg.sigma_max)
    if not todo:
        return results  # type: ignore

    # group pairs so the (g*S, N) work tensor stays within budget
    gP = max(1, cfg.points_budget // max(1, S * N))
    for g0 in range(0, len(todo), gP):
        gidx = torch.tensor(todo[g0:g0 + gP], device=device)
        g = gidx.numel()
        P1 = p1[gidx]                                   # (g, N, 2)
        P2 = p2[gidx]
        V = valid[gidx]                                 # (g, N)

        # --- sample / replay S minimal samples per pair ---
        if hypothesis_indices is not None:
            samp = hypothesis_indices[gidx].to(device=device, dtype=torch.long)   # (g, S, 4)
        else:
            w = V.double().repeat_interleave(S, 0)      # (g*S, N)
            samp = torch.multinomial(w, ms, replacement=False, generator=generator).view(g, S, ms)
        pair_local = torch.arange(g, device=device).repeat_interleave(S)          # (g*S,)
        s_flat = samp.reshape(g * S, ms)                                          # (g*S, 4)
        gath = s_flat.unsqueeze(-1).expand(-1, -1, 2)                             # (g*S, 4, 2)
        p1s = P1[pair_local].gather(1, gath)            # (g*S, 4, 2)
        p2s = P2[pair_local].gather(1, gath)
        nondeg = _collinear_guard(p1s, cfg.collinearity_eps) & _collinear_guard(p2s, cfg.collinearity_eps)
        Hs, ok = solve_minimal(p1s, p2s, cfg.minimal_solver, cfg.min_singular_ratio)  # (g*S,3,3),(g*S,)
        ok = ok & nondeg

        # --- score every hypothesis against ALL its pair's points ---
        sq = forward_sq_residual(Hs, P1[pair_local], P2[pair_local], cfg.min_abs_denominator)  # (g*S,N)
        loss = gm.total_loss(sq, valid=V[pair_local], dim=1)                      # (g*S,)
        loss = torch.where(ok, loss, torch.full_like(loss, INF)).view(g, S)
        best_loss, best_s = torch.min(loss, dim=1)                               # (g,),(g,)
        ar = torch.arange(g, device=device)
        bestH = Hs.view(g, S, 3, 3)[ar, best_s].clone()                          # (g,3,3)
        okg = ok.view(g, S)
        feasible = torch.isfinite(best_loss) & okg.any(1)

        # --- sigma-consensus++ IRLS (batched over the group) ---
        for _ in range(cfg.irls_iters):
            sqb = forward_sq_residual(bestH, P1, P2, cfg.min_abs_denominator)     # (g,N)
            wts = torch.where(V, gm.weight(sqb), torch.zeros_like(sqb))           # (g,N)
            enough = (wts > 0).sum(1) >= ms
            Href, okr = solve_weighted_h(P1, P2, wts, V, cfg.min_singular_ratio)  # (g,3,3),(g,)
            sqr = forward_sq_residual(Href, P1, P2, cfg.min_abs_denominator)
            loss_r = gm.total_loss(sqr, valid=V, dim=1)                           # (g,)
            cond = (loss_r <= best_loss) if cfg.irls_require_improvement \
                else torch.ones_like(okr)
            improve = okr & enough & cond
            bestH = torch.where(improve.view(g, 1, 1), Href, bestH)
            best_loss = torch.where(improve, loss_r, best_loss)

        # --- explicit inlier mask (separate threshold) ---
        sqf = forward_sq_residual(bestH, P1, P2, cfg.min_abs_denominator)         # (g,N)
        inl = (sqf <= cfg.inlier_threshold ** 2) & V                             # (g,N)
        for j in range(g):
            i = int(gidx[j])
            if not bool(feasible[j]):
                results[i] = HomographyResult(nan_H.clone(), False, "no_valid_hypothesis", 0.0, INF,
                                              torch.zeros(N, dtype=torch.bool, device=device),
                                              0, int(okg[j].sum()), 0, cfg.sigma_max)
                continue
            tl = float(best_loss[j])
            results[i] = HomographyResult(bestH[j].clone(), True, "ok",
                                          (1.0 / tl) if tl > 0 else INF, tl, inl[j].clone(),
                                          int(inl[j].sum()), int(okg[j].sum()), cfg.irls_iters,
                                          cfg.sigma_max)
    return results  # type: ignore
