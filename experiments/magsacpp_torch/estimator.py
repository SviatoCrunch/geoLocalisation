"""Public estimator: marginalized robust homography fitting in Torch (MAGSAC++ formulas).

``estimate_homography_magsacpp`` takes matched coordinate pairs (map-crop -> query, the
patch_rerank convention) and returns the homography, a marginalized-loss-based quality score, an
explicit inlier mask/count, and diagnostics. Everything in the fitting / scoring / refinement path
is pure Torch (float64 correctness mode), ``torch.inference_mode``; NumPy/OpenCV are not used.

Pipeline per pair:
  1. sample ``max_hypotheses`` minimal 4-point samples WITHOUT replacement (or replay a recorded
     ``hypothesis_indices`` schedule for parity),
  2. solve each minimal homography (correct 9-vector nullspace, Hartley-normalized),
  3. score every hypothesis by the total marginalized MAGSAC++ loss over ALL the pair's points
     (chunked over hypotheses so the (chunk, N) work tensor stays within the memory budget),
  4. take the minimum-loss hypothesis, run sigma-consensus++ IRLS (weights -> weighted refit),
     accepting a refit only if the total loss did not increase,
  5. report inliers with the SEPARATE explicit threshold (never used by the score).

Hypothesis scoring is independent per hypothesis, so the selected model is invariant to the
hypothesis chunk size (ties aside) -- this is asserted by the test suite.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import torch

from .config import MagsacppConfig
from .gamma import GammaLUT, GammaMath
from .solver import forward_sq_residual, solve_minimal_h, solve_weighted_h

ArrayLike = Union[torch.Tensor, "Sequence"]


@dataclass
class HomographyResult:
    """Result for a single pair (batched API returns a list of these)."""
    H: torch.Tensor                       # (3,3) map->query, or NaN if failed
    success: bool
    status: str                           # ok | n_lt_4 | all_invalid | no_valid_hypothesis | degenerate
    score: float                          # quality = 1/total_loss (higher better); 0.0 on failure
    total_loss: float                     # marginalized MAGSAC++ loss (lower better); inf on failure
    inlier_mask: torch.Tensor             # (N,) bool over the INPUT points (padded -> False)
    inlier_count: int
    n_hypotheses: int                     # valid minimal hypotheses actually scored
    n_irls_steps: int
    sigma_max: float
    trace: Optional[dict] = field(default=None)


def _as_bNx2(points: ArrayLike, dtype, device) -> torch.Tensor:
    t = points if isinstance(points, torch.Tensor) else torch.as_tensor(points)
    t = t.to(dtype=dtype, device=device)
    if t.dim() == 2:
        t = t.unsqueeze(0)
    if t.dim() != 3 or t.shape[-1] != 2:
        raise ValueError(f"points must be (N,2) or (B,N,2); got {tuple(t.shape)}")
    return t


def _collinear_guard(p1: torch.Tensor, eps: float) -> torch.Tensor:
    """(S,) bool: True if a minimal sample (S,4,2) is NON-degenerate (no ~collinear triple)."""
    # areas of the 4 triangles from choosing 3 of the 4 points; all must exceed eps (scaled by
    # the sample's own spread so the test is scale-aware).
    idx = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    spread = (p1.amax(1) - p1.amin(1)).norm(dim=-1).clamp_min(1e-12)   # (S,)
    ok = torch.ones(p1.shape[0], dtype=torch.bool, device=p1.device)
    for a, b, c in idx:
        pa, pb, pc = p1[:, a], p1[:, b], p1[:, c]
        area2 = ((pb[:, 0] - pa[:, 0]) * (pc[:, 1] - pa[:, 1])
                 - (pb[:, 1] - pa[:, 1]) * (pc[:, 0] - pa[:, 0])).abs()
        ok = ok & (area2 > eps * spread * spread)
    return ok


def _sample_indices(valid: torch.Tensor, n_hyp: int, min_sample: int,
                    generator: Optional[torch.Generator]) -> torch.Tensor:
    """(n_hyp, min_sample) distinct point indices drawn WITHOUT replacement from valid points."""
    w = valid.to(torch.float64).expand(n_hyp, -1).contiguous()
    return torch.multinomial(w, min_sample, replacement=False, generator=generator)


def _gamma_for(cfg: MagsacppConfig, device) -> Union[GammaMath, GammaLUT]:
    gm = GammaMath(cfg.dof, cfg.quantile_k, cfg.normalizer_c, cfg.sigma_max, dtype=cfg.dtype)
    return GammaLUT(gm) if cfg.use_lut else gm


@torch.inference_mode()
def _estimate_one(p1: torch.Tensor, p2: torch.Tensor, valid: torch.Tensor,
                  cfg: MagsacppConfig, generator: Optional[torch.Generator],
                  schedule: Optional[torch.Tensor], gammamath, return_trace: bool) -> HomographyResult:
    """Single pair. p1,p2 (N,2), valid (N,) bool. ``schedule`` (S,4) optional replay indices."""
    device = p1.device
    N = p1.shape[0]
    nan_H = torch.full((3, 3), float("nan"), dtype=cfg.dtype, device=device)
    empty_mask = torch.zeros(N, dtype=torch.bool, device=device)
    n_valid = int(valid.sum())
    if n_valid < cfg.min_sample:
        status = "n_lt_4" if N >= 1 else "all_invalid"
        if n_valid == 0:
            status = "all_invalid"
        return HomographyResult(nan_H, False, status, 0.0, float("inf"), empty_mask, 0, 0, 0,
                                cfg.sigma_max)

    # --- 1. hypotheses (sampled or replayed) ---
    if schedule is not None:
        samp = schedule.to(device=device, dtype=torch.long)
        if samp.dim() != 2 or samp.shape[1] != cfg.min_sample:
            raise ValueError(f"hypothesis_indices must be (S,{cfg.min_sample}); got {tuple(samp.shape)}")
        if not bool(valid[samp].all()):
            raise ValueError("hypothesis_indices reference an invalid/padded point")
    else:
        samp = _sample_indices(valid, cfg.max_hypotheses, cfg.min_sample, generator)   # (S,4)
    S = samp.shape[0]

    p1s = p1[samp]                                  # (S, 4, 2)
    p2s = p2[samp]
    nondeg = _collinear_guard(p1s, cfg.collinearity_eps) & _collinear_guard(p2s, cfg.collinearity_eps)
    Hs, ok = solve_minimal_h(p1s, p2s, cfg.min_singular_ratio)       # (S,3,3), (S,)
    ok = ok & nondeg
    if not bool(ok.any()):
        return HomographyResult(nan_H, False, "degenerate", 0.0, float("inf"), empty_mask, 0,
                                int(S), 0, cfg.sigma_max)

    # --- 2+3. score every valid hypothesis by total marginalized loss (chunked over hypotheses) ---
    INF = torch.tensor(float("inf"), dtype=cfg.dtype, device=device)
    best_loss = INF.clone()
    best_idx = -1
    valid_row = valid.view(1, N)
    # chunk so that chunk_size * N <= points_budget
    chunk = max(1, min(S, cfg.points_budget // max(1, N)))
    per_hyp_loss = torch.full((S,), float("inf"), dtype=cfg.dtype, device=device)
    for lo in range(0, S, chunk):
        hi = min(S, lo + chunk)
        Hc = Hs[lo:hi]                                  # (c,3,3)
        okc = ok[lo:hi]
        sq = forward_sq_residual(Hc.unsqueeze(1), p1.view(1, N, 2), p2.view(1, N, 2),
                                 cfg.min_abs_denominator).squeeze(1)      # (c, N)
        loss = gammamath.total_loss(sq, valid=valid_row.expand(hi - lo, N), dim=1)   # (c,)
        loss = torch.where(okc, loss, INF)
        per_hyp_loss[lo:hi] = loss
        cmin, cpos = torch.min(loss, dim=0)
        if float(cmin) < float(best_loss):
            best_loss = cmin
            best_idx = lo + int(cpos)
    if best_idx < 0 or not torch.isfinite(best_loss):
        return HomographyResult(nan_H, False, "no_valid_hypothesis", 0.0, float("inf"), empty_mask,
                                int(ok.sum()), 0, cfg.sigma_max)

    bestH = Hs[best_idx].clone()
    total_loss = best_loss.clone()

    # --- 4. sigma-consensus++ IRLS from the best hypothesis ---
    irls_done = 0
    for _ in range(cfg.irls_iters):
        sq = forward_sq_residual(bestH, p1, p2, cfg.min_abs_denominator)             # (N,)
        w = gammamath.weight(sq)                                                     # (N,)
        w = torch.where(valid, w, torch.zeros_like(w))
        if int((w > 0).sum()) < cfg.min_sample:
            break
        Href, ok_r = solve_weighted_h(p1.unsqueeze(0), p2.unsqueeze(0), w.unsqueeze(0),
                                      valid.unsqueeze(0), cfg.min_singular_ratio)
        if not bool(ok_r[0]):
            break
        sq_r = forward_sq_residual(Href[0], p1, p2, cfg.min_abs_denominator)
        loss_r = gammamath.total_loss(sq_r, valid=valid.view(1, N), dim=1)[0]
        irls_done += 1
        if (not cfg.irls_require_improvement) or float(loss_r) <= float(total_loss):
            bestH, total_loss = Href[0].clone(), loss_r.clone()
        else:
            break

    # --- 5. explicit inlier mask (SEPARATE threshold; never feeds the score) ---
    sq_final = forward_sq_residual(bestH, p1, p2, cfg.min_abs_denominator)
    inl = (sq_final <= cfg.inlier_threshold ** 2) & valid
    tl = float(total_loss)
    score = (1.0 / tl) if tl > 0 else float("inf")

    trace = None
    if return_trace:
        trace = {
            "schedule": samp.detach().cpu(),
            "per_hyp_loss": per_hyp_loss.detach().cpu(),
            "best_hypothesis": int(best_idx),
            "minimal_H": Hs[best_idx].detach().cpu(),
            "final_sq_residual": sq_final.detach().cpu(),
            "n_valid_hypotheses": int(ok.sum()),
            "reference": cfg.reference,
        }
    return HomographyResult(bestH, True, "ok", score, tl, inl, int(inl.sum()),
                            int(ok.sum()), irls_done, cfg.sigma_max, trace)


@torch.inference_mode()
def estimate_homography_magsacpp(
    points1: ArrayLike,
    points2: ArrayLike,
    valid_mask: Optional[ArrayLike] = None,
    *,
    config: Optional[MagsacppConfig] = None,
    generator: Optional[torch.Generator] = None,
    hypothesis_indices: Optional[ArrayLike] = None,
    return_trace: bool = False,
) -> Union[HomographyResult, List[HomographyResult]]:
    """Estimate homography(ies) ``points1 -> points2`` with the marginalized MAGSAC++ core.

    ``points1`` (source / map-crop) and ``points2`` (dest / query) are ``(N,2)`` for one pair or
    ``(B,N,2)`` for a batch sharing N (use ``valid_mask`` to pad ragged pairs). Returns a single
    :class:`HomographyResult` for a single pair, else a list aligned to the batch.

    ``hypothesis_indices`` is an optional recorded sample schedule ``(S,4)`` (single) or ``(B,S,4)``
    (batch) for parity/replay -- this bypasses the RNG so a C++ oracle and this code evaluate the
    SAME minimal samples (a shared seed alone does NOT guarantee that; see oracle.py / brief sec.8).
    """
    cfg = config or MagsacppConfig()
    device = points1.device if isinstance(points1, torch.Tensor) else torch.device("cpu")
    p1 = _as_bNx2(points1, cfg.dtype, device)
    p2 = _as_bNx2(points2, cfg.dtype, device)
    if p1.shape != p2.shape:
        raise ValueError(f"points1 {tuple(p1.shape)} and points2 {tuple(p2.shape)} must match")
    B, N, _ = p1.shape
    device = p1.device

    if valid_mask is None:
        valid = torch.ones(B, N, dtype=torch.bool, device=device)
    else:
        valid = (valid_mask if isinstance(valid_mask, torch.Tensor)
                 else torch.as_tensor(valid_mask)).to(device=device, dtype=torch.bool)
        if valid.dim() == 1:
            valid = valid.unsqueeze(0)
        if valid.shape != (B, N):
            raise ValueError(f"valid_mask must be (N,) or (B,N)=({B},{N}); got {tuple(valid.shape)}")
    # padded points carry non-finite coords sometimes; neutralize so solver math stays finite.
    p1 = torch.where(valid.unsqueeze(-1), p1, torch.zeros_like(p1))
    p2 = torch.where(valid.unsqueeze(-1), p2, torch.zeros_like(p2))

    sched_b = None
    if hypothesis_indices is not None:
        s = (hypothesis_indices if isinstance(hypothesis_indices, torch.Tensor)
             else torch.as_tensor(hypothesis_indices)).to(device=device, dtype=torch.long)
        sched_b = s.unsqueeze(0) if s.dim() == 2 else s
        if sched_b.shape[0] != B:
            raise ValueError("hypothesis_indices batch dim must match points batch")

    gammamath = _gamma_for(cfg, device)
    results = [
        _estimate_one(p1[b], p2[b], valid[b], cfg, generator,
                      None if sched_b is None else sched_b[b], gammamath, return_trace)
        for b in range(B)
    ]
    single = (not isinstance(points1, torch.Tensor) and _is_single(points1)) or \
             (isinstance(points1, torch.Tensor) and points1.dim() == 2)
    return results[0] if (single and B == 1) else results


def _is_single(x) -> bool:
    try:
        import numpy as np
        a = np.asarray(x)
        return a.ndim == 2
    except Exception:
        return False
