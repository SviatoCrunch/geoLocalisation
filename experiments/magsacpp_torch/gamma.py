"""Marginalized MAGSAC++ robust loss and sigma-consensus++ weights (incomplete-gamma math).

All quantities are expressed in terms of the SQUARED forward reprojection residual ``sq`` (px^2),
so no sqrt is ever needed. Let, with DoF ``n``, cutoff quantile ``k`` and noise bound ``sigma`` (=
``sigma_max``):

    t   = sq / (2 * sigma^2)            # gamma argument
    tk  = k^2 / 2                       # argument at the cutoff tau = k*sigma
    a1  = (n+1)/2 ,  a2 = (n-1)/2       # gamma shape parameters

PER-POINT LOSS  rho(sq)  -- paper Eq.(3) / danini getModelQualityPlusPlus (magsac.h ~L947-1045),
MAXIMIZES quality 1/sum(rho)  <=>  MINIMIZES total loss sum(rho):

    rho = (1/(sigma*C*2^((n+1)/2))) * [ sigma^2 * gl(a1, t)
                                        + (sq/4) * ( gl(a2, t) - gl(a2, tk) ) ]      for t <= tk
    rho = (1/(sigma*C*2^((n+1)/2))) *   sigma^2 * gl(a1, tk)                          for t  > tk
          (the second term vanishes at t=tk, so rho is continuous; outliers saturate)

PER-POINT WEIGHT  w(sq)  -- paper Eq.(2) / danini sigmaConsensusPlusPlus (magsac.h ~L835-865):

    w = (C*2^((n-1)/2)/sigma) * ( gu(a2, t) - gu(a2, tk) )       for t <= tk ,  else 0

where ``gl(a,x)`` = lower-incomplete gamma (unnormalized), ``gu(a,x)`` = upper-incomplete gamma
(unnormalized). We compute them analytically with ``torch.special.gammainc`` /
``torch.special.gammaincc`` (regularized) times ``Gamma(a)``; this runs on CPU and CUDA in float64
and float32. The optional LUT (:class:`GammaLUT`) is a drop-in OPTIMIZATION and must agree with the
analytic path within a documented tolerance.

Reference: danini/magsac @ d259f8b (graph-cut-ransac @ 9fa075d), paper arXiv:1912.05909, n=4.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

__all__ = ["GammaMath", "GammaLUT", "complete_gamma"]


def complete_gamma(a: float, dtype=torch.float64, device=None) -> torch.Tensor:
    """Complete gamma Gamma(a) = exp(lgamma(a)) as a tensor."""
    return torch.lgamma(torch.as_tensor(float(a), dtype=dtype, device=device)).exp()


def _gl(a: float, x: torch.Tensor, ga: torch.Tensor) -> torch.Tensor:
    """Unnormalized lower-incomplete gamma gl(a, x) = P(a, x) * Gamma(a)."""
    av = torch.full_like(x, float(a))
    return torch.special.gammainc(av, x) * ga


def _gu(a: float, x: torch.Tensor, ga: torch.Tensor) -> torch.Tensor:
    """Unnormalized upper-incomplete gamma gu(a, x) = Q(a, x) * Gamma(a)."""
    av = torch.full_like(x, float(a))
    return torch.special.gammaincc(av, x) * ga


@dataclass
class GammaMath:
    """Analytic marginalized loss + sigma-consensus++ weights for one (n, k, C, sigma_max)."""

    dof: int
    quantile_k: float
    normalizer_c: float
    sigma_max: float
    dtype: torch.dtype = torch.float64

    def __post_init__(self):
        n = self.dof
        self.a1 = (n + 1) / 2.0           # shape for the loss' first (saturating) term
        self.a2 = (n - 1) / 2.0           # shape for the loss' second term and the weights
        self.tk = (self.quantile_k ** 2) / 2.0
        self.sigma2 = self.sigma_max ** 2
        self.tau2 = (self.quantile_k * self.sigma_max) ** 2   # squared hard cutoff tau^2
        # precompute the constant gamma values at the cutoff (Python floats; cheap scalars)
        self._ga1 = complete_gamma(self.a1, self.dtype)
        self._ga2 = complete_gamma(self.a2, self.dtype)
        tk_t = torch.as_tensor(self.tk, dtype=self.dtype)
        self._gu_a2_tk = _gu(self.a2, tk_t, self._ga2)       # gu(a2, tk) -- weight & loss subtrahend
        self._gl_a1_tk = _gl(self.a1, tk_t, self._ga1)       # gl(a1, tk) -- loss saturation level
        # prefactors. The loss is rho(r) = integral_0^r x*w(x) dx of the (verified) weight; carrying
        # out that integral analytically gives
        #   rho = C*2^a2*sigma * [ gl(a1, t) + t*( gu(a2, t) - gu(a2, tk) ) ]
        # which is monotone non-decreasing and saturates to rho_sat at t=tk (second term -> 0).
        self._loss_pref = self.normalizer_c * (2.0 ** self.a2) * self.sigma_max
        self._w_pref = self.normalizer_c * (2.0 ** self.a2) / self.sigma_max
        # saturated loss (outlier loss), a scalar: rho at t=tk
        self.rho_sat = float(self._loss_pref * self._gl_a1_tk)

    # ----- core ------------------------------------------------------------------------------
    def _t(self, sq: torch.Tensor) -> torch.Tensor:
        return sq / (2.0 * self.sigma2)

    def loss(self, sq: torch.Tensor) -> torch.Tensor:
        """Per-point marginalized loss rho(sq). ``sq`` = squared residual (px^2), any shape.

        Non-finite / near-infinity residuals should arrive as +inf -> they saturate to ``rho_sat``
        (treated as outliers), never as NaN. Caller masks padded points separately (loss 0 there).
        """
        sq = sq.to(self.dtype)
        t = self._t(sq)
        ga1, ga2 = self._ga1.to(sq.device), self._ga2.to(sq.device)
        inside = sq <= self.tau2
        # use a finite clamp for the gamma calls so +inf -> gl=Gamma(a) cleanly (no NaN); the
        # ``inside`` mask discards those values anyway.
        t_safe = torch.where(inside, t, torch.full_like(t, self.tk))
        term1 = _gl(self.a1, t_safe, ga1)
        term2 = t_safe * (_gu(self.a2, t_safe, ga2) - self._gu_a2_tk.to(sq.device))
        rho_in = self._loss_pref * (term1 + term2)
        return torch.where(inside, rho_in, torch.full_like(rho_in, self.rho_sat))

    def weight(self, sq: torch.Tensor) -> torch.Tensor:
        """Per-point sigma-consensus++ weight w(sq) >= 0; zero beyond the cutoff tau^2."""
        sq = sq.to(self.dtype)
        t = self._t(sq)
        ga2 = self._ga2.to(sq.device)
        inside = sq <= self.tau2
        t_safe = torch.where(inside, t, torch.full_like(t, self.tk))
        w_in = self._w_pref * (_gu(self.a2, t_safe, ga2) - self._gu_a2_tk.to(sq.device))
        return torch.where(inside, w_in.clamp_min(0.0), torch.zeros_like(w_in))

    def total_loss(self, sq: torch.Tensor, valid: Optional[torch.Tensor] = None,
                   dim: int = -1) -> torch.Tensor:
        """Sum of per-point loss over ``dim``, excluding padded points (``valid`` False -> 0)."""
        rho = self.loss(sq)
        if valid is not None:
            rho = torch.where(valid, rho, torch.zeros_like(rho))
        return rho.sum(dim=dim)


class GammaLUT:
    """Round-indexed lookup table for the loss/weight terms (OPTIMIZATION; analytic is the truth).

    Tabulates the three argument-dependent gamma terms on a uniform grid of ``t = sq/(2 sigma^2)``
    over ``[0, tk]`` (beyond tk everything is the constant saturation / zero). This mirrors danini's
    round-indexed table (step 1e-4) but is regenerated here from the analytic path at the configured
    (n, k) rather than copied, so it is correct by construction for any DoF. Linear interpolation is
    used (danini uses nearest/round); set ``interp=False`` for round-indexing parity experiments.
    """

    def __init__(self, math_: GammaMath, size: int = 100_000, interp: bool = True):
        self.m = math_
        self.size = int(size)
        self.interp = bool(interp)
        self.tk = math_.tk
        dt = math_.dtype
        t = torch.linspace(0.0, self.tk, self.size, dtype=dt)
        ga1, ga2 = math_._ga1, math_._ga2
        self._gl_a1 = _gl(math_.a1, t, ga1)
        self._gl_a2 = _gl(math_.a2, t, ga2)
        self._gu_a2 = _gu(math_.a2, t, ga2)
        self._step = self.tk / (self.size - 1)

    def _lookup(self, table: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pos = (t / self._step).clamp(0, self.size - 1)
        tbl = table.to(t.device)
        if not self.interp:
            return tbl[pos.round().long()]
        lo = pos.floor().long()
        hi = (lo + 1).clamp_max(self.size - 1)
        frac = (pos - lo.to(pos.dtype))
        return tbl[lo] * (1 - frac) + tbl[hi] * frac

    def loss(self, sq: torch.Tensor) -> torch.Tensor:
        sq = sq.to(self.m.dtype)
        t = self.m._t(sq)
        inside = sq <= self.m.tau2
        t_safe = torch.where(inside, t, torch.full_like(t, self.tk))
        term1 = self._lookup(self._gl_a1, t_safe)
        term2 = t_safe * (self._lookup(self._gu_a2, t_safe) - self.m._gu_a2_tk.to(sq.device))
        rho_in = self.m._loss_pref * (term1 + term2)
        return torch.where(inside, rho_in, torch.full_like(rho_in, self.m.rho_sat))

    def weight(self, sq: torch.Tensor) -> torch.Tensor:
        sq = sq.to(self.m.dtype)
        t = self.m._t(sq)
        inside = sq <= self.m.tau2
        t_safe = torch.where(inside, t, torch.full_like(t, self.tk))
        w_in = self.m._w_pref * (self._lookup(self._gu_a2, t_safe) - self.m._gu_a2_tk.to(sq.device))
        return torch.where(inside, w_in.clamp_min(0.0), torch.zeros_like(w_in))

    def total_loss(self, sq: torch.Tensor, valid: Optional[torch.Tensor] = None,
                   dim: int = -1) -> torch.Tensor:
        rho = self.loss(sq)
        if valid is not None:
            rho = torch.where(valid, rho, torch.zeros_like(rho))
        return rho.sum(dim=dim)
