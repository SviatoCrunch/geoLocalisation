"""Multi-positive contrastive losses (batch B×B and single-direction).

Model-agnostic: everything here operates on a score matrix ``S`` (or logits) plus bool
relevance masks — no model, no gallery IO. Ported verbatim (parity-tested) from
``siam_model_stage4_full_gallery.batching.dss`` so the objective is identical to the
proven e2c/DSS training loss.

For anchor ``a`` with positives ``P(a)`` and candidates ``C(a) = P(a) ∪ safe(a)``:

    z(a,b) = S[a,b] / tau_loss
    L(a)   = logsumexp_{b∈C(a)} z(a,b)  −  logsumexp_{p∈P(a)} z(a,p)

``symmetric_multipositive_ce`` averages the query→tile and tile→query directions.
"""
from __future__ import annotations

import torch

TAU_LOSS_DEFAULT = 0.10
_NEG_INF = float("-inf")


def masked_logsumexp(z: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """``logsumexp`` of ``z`` over ``dim`` restricted to ``mask`` (True = member).

    Non-members are set to ``-inf`` (contribute ``exp(-inf)=0``) — exact and
    differentiable w.r.t. the members; empty rows return ``-inf``.
    """
    neg_inf = torch.full_like(z, _NEG_INF)
    zz = torch.where(mask, z, neg_inf)
    return torch.logsumexp(zz, dim=dim)


def _weighted_num(z: torch.Tensor, W: torch.Tensor, pos_mask: torch.Tensor,
                  weight_power: float) -> torch.Tensor:
    """Soft-label numerator ``Σ_t y_t·z_t`` with ``y = w^k / Σ w^k`` over the positives of each
    anchor row (Game4Loc weighted-InfoNCE / distribution-matching CE). With one positive per
    row this equals the hard ``logsumexp`` numerator, so single-positive weighting is a no-op."""
    w = torch.where(pos_mask.to(z.device), W.to(z), torch.zeros_like(z)).clamp_min(0.0)
    if float(weight_power) != 1.0:
        w = w ** float(weight_power)
    denom = w.sum(dim=1, keepdim=True)
    if (denom <= 0).any():
        bad = torch.nonzero(denom.flatten() <= 0).flatten().tolist()
        raise ValueError(f"weighted CE: anchor rows {bad} have zero total positive weight")
    y = w / denom
    return (y * z).sum(dim=1)


def multipositive_ce(S: torch.Tensor, pos_mask: torch.Tensor, cand_mask: torch.Tensor, *,
                     tau_loss: float = TAU_LOSS_DEFAULT, weights: torch.Tensor = None,
                     weight_power: float = 1.0) -> torch.Tensor:
    """Single-direction multi-positive CE (mean over anchors). ``cand_mask`` ⊇ ``pos_mask``.

    ``weights=None`` (default) → the hard ``logsumexp_P`` numerator (unchanged). A weight matrix
    (same shape as ``S``) switches to the soft-label numerator ``Σ y_t z_t`` — the one, toggleable
    weighted mode.
    """
    z = S.float() / float(tau_loss)
    den = masked_logsumexp(z, cand_mask.to(z.device), dim=1)
    if weights is None:
        num = masked_logsumexp(z, pos_mask.to(z.device), dim=1)
    else:
        num = _weighted_num(z, weights, pos_mask, weight_power)
    return (den - num).mean()


def symmetric_multipositive_ce(S: torch.Tensor, R_pos: torch.Tensor, R_cand: torch.Tensor, *,
                               tau_loss: float = TAU_LOSS_DEFAULT, weights: torch.Tensor = None,
                               weight_power: float = 1.0) -> torch.Tensor:
    """0.5·(query→tile + tile→query) multi-positive CE over the full B×B matrix.

    Relevance masks (often built on CPU from numpy) are moved to the score device. ``weights``
    (B×B soft-target, 0 off-positives) enables the weighted numerator in both directions
    (the tile→query pass uses the transpose).
    """
    R_pos = R_pos.to(S.device)
    R_cand = R_cand.to(S.device)
    W = None if weights is None else weights.to(S.device)
    row = multipositive_ce(S, R_pos, R_cand, tau_loss=tau_loss, weights=W, weight_power=weight_power)
    col = multipositive_ce(S.t().contiguous(), R_pos.t().contiguous(), R_cand.t().contiguous(),
                           tau_loss=tau_loss,
                           weights=(None if W is None else W.t().contiguous()),
                           weight_power=weight_power)
    return 0.5 * (row + col)
