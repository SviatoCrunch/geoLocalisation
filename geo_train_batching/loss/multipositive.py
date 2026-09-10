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


def multipositive_ce(S: torch.Tensor, pos_mask: torch.Tensor, cand_mask: torch.Tensor, *,
                     tau_loss: float = TAU_LOSS_DEFAULT) -> torch.Tensor:
    """Single-direction multi-positive CE (mean over anchors). ``cand_mask`` ⊇ ``pos_mask``."""
    z = S.float() / float(tau_loss)
    den = masked_logsumexp(z, cand_mask.to(z.device), dim=1)
    num = masked_logsumexp(z, pos_mask.to(z.device), dim=1)
    return (den - num).mean()


def symmetric_multipositive_ce(S: torch.Tensor, R_pos: torch.Tensor, R_cand: torch.Tensor, *,
                               tau_loss: float = TAU_LOSS_DEFAULT) -> torch.Tensor:
    """0.5·(query→tile + tile→query) multi-positive CE over the full B×B matrix.

    Relevance masks (often built on CPU from numpy) are moved to the score device.
    """
    R_pos = R_pos.to(S.device)
    R_cand = R_cand.to(S.device)
    row = multipositive_ce(S, R_pos, R_cand, tau_loss=tau_loss)
    col = multipositive_ce(S.t().contiguous(), R_pos.t().contiguous(),
                           R_cand.t().contiguous(), tau_loss=tau_loss)
    return 0.5 * (row + col)
