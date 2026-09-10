"""Full-gallery multi-positive NLL (dense + exact streaming) and soft-relevance CE.

Ported verbatim (parity-tested) from
``siam_model_stage4_full_gallery.loss.full_gallery``. Model-agnostic: the streaming
form takes a ``score_fn(rows) -> (B, len(rows))`` callback (grad flows through it), so
the caller's model/scorer stays in a different package.

For query ``q`` with positives ``P_pos(q)`` and candidates ``C(q)=P_pos∪P_safe``:

    z(q,t) = score(q,t)/tau_loss
    L(q)   = logsumexp_{t∈C(q)} z − logsumexp_{p∈P_pos(q)} z
"""
from __future__ import annotations

import torch

from .multipositive import masked_logsumexp, TAU_LOSS_DEFAULT

_NEG_INF = float("-inf")


def _preflight(pos_mask: torch.Tensor, cand_mask: torch.Tensor) -> None:
    """Fail-closed: a query with no positive / no candidate is a geometry error, not a skip."""
    n_pos = pos_mask.sum(dim=1)
    n_cand = cand_mask.sum(dim=1)
    if (n_pos < 1).any():
        bad = torch.nonzero(n_pos < 1).flatten().tolist()
        raise ValueError(f"queries with no positive (rows {bad}) — geometry/preflight error")
    if (n_cand < 1).any():
        bad = torch.nonzero(n_cand < 1).flatten().tolist()
        raise ValueError(f"queries with no candidate (rows {bad}) — geometry/preflight error")
    if not torch.all(cand_mask | ~pos_mask):
        raise ValueError("cand_mask is not a superset of pos_mask (C must equal P_pos ∪ P_safe)")


def validate_masks(pos_mask: torch.Tensor, cand_mask: torch.Tensor, n_gallery: int) -> None:
    """Fail-closed mask contract: bool dtype, equal shape, width == n_gallery, ≥1 positive/query,
    pos ⊆ cand, ignore excluded (cand == P_pos ∪ P_safe)."""
    if pos_mask.dtype != torch.bool or cand_mask.dtype != torch.bool:
        raise TypeError("pos_mask/cand_mask must be bool")
    if tuple(pos_mask.shape) != tuple(cand_mask.shape):
        raise ValueError(f"pos_mask shape {tuple(pos_mask.shape)} != cand_mask {tuple(cand_mask.shape)}")
    if pos_mask.shape[1] != int(n_gallery):
        raise ValueError(f"mask width {pos_mask.shape[1]} != n_gallery {n_gallery} — the denominator "
                         "must span the whole global gallery (no truncation).")
    _preflight(pos_mask, cand_mask)


def dense_full_gallery_nll(scores: torch.Tensor, pos_mask: torch.Tensor, cand_mask: torch.Tensor, *,
                           tau_loss: float = TAU_LOSS_DEFAULT, check: bool = True) -> dict:
    """Dense reference. ``scores`` (B,M); ``pos_mask``/``cand_mask`` (B,M) bool, cand ⊇ pos."""
    if pos_mask.dtype != torch.bool or cand_mask.dtype != torch.bool:
        raise TypeError("pos_mask/cand_mask must be bool")
    z = scores.float() / float(tau_loss)
    if check:
        _preflight(pos_mask, cand_mask)
    den = masked_logsumexp(z, cand_mask, dim=1)
    num = masked_logsumexp(z, pos_mask, dim=1)
    per_q = den - num
    return {"loss": per_q.mean(), "per_query": per_q, "den_lse": den, "pos_lse": num}


def streaming_full_gallery_nll(score_fn, n_gallery: int, pos_mask: torch.Tensor,
                               cand_mask: torch.Tensor, *, tau_loss: float = TAU_LOSS_DEFAULT,
                               chunk_size: int = 64, check: bool = True, device=None) -> dict:
    """Exact streaming NLL. ``score_fn(rows) -> (B, len(rows))`` (grad flows). Accumulates one
    num/den per query via ``torch.logaddexp`` in fp32; equals :func:`dense_full_gallery_nll`.
    ``chunk_size`` is a pure memory knob; the denominator is never truncated."""
    if check:
        _preflight(pos_mask, cand_mask)
    B = pos_mask.shape[0]
    dev = device if device is not None else pos_mask.device
    den_lse = torch.full((B,), _NEG_INF, dtype=torch.float32, device=dev)
    pos_lse = torch.full((B,), _NEG_INF, dtype=torch.float32, device=dev)
    cs = int(chunk_size)
    if cs < 1:
        raise ValueError("chunk_size must be >= 1")
    for c0 in range(0, n_gallery, cs):
        c1 = min(c0 + cs, n_gallery)
        rows = torch.arange(c0, c1, device=dev)
        s = score_fn(rows).float() / float(tau_loss)
        chunk_den = masked_logsumexp(s, cand_mask[:, c0:c1], dim=1)
        chunk_pos = masked_logsumexp(s, pos_mask[:, c0:c1], dim=1)
        den_lse = torch.logaddexp(den_lse, chunk_den)
        pos_lse = torch.logaddexp(pos_lse, chunk_pos)
    per_q = den_lse - pos_lse
    return {"loss": per_q.mean(), "per_query": per_q, "den_lse": den_lse, "pos_lse": pos_lse}


def streaming_soft_relevance_ce(score_fn, n_gallery: int, cand_mask: torch.Tensor,
                                soft_target: torch.Tensor, *, tau_loss: float = TAU_LOSS_DEFAULT,
                                chunk_size: int = 64, device=None) -> dict:
    """Exact streaming soft-relevance CE: ``L(q) = logZ_cand − Σ_t y(q,t)·z(q,t)`` (y sums to 1 over P)."""
    B = cand_mask.shape[0]
    dev = device if device is not None else cand_mask.device
    y = soft_target.float().to(dev)
    den_lse = torch.full((B,), _NEG_INF, dtype=torch.float32, device=dev)
    wnum = torch.zeros(B, dtype=torch.float32, device=dev)
    cs = int(chunk_size)
    for c0 in range(0, n_gallery, cs):
        c1 = min(c0 + cs, n_gallery)
        rows = torch.arange(c0, c1, device=dev)
        z = score_fn(rows).float() / float(tau_loss)
        den_lse = torch.logaddexp(den_lse, masked_logsumexp(z, cand_mask[:, c0:c1], dim=1))
        wnum = wnum + (y[:, c0:c1] * z).sum(dim=1)
    per_q = den_lse - wnum
    return {"loss": per_q.mean(), "per_query": per_q, "den_lse": den_lse, "weighted_num": wnum}
