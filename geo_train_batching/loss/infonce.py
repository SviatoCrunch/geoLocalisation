"""Symmetric (diagonal) InfoNCE — the Sample4Geo-style paradigm.

Self-contained re-implementation of the official symmetric InfoNCE used in
``sample4geo_finetune`` (which imports ``sample4geo.loss.InfoNCE``): a temperature-
scaled cross-entropy in both directions with the *diagonal* as the positive. Use this
when one query has exactly one positive reference per batch row (1:1 pairing), as
opposed to the geometry-driven multi-positive CE in :mod:`.multipositive`.

    logits = logit_scale · (Q_norm @ R_norm.T)          # (B, B)
    labels = arange(B)                                   # the diagonal is the positive
    L      = 0.5·(CE(logits, labels) + CE(logits.T, labels))

``logit_scale`` may be a Python float or a (learnable) scalar tensor — pass the model's
``logit_scale.exp()`` to keep its gradient. Model-agnostic: takes embeddings only.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

LOGIT_SCALE_DEFAULT = 1.0 / 0.07          # ~14.28 (CLIP init: log(1/0.07))


def symmetric_infonce(q: torch.Tensor, r: torch.Tensor, *,
                      logit_scale=LOGIT_SCALE_DEFAULT, label_smoothing: float = 0.0,
                      normalize: bool = True) -> torch.Tensor:
    """Symmetric InfoNCE over paired embeddings ``q``/``r`` (B,d), diagonal = positive.

    ``normalize=True`` L2-normalises both sides (cosine logits). Requires B >= 2.
    """
    if q.shape != r.shape or q.dim() != 2:
        raise ValueError(f"q and r must be equal 2-D shapes, got {tuple(q.shape)} / {tuple(r.shape)}")
    b = q.shape[0]
    if b < 2:
        raise ValueError("InfoNCE needs at least 2 rows (negatives come from the batch)")
    if normalize:
        q = F.normalize(q, dim=-1)
        r = F.normalize(r, dim=-1)
    logits = logit_scale * (q @ r.t())                       # (B, B)
    labels = torch.arange(b, device=logits.device)
    row = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
    col = F.cross_entropy(logits.t(), labels, label_smoothing=label_smoothing)
    return 0.5 * (row + col)
