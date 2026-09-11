"""Recast K-means centroids → the e2c dictionary blob (assign_weight + centroids).

Uses the vendored ``siam_e2c_model.vendored.stage2_core.assignment_from_centroids`` (single
source of truth for the SuperVLAD γ·L2(centroids) derivation, γ calibrated on tokens) so the
blob is exactly what ``build_e2c_model``'s loaders expect:

  * ``assign_io.load_assign_weight`` reads ``assign_weight`` (supervlad arm),
  * ``VladAggregation.build`` reads ``centroids``            (vlad/residual arm).
"""
from __future__ import annotations


def recast_centroids(centroids, calib_tokens, *, init_prob: float = 0.01, eps: float = 1e-8) -> dict:
    """(K,D) centroids + (N,D) calibration tokens → blob dict {assign_weight, centroids, ...}."""
    import torch
    from siam_e2c_model.vendored.stage2_core import assignment_from_centroids

    C = torch.as_tensor(centroids).float()
    X = torch.as_tensor(calib_tokens).float()
    blob = assignment_from_centroids(C, X, init_prob=init_prob, eps=eps)
    # assignment_from_centroids already stores normalised centroids + γ·Ĉ assign_weight.
    blob["source"] = "vlad_vocab.recast"
    blob["init_prob"] = float(init_prob)
    return blob
