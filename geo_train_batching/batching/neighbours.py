"""Neighbour cache over per-pair embeddings (cosine kNN, deterministic).

Ported verbatim from ``siam_model_stage4_full_gallery.batching.dss``. Embeddings are
passed IN as a plain array ``[P, d]`` (the caller/model produces them in eval mode) —
so this module stays model-agnostic. Refreshed every N epochs by the training loop.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass
class NeighbourCache:
    neighbours: dict             # pair_index -> list[pair_index] (nearest first, self excluded)
    epoch: int
    version_hash: str

    def of(self, pair_index: int) -> list:
        return list(self.neighbours.get(int(pair_index), []))


def build_neighbour_cache(pair_vectors, *, epoch: int, top_k: int = 32) -> NeighbourCache:
    """Cosine-similarity neighbour lists over per-pair vectors ``[P, d]`` (deterministic)."""
    X = np.asarray(pair_vectors, np.float64)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    order = np.argsort(-S, axis=1)[:, :top_k]
    neigh = {i: [int(j) for j in order[i]] for i in range(X.shape[0])}
    vh = "nbr:" + hashlib.sha256(np.ascontiguousarray(order).tobytes()).hexdigest()[:24]
    return NeighbourCache(neighbours=neigh, epoch=int(epoch), version_hash=vh)
