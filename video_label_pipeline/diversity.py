"""Farthest-point selection of visually diverse frames.

Given L2-comparable embeddings (DINO CLS features), pick ``k`` indices that are
as mutually different as possible — a greedy farthest-point sampling (FPS). This
is the deliberate, reproducible alternative to random selection: it spreads the
chosen frames across the visual space so the annotation set covers distinct
views rather than near-duplicates.

Embeddings are L2-normalised first, so distances behave like cosine distance
(appropriate for DINO features). The first point is chosen deterministically as
the one farthest from the centroid, making the whole selection deterministic.
"""
from __future__ import annotations

import numpy as np


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return X / norm


def farthest_point_select(embeddings, k: int) -> list[int]:
    """Return up to ``k`` indices chosen by greedy farthest-point sampling.

    Order of the returned list is the selection order (rank 1 = first picked).
    If there are ``<= k`` embeddings, all indices are returned (in order).
    """
    X = np.asarray(embeddings, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"embeddings must be 2-D [N, D], got shape {X.shape}")
    n = X.shape[0]
    if n == 0 or k <= 0:
        return []
    if k >= n:
        return list(range(n))

    Xn = _l2_normalize(X)
    centroid = Xn.mean(axis=0, keepdims=True)
    first = int(np.argmax(np.linalg.norm(Xn - centroid, axis=1)))

    selected = [first]
    min_dist = np.linalg.norm(Xn - Xn[first], axis=1)
    min_dist[first] = -1.0                         # never re-pick a chosen frame
    for _ in range(1, k):
        nxt = int(np.argmax(min_dist))             # distinct even if embeddings coincide
        selected.append(nxt)
        min_dist = np.minimum(min_dist, np.linalg.norm(Xn - Xn[nxt], axis=1))
        min_dist[selected] = -1.0
    return selected
