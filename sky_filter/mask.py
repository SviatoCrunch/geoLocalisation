"""Pixel keep-mask → token-grid reduction + token dropping (DROP semantics).

Sky is DROPPED: a token cell is removed if its sky fraction exceeds ``cell_sky_max``
(0.0 ⇒ drop on any sky). ``apply_drop`` returns only the kept ground tokens (N,D).
"""
from __future__ import annotations

import numpy as np


def reduce_to_grid(keep_pixels: np.ndarray, gh: int, gw: int, cell_sky_max: float = 0.5) -> np.ndarray:
    """(H,W) pixel keep-mask → (gh*gw,) token keep-mask (row-major).

    A cell is kept iff its sky fraction <= ``cell_sky_max``. Cells map to equal pixel
    blocks via ``np.array_split`` (handles non-divisible sizes)."""
    keep = np.asarray(keep_pixels, bool)
    if keep.ndim != 2:
        raise ValueError(f"keep_pixels must be 2-D (H,W), got {keep.shape}")
    rows = np.array_split(np.arange(keep.shape[0]), gh)
    cols = np.array_split(np.arange(keep.shape[1]), gw)
    out = np.zeros(gh * gw, bool)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            block = keep[np.ix_(r, c)]
            sky_frac = 1.0 - (block.mean() if block.size else 1.0)
            out[i * gw + j] = sky_frac <= cell_sky_max
    return out


def apply_drop(tokens: np.ndarray, token_keep: np.ndarray) -> np.ndarray:
    """Drop sky tokens. ``tokens`` (H,W,D) or (N,D); ``token_keep`` (H*W,) or (N,) bool → (Nkept, D)."""
    t = np.asarray(tokens)
    flat = t.reshape(-1, t.shape[-1])
    k = np.asarray(token_keep, bool).reshape(-1)
    if k.shape[0] != flat.shape[0]:
        raise ValueError(f"token_keep {k.shape[0]} != n_tokens {flat.shape[0]}")
    return flat[k]
