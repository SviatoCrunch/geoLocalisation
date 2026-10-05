"""Shared synthetic-data helpers for the magsacpp_torch tests (CPU float64)."""
from __future__ import annotations

import numpy as np
import torch

DT = torch.float64


def random_homography(rng: np.random.Generator, perspective: float = 2e-3) -> torch.Tensor:
    """A well-conditioned random homography near identity (translation+small rot/scale+perspective)."""
    ang = rng.uniform(-0.3, 0.3)
    s = rng.uniform(0.8, 1.25)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) * s
    t = rng.uniform(-20, 20, size=2)
    H = np.eye(3)
    H[:2, :2] = R
    H[:2, 2] = t
    H[2, 0] = rng.uniform(-perspective, perspective)
    H[2, 1] = rng.uniform(-perspective, perspective)
    return torch.tensor(H / H[2, 2], dtype=DT)


def apply_h(H: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(pts.shape[0], 1, dtype=pts.dtype)
    homog = torch.cat([pts, ones], 1) @ H.t()
    return homog[:, :2] / homog[:, 2:3]


def make_correspondences(rng, n_total: int, outlier_frac: float, noise: float = 0.0,
                         coord_range: float = 100.0):
    """Return (p1 (N,2), p2 (N,2), H, inlier_mask) with ``outlier_frac`` random-dest outliers."""
    H = random_homography(rng)
    p1 = torch.tensor(rng.uniform(0, coord_range, size=(n_total, 2)), dtype=DT)
    p2 = apply_h(H, p1)
    if noise > 0:
        p2 = p2 + torch.tensor(rng.normal(0, noise, size=p2.shape), dtype=DT)
    n_out = int(round(outlier_frac * n_total))
    inl = torch.ones(n_total, dtype=torch.bool)
    if n_out > 0:
        idx = rng.choice(n_total, size=n_out, replace=False)
        p2[idx] = torch.tensor(rng.uniform(-coord_range, 2 * coord_range, size=(n_out, 2)), dtype=DT)
        inl[idx] = False
    return p1, p2, H, inl


def gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g
