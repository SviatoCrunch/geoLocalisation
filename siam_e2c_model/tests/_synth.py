"""Synthetic fixtures for siam_e2c_model tests (small d_token, random assign/centroids)."""
from __future__ import annotations

import torch

from siam_e2c_model.config import E2cModelConfig
from siam_e2c_model.model import build_e2c_model


def weights(k=6, d=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    aw = torch.randn(k, d, generator=g)
    centroids = torch.randn(k, d, generator=g)
    return aw, centroids


def build(agg="supervlad", *, k=6, d=64, scales=(4, 2, 1), d_group=8, d_out=32,
          d_hidden=64, seed=0):
    aw, centroids = weights(k, d, seed)
    cfg = E2cModelConfig(agg=agg, k=k, d_token=d, scales_cells=scales, d_group=d_group,
                         d_out=d_out, d_hidden=d_hidden)
    return build_e2c_model(cfg, assign_weight=aw, centroids=centroids), aw, centroids


def grids(m=3, g=6, d=64, seed=1):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(m, g, g, d, generator=gen)


def tokens(n=20, d=64, seed=2):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=gen)
