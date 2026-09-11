"""Tiny synthetic fixtures for vlad_vocab tests."""
from __future__ import annotations

import torch


def grids(m=2, g=4, d=32, seed=1):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(m, g, g, d, generator=gen)
