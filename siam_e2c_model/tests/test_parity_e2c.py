"""Parity: the vlad/residual arm must reproduce train_multicity_e2c.residual_* exactly.

Skipped automatically if train_multicity_e2c is not importable.
"""
import pytest
import torch

from siam_e2c_model.adapters import stage3 as S3
from ._synth import build, grids, tokens


def test_vlad_build_V_matches_e2c_residual():
    e2c = pytest.importorskip("train_multicity_e2c")
    m, _aw, centroids = build("vlad", k=6, d=64, scales=(4, 2, 1), d_out=32, seed=3)
    m.eval()                                            # deterministic (dropout off)
    g = grids(m=3, g=6, d=64)
    Cnorm = S3.normalize_centroids(centroids)
    ours = m.build_V(g)
    theirs = e2c.residual_build_V(m.core, g, Cnorm)     # same core, same centroids
    for n in m.scales_cells:
        assert torch.allclose(ours[n], theirs[n], atol=1e-5)


def test_vlad_encode_query_matches_e2c_residual():
    e2c = pytest.importorskip("train_multicity_e2c")
    m, _aw, centroids = build("vlad", k=6, d=64, scales=(4, 2, 1), d_out=32, seed=4)
    m.eval()
    t = tokens(n=25, d=64)
    Cnorm = S3.normalize_centroids(centroids)
    ours = m.encode_query(t)
    theirs = e2c.residual_encode_query(m.core, t, Cnorm, eps=m.core.eps)
    assert torch.allclose(ours, theirs, atol=1e-5)


def test_supervlad_delegates_to_core():
    m, _aw, _c = build("supervlad", d_out=32, seed=5)
    m.eval()
    g = grids(m=3, g=6, d=64)
    ours = m.build_V(g)
    core = m.core.build_V(g)
    for n in m.scales_cells:
        assert torch.equal(ours[n], core[n])
