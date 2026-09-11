"""Scoring + diagnostics in both modes."""
import pytest
import torch

from ._synth import build, grids, tokens

FULL = (1000.0, 840.0, 710.0, 600.0, 500.0, 420.0, 350.0, 300.0, 250.0)


def _q(m, n=4):
    return torch.stack([m.encode_query(tokens(n=15, d=64, seed=s)) for s in range(n)])


@pytest.mark.parametrize("mode,kw", [("cell", {}), ("concentric", {"concentric_sizes_m": FULL})])
def test_score_shape_both_modes(mode, kw):
    m, _, _ = build("supervlad", d_out=32, pyramid_mode=mode, **kw)
    m.eval()
    V = m.build_V(grids(m=5, g=6, d=64))
    S = m.score(_q(m, 4), V)
    assert tuple(S.shape) == (4, 5)


@pytest.mark.parametrize("mode,kw,L", [("cell", {"scales": (4, 2, 1)}, 3),
                                       ("concentric", {"concentric_sizes_m": FULL}, 9)])
def test_details_shapes_and_consistency(mode, kw, L):
    m, _, _ = build("supervlad", d_out=32, pyramid_mode=mode, **kw)
    m.eval()
    V = m.build_V(grids(m=5, g=6, d=64))
    q = _q(m, 4)
    det = m.score_with_details(q, V)
    assert tuple(det["tile_scores"].shape) == (4, 5)
    assert tuple(det["level_scores"].shape) == (4, 5, L)
    assert det["pyramid_mode"] == mode
    assert tuple(det["best_level_index"].shape) == (4, 5)
    assert tuple(det["confidence_margin"].shape) == (4, 5)
    assert tuple(det["entropy"].shape) == (4, 5)
    # tile_scores from the detail path == the plain scorer (same formula)
    assert torch.allclose(det["tile_scores"], m.score(q, V), atol=1e-6)
    # best_level_value == level_values gathered at argmax of level_scores
    exp_idx = det["level_scores"].argmax(dim=2)
    assert torch.equal(det["best_level_index"], exp_idx)
    assert torch.allclose(det["best_level_value"], det["level_values"][exp_idx])


def test_concentric_best_level_value_is_physical_metres():
    m, _, _ = build("supervlad", d_out=32, pyramid_mode="concentric", concentric_sizes_m=FULL)
    m.eval()
    V = m.build_V(grids(m=5, g=6, d=64))
    det = m.score_with_details(_q(m, 3), V)
    vals = set(float(x) for x in det["best_level_value"].reshape(-1).tolist())
    assert vals.issubset(set(FULL))                     # every picked value is a real level size
    assert det["level_values"].tolist() == list(FULL)


@pytest.mark.parametrize("mode,kw", [("cell", {"scales": (4, 2, 1)}),
                                     ("concentric", {"concentric_sizes_m": FULL})])
def test_tile_chunk_matches_unchunked(mode, kw):
    m, _, _ = build("supervlad", d_out=32, pyramid_mode=mode, **kw)
    m.eval()
    V = m.build_V(grids(m=7, g=6, d=64))
    q = _q(m, 3)
    assert torch.allclose(m.score(q, V), m.score(q, V, tile_chunk=2), atol=1e-6)
    d0 = m.score_with_details(q, V)
    d1 = m.score_with_details(q, V, tile_chunk=2)
    assert torch.allclose(d0["level_scores"], d1["level_scores"], atol=1e-6)


@pytest.mark.parametrize("B,M", [(1, 1), (1, 4), (3, 1), (3, 5)])
def test_batch_and_tile_counts(B, M):
    m, _, _ = build("supervlad", d_out=16, pyramid_mode="concentric", concentric_sizes_m=FULL)
    m.eval()
    V = m.build_V(grids(m=M, g=6, d=64))
    S = m.score(_q(m, B), V)
    assert tuple(S.shape) == (B, M)
