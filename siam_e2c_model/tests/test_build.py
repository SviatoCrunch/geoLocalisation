import pytest
import torch

from ._synth import build, grids, tokens

ARMS = ["supervlad", "vlad", "residual"]


@pytest.mark.parametrize("agg", ARMS)
def test_build_V_shapes(agg):
    m, _, _ = build(agg, k=6, d=64, scales=(4, 2, 1), d_out=32)
    V = m.build_V(grids(m=3, g=6, d=64))
    assert set(V) == {4, 2, 1}
    assert tuple(V[4].shape) == (3, 16, 32)
    assert tuple(V[2].shape) == (3, 4, 32)
    assert tuple(V[1].shape) == (3, 1, 32)


@pytest.mark.parametrize("agg", ARMS)
def test_encode_and_score(agg):
    m, _, _ = build(agg, d_out=32)
    q = m.encode_query(tokens(n=20, d=64))
    assert tuple(q.shape) == (32,)
    V = m.build_V(grids(m=3, g=6, d=64))
    S = m.score(torch.stack([q, q]), V)
    assert tuple(S.shape) == (2, 3)


@pytest.mark.parametrize("agg", ARMS)
def test_trainable_params_and_scales(agg):
    m, _, _ = build(agg, scales=(8, 4, 2, 1))
    assert m.scales_cells == (8, 4, 2, 1)
    assert len(m.trainable_parameters()) > 0
    assert sum(p.numel() for p in m.trainable_parameters()) > 0


def test_resolved_config_records_agg_and_size():
    m, _, _ = build("vlad", k=6, scales=(4, 2, 1))
    rc = m.resolved_config()
    assert rc["agg"] == "vlad" and rc["k"] == 6
    assert rc["aggregation"]["classic_vlad"] is True
    assert rc["scales_cells"] == [4, 2, 1]


def test_gradient_flows_through_trainable_tail():
    m, _, _ = build("supervlad", d_out=16)
    V = m.build_V(grids(m=4, g=6, d=64))
    q = torch.stack([m.encode_query(tokens(n=15, d=64, seed=s)) for s in range(4)])
    S = m.score(q, V)
    S.sum().backward()
    assert any(p.grad is not None for p in m.trainable_parameters())
