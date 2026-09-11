"""Concentric aggregation: shapes, finiteness, L2, the parity invariant, gradients."""
import pytest
import torch

from siam_e2c_model.config import E2cModelConfig
from siam_e2c_model.model import build_e2c_model
from ._synth import build, grids, tokens, weights

ARMS = ["supervlad", "vlad", "residual"]


@pytest.mark.parametrize("agg", ARMS)
def test_shapes_finite_l2(agg):
    m, _, _ = build(agg, k=6, d=64, d_out=32, pyramid_mode="concentric",
                    concentric_sizes_m=(1000.0, 500.0, 250.0))
    m.eval()
    V = m.build_V(grids(m=3, g=6, d=64))
    assert set(V) == {0, 1, 2}                          # level indices 0..L-1
    for li in (0, 1, 2):
        assert tuple(V[li].shape) == (3, 1, 32)         # (M, 1, d_out)
        assert torch.isfinite(V[li]).all()
        assert torch.allclose(V[li].norm(dim=2), torch.ones(3, 1), atol=1e-4)   # L2 over d_out


@pytest.mark.parametrize("agg", ARMS)
def test_parity_concentric_1000_equals_cell_1x1(agg):
    # concentric level 1000 m (full tile) == cell 1×1, given the SAME weights/heads/agg.
    aw, cent = weights(k=6, d=64, seed=11)
    common = dict(agg=agg, k=6, d_token=64, d_group=8, d_out=32, d_hidden=64)
    cell = build_e2c_model(E2cModelConfig(scales_cells=(4, 2, 1), **common),
                           assign_weight=aw, centroids=cent)
    conc = build_e2c_model(E2cModelConfig(pyramid_mode="concentric",
                                          concentric_sizes_m=(1000.0, 500.0, 250.0),
                                          tile_size_m=1000.0, **common),
                           assign_weight=aw, centroids=cent)
    # share the (randomly-initialised) trained submodules so ONLY the pyramid differs
    conc.core.agg.load_state_dict(cell.core.agg.state_dict())
    conc.core.group_proj.load_state_dict(cell.core.group_proj.state_dict())
    conc.core.map_head.load_state_dict(cell.core.map_head.state_dict())
    cell.eval(); conc.eval()

    g = grids(m=3, g=6, d=64)
    Vc = cell.build_V(g)                                 # cell keys 4,2,1
    Vk = conc.build_V(g)                                 # concentric keys 0(1000),1(500),2(250)
    assert torch.allclose(Vk[0], Vc[1], atol=1e-5)      # 1000 m  ==  1×1


@pytest.mark.parametrize("agg", ARMS)
def test_gradient_flows_and_assignment_frozen(agg):
    m, _, _ = build(agg, d_out=16, pyramid_mode="concentric")
    V = m.build_V(grids(m=4, g=6, d=64))
    q = torch.stack([m.encode_query(tokens(n=15, d=64, seed=s)) for s in range(4)])
    m.score(q, V).sum().backward()
    assert any(p.grad is not None for p in m.trainable_parameters())
    # frozen SuperVLAD assignment never trains (not in trainable set, no grad)
    assert all(not p.requires_grad for p in m.core.agg.assign.parameters())


def test_scale_gate_sized_to_levels_and_no_rho():
    m, _, _ = build("supervlad", pyramid_mode="concentric",
                    concentric_sizes_m=(1000.0, 840.0, 710.0, 600.0, 500.0, 420.0, 350.0, 300.0, 250.0))
    # scale_gate final layer outputs L = 9; concentric has no per-cell temperature (empty rho)
    assert m.core.scale_gate.net[-1].out_features == 9
    assert len(m.core.rho) == 0
