import pytest

from ._synth import build, grids


@pytest.mark.parametrize("scales", [(8, 4, 2, 1), (4, 2, 1), (16, 8, 4, 2, 1), (2, 1)])
def test_pyramid_is_configurable(scales):
    m, _, _ = build("supervlad", scales=scales, d_out=16)
    m.eval()
    assert m.scales_cells == scales
    V = m.build_V(grids(m=2, g=max(scales), d=64))     # token grid must be >= the finest cell count
    assert set(V) == set(scales)
    for n in scales:
        assert tuple(V[n].shape) == (2, n * n, 16)


def test_swap_aggregation_by_config_only_same_pyramid():
    # only the agg name changes; the pyramid (and its keys) are identical
    ms, _, _ = build("supervlad", scales=(4, 2, 1), d_out=16)
    mv, _, _ = build("vlad", scales=(4, 2, 1), d_out=16)
    ms.eval(); mv.eval()
    assert ms.scales_cells == mv.scales_cells == (4, 2, 1)
    Vs = ms.build_V(grids(m=2, g=6, d=64))
    Vv = mv.build_V(grids(m=2, g=6, d=64))
    assert set(Vs) == set(Vv)
    # different aggregation -> different V values (sanity: not accidentally identical)
    import torch
    assert not torch.allclose(Vs[1], Vv[1])
