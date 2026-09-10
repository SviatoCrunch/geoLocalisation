import numpy as np

from sky_filter.mask import reduce_to_grid, apply_drop
from ._synth import keep_top_sky


def test_reduce_drops_sky_cells():
    keep = keep_top_sky(H=8, W=8, sky_rows=4)          # top half sky
    tk = reduce_to_grid(keep, gh=4, gw=4, cell_sky_max=0.5).reshape(4, 4)
    assert not tk[:2].any()          # top 2 token rows dropped (all sky)
    assert tk[2:].all()              # bottom 2 token rows kept (all ground)


def test_reduce_any_sky_drops_when_zero_threshold():
    keep = keep_top_sky(H=8, W=8, sky_rows=1)          # 1 sky row of 8
    tk = reduce_to_grid(keep, gh=4, gw=4, cell_sky_max=0.0).reshape(4, 4)
    assert not tk[0].any()           # first cell row touches sky -> dropped at 0.0
    assert tk[1:].all()


def test_apply_drop_removes_sky_tokens():
    keep = keep_top_sky(H=4, W=4, sky_rows=2)
    tk = reduce_to_grid(keep, gh=4, gw=4, cell_sky_max=0.5)
    tokens = np.random.default_rng(0).standard_normal((4, 4, 5))
    kept = apply_drop(tokens, tk)
    assert kept.shape == (8, 5)      # bottom 2 rows * 4 cols kept
    assert kept.shape[0] == int(tk.sum())


def test_apply_drop_length_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        apply_drop(np.zeros((4, 4, 3)), np.ones(10, bool))
