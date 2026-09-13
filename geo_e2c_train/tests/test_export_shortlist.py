"""Pure geometry of the coarse→fine shortlist: which dense-250 tiles a cell covers."""
import numpy as np

from geo_e2c_train.export_shortlist import dense_within_cell, gather_dense

_COS = 0.657


def test_dense_within_cell_counts_4x4_block():
    # dense centres on a 250 m grid; a 1000 m cell centred on one of them covers a 4x4-ish block.
    step = 250.0 / _COS
    xs = np.arange(-4, 5) * step                         # 9x9 dense grid around origin
    gx, gy = np.meshgrid(xs, xs)
    dcx, dcy = gx.ravel(), gy.ravel()
    reach = (1000.0 / 2.0) / _COS                        # cell half in grid units
    m = dense_within_cell(0.0, 0.0, dcx, dcy, reach)
    # |dx| <= 500 true m → offsets {-500,-250,0,250,500} = 5 per axis → 5x5 = 25 (inclusive edges)
    assert int(m.sum()) == 25


def test_gather_dense_unions_same_city_only():
    step = 250.0 / _COS
    # two cells (same city) 1000 m apart on x; one far cell in another city
    cell_xy = np.array([[0.0, 0.0], [1000.0 / _COS, 0.0], [0.0, 0.0]])
    cell_city = np.array(["k", "k", "u"])
    xs = np.arange(-2, 7) * step
    gx, gy = np.meshgrid(xs, np.arange(-2, 3) * step)
    dcx, dcy = gx.ravel(), gy.ravel()
    dense_xy = np.stack([dcx, dcy], 1)
    dense_city = np.array(["k"] * len(dcx))
    reach = (1000.0 / 2.0) / _COS
    only_k_cells = gather_dense([0, 1], cell_xy, dense_xy, dense_city, cell_city, reach)
    # union of two overlapping 5x5 blocks shifted by 4 steps → distinct x offsets span → 5x5 ∪ 5x5
    assert len(only_k_cells) > 25 and len(only_k_cells) < 50
    # a cell whose city has no dense tiles contributes nothing
    none_from_u = gather_dense([2], cell_xy, dense_xy, dense_city, cell_city, reach)
    assert none_from_u == []                              # cell city 'u' but dense are all 'k'
