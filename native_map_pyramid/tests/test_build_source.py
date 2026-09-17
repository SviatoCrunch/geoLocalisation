"""8.4 feature-source tests: with a spied DINO extractor, verify exactly 21 views/tile, the right
source per scale, and that n=8 comes ONLY from quadrants of the 250 m grids (no 125 m DINO calls).

Run: cd geoLocalisation && python -m pytest native_map_pyramid/tests/test_build_source.py -q
"""
import numpy as np
import torch

from siam_e2c_model.vendored.stage2_core import (
    build_stage2_query_conditioned_model, tokens_to_group_sum)
from native_map_pyramid.build_cache import compute_tile_cell_sums
from native_map_pyramid.geometry import n8_flat_index, quadrant_token_slices
from native_map_pyramid import cache as C


def _agg(d_token=16, K=4, seed=0):
    torch.manual_seed(seed)
    assign = torch.nn.functional.normalize(torch.randn(K, d_token), dim=1)
    m = build_stage2_query_conditioned_model(d_token=d_token, n_groups=K, n_ghost=0,
                                             group_projection_dim=4, scales_cells=(8, 4, 2, 1),
                                             d_out=8, head_hidden=16, dropout=0.0,
                                             assign_weight=assign, freeze_assignment=True)
    return m.agg


class SpyDino:
    """Deterministic mock DINO: returns one (gh,gw,Dv) grid per crop, records call footprint."""
    def __init__(self, gh=8, gw=8, d_token=16):
        self.gh, self.gw, self.d = gh, gw, d_token
        self.n_calls = 0
        self.total_imgs = 0
        self.returned = []

    def __call__(self, imgs):
        self.n_calls += 1
        self.total_imgs += len(imgs)
        out = []
        for k in range(len(imgs)):
            g = torch.from_numpy(imgs[k].astype(np.float32)).reshape(self.gh, self.gw, -1)
            out.append(g)
        self.returned.extend(out)
        return out


def _make_read_crop(gh=8, gw=8, d_token=16):
    """read_crop_fn returning a deterministic (gh*gw, d_token)->reshaped array + a bounds log."""
    log = []

    def read(minx, miny, maxx, maxy, output_px):
        log.append((minx, miny, maxx, maxy))
        # deterministic content from the bounds so each crop's grid is distinct
        rng = np.random.RandomState(abs(hash((round(minx, 3), round(miny, 3),
                                              round(maxx, 3), round(maxy, 3)))) % (2**31))
        return rng.randn(gh * gw, d_token).astype(np.float32)

    return read, log


def test_exactly_21_dino_views_and_shapes():
    agg = _agg()
    dino = SpyDino()
    read, _ = _make_read_crop()
    S, n_views = compute_tile_cell_sums(0.0, 0.0, 1000.0, 1.5, output_px=112,
                                        read_crop_fn=read, dino_batch_fn=dino, agg=agg)
    assert n_views == 21                       # 1 + 4 + 16
    assert dino.total_imgs == 21               # n=8 added NO extra DINO calls
    assert S.shape == (C.N_CELLS_TOTAL, agg.n_groups, 16) == (85, 4, 16)


def test_source_footprints_per_scale():
    """1×1000 m + 4×500 m + 16×250 m crops read from the COG (bbox sizes, not tile subdivision)."""
    agg = _agg()
    read, log = _make_read_crop()
    ts, tile_m = 1.5, 1000.0
    compute_tile_cell_sums(0.0, 0.0, tile_m, ts, 112, read, SpyDino(), agg)
    sizes = sorted(round(maxx - minx, 4) for (minx, miny, maxx, maxy) in log)
    full = tile_m * ts
    expected = sorted([full] + [full / 2] * 4 + [full / 4] * 16)
    assert np.allclose(sizes, expected)


def test_n8_rows_are_quadrants_of_the_250m_grids():
    """Each n=8 row equals tokens_to_group_sum of the matching 30×30-style quadrant of its 250 m
    parent grid — proving n=8 carries 250 m context and uses the pinned (2i+dy,2j+dx) ordering."""
    agg = _agg()
    gh = gw = 8                                # quadrants = 4×4
    dino = SpyDino(gh=gh, gw=gw)
    read, _ = _make_read_crop(gh=gh, gw=gw)
    S, _ = compute_tile_cell_sums(0.0, 0.0, 1000.0, 1.5, 112, read, dino, agg)

    # returned grids order = plan order: [n=1](1) + [n=2](4) + [n=4](16). The 250 m grids are 5..20.
    grids_250 = dino.returned[5:21]            # flat 0..15, row-major (i,j)=divmod(flat,4)
    slices = scale_slices = C.scale_row_slices()
    S8 = S[slices[8]]                          # (64,K,Dv)
    for p, g in enumerate(grids_250):
        i, j = divmod(p, 4)
        for dy, dx, rs, cs in quadrant_token_slices(gh, gw):
            expected = tokens_to_group_sum(g[rs, cs], agg)
            got = S8[n8_flat_index(i, j, dy, dx)]
            assert torch.allclose(got, expected, atol=1e-5), (p, dy, dx)


def test_n4_and_n1_rows_match_their_full_grids():
    agg = _agg()
    dino = SpyDino()
    read, _ = _make_read_crop()
    S, _ = compute_tile_cell_sums(0.0, 0.0, 1000.0, 1.5, 112, read, dino, agg)
    sl = C.scale_row_slices()
    # n=1 (1 grid at index 0)
    assert torch.allclose(S[sl[1]][0], tokens_to_group_sum(dino.returned[0], agg), atol=1e-5)
    # n=4 (16 grids at indices 5..20)
    S4 = S[sl[4]]
    for flat, g in enumerate(dino.returned[5:21]):
        assert torch.allclose(S4[flat], tokens_to_group_sum(g, agg), atol=1e-5)
