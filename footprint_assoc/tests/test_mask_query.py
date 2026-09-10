import numpy as np

from footprint_assoc.mask_query import sky_filtered_query
from ._synth import FakeExtractor


def _keep_top_sky(H, W, sky_rows):
    keep = np.ones((H, W), bool)
    keep[:sky_rows] = False
    return keep


def test_drops_sky_tokens():
    ext = FakeExtractor(D=8, g=6)                     # grid 6x6
    uav = np.zeros((12, 12, 3), np.uint8)
    keep = _keep_top_sky(12, 12, 6)                   # top half sky
    q = sky_filtered_query(ext, uav, keep, cell_sky_max=0.5)
    # bottom 3 token-rows kept = 18 tokens
    assert q.patch_grid.shape == (18, 1, 8)
    assert abs(np.linalg.norm(q.global_vec) - 1.0) < 1e-6


def test_no_mask_returns_full_grid():
    ext = FakeExtractor(D=8, g=6)
    q = sky_filtered_query(ext, np.zeros((12, 12, 3), np.uint8), None)
    assert q.patch_grid.shape == (6, 6, 8)


def test_all_sky_falls_back_to_full_grid():
    ext = FakeExtractor(D=8, g=6)
    keep = np.zeros((12, 12), bool)                   # everything sky
    q = sky_filtered_query(ext, np.zeros((12, 12, 3), np.uint8), keep, cell_sky_max=0.5)
    assert q.patch_grid.shape == (6, 6, 8)            # fallback: keep all
