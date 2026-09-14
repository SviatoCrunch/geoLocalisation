"""Pure helpers of the real-map sliding-window pyramid reranker (no rasterio/DINO needed)."""
import math

from patch_rerank.map_rerank import cell_window_centres, pyramid_sizes, snap, _haversine_m
from patch_rerank.map_source import latlon_to_merc, merc_to_latlon, true_m_to_crs


def test_pyramid_sizes():
    assert pyramid_sizes(1000, 300, 100) == [1000, 900, 800, 700, 600, 500, 400, 300]
    assert pyramid_sizes(1000, 1000, 100) == [1000]


def test_snap_to_grid():
    assert snap(137.0, 100.0) == 100.0
    assert snap(151.0, 100.0) == 200.0
    assert snap(1000.0, 250.0) == 1000.0


def test_window_centres_grid_and_count():
    lat = 48.9
    cx, cy = latlon_to_merc(lat, 37.6)
    centres = cell_window_centres(cx, cy, lat, radius_m=500.0, step_m=250.0)
    step_crs = true_m_to_crs(250.0, lat)
    # radius 500 / step 250 → offsets {-2,-1,0,1,2} per axis → 5x5 = 25
    assert len(centres) == 25
    # every centre lies on the global snap grid (multiple of step_crs)
    for (px, py) in centres:
        assert abs(px / step_crs - round(px / step_crs)) < 1e-6


def test_overlapping_cells_share_snapped_centres():
    lat = 48.9
    ax, ay = latlon_to_merc(lat, 37.60)
    bx, by = latlon_to_merc(lat, 37.60 + 250.0 / (math.cos(math.radians(lat)) * 111320.0))  # ~250 m east
    A = set(cell_window_centres(ax, ay, lat, 500.0, 250.0))
    B = set(cell_window_centres(bx, by, lat, 500.0, 250.0))
    assert A & B                                            # adjacent cells → shared (de-dupable) centres


def test_haversine_roundtrip_and_merc():
    lat, lon = 48.9, 37.6
    x, y = latlon_to_merc(lat, lon)
    la, lo = merc_to_latlon(x, y)
    assert abs(la - lat) < 1e-6 and abs(lo - lon) < 1e-6
    d = _haversine_m(48.90, 37.60, 48.90, 37.60 + 0.01)
    assert 600 < d < 800                                    # ~0.01° lon at 48.9° ≈ 730 m
