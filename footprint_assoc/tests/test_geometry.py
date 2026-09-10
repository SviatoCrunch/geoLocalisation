import math

from footprint_assoc.geometry import (merc, merc_inv, square_bounds, overlap_metrics,
                                      footprint_bounds, square_corners_lonlat)
from footprint_assoc.config import geometric_scales


def test_merc_roundtrip():
    lon, lat = merc_inv(*merc(48.5, 37.8))
    assert abs(lat - 48.5) < 1e-6 and abs(lon - 37.8) < 1e-6


def test_identical_squares_full_overlap():
    a = square_bounds(0, 0, 1000)
    cq, ct, iou = overlap_metrics(a, a)
    assert abs(cq - 1) < 1e-9 and abs(ct - 1) < 1e-9 and abs(iou - 1) < 1e-9


def test_footprint_inside_bigger_tile():
    fp = square_bounds(0, 0, 500)      # small footprint
    tile = square_bounds(0, 0, 1000)   # big tile, concentric
    cq, ct, iou = overlap_metrics(fp, tile)
    assert abs(cq - 1) < 1e-9          # whole footprint inside tile
    assert abs(ct - 0.25) < 1e-9       # footprint covers 1/4 of the tile area
    assert abs(iou - 0.25) < 1e-9


def test_disjoint_squares_zero():
    cq, ct, iou = overlap_metrics(square_bounds(0, 0, 100), square_bounds(1000, 0, 100))
    assert cq == 0 and ct == 0 and iou == 0


def test_corners_ring_closed():
    ring = square_corners_lonlat(*merc(48.5, 37.8), 1000)
    assert len(ring) == 5 and ring[0] == ring[-1]


def test_geometric_scales_grid():
    g = geometric_scales(100, 1000, 1.21)
    assert g[0] == 100.0 and g[-1] == 1000.0
    assert all(b > a for a, b in zip(g, g[1:]))   # strictly increasing
