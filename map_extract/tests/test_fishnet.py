"""Fishnet grid math + torchgeo-free BoundingBox (stdlib only; shapely-gated AOI)."""
from collections import namedtuple

import pytest

from map_extract.fishnet import BoundingBox, grid_bboxes, grid_centers

Bounds = namedtuple("Bounds", ["left", "right", "bottom", "top"])


def test_boundingbox_fields_and_order():
    b = BoundingBox(1.0, 2.0, 3.0, 4.0, 0, 1)
    assert (b.minx, b.maxx, b.miny, b.maxy, b.mint, b.maxt) == (1.0, 2.0, 3.0, 4.0, 0, 1)


def test_grid_centers_exact_stride_and_count():
    # 1000-wide bounds, tile 200, stride 200, no margin -> centres at 100,300,...,900
    b = Bounds(0.0, 1000.0, 0.0, 1000.0)
    centers = grid_centers(b, tile_size_m=200.0, stride_m=200.0)
    xs = sorted({cx for cx, _ in centers})
    assert xs == [100.0, 300.0, 500.0, 700.0, 900.0]
    assert len(centers) == 25                                    # 5x5


def test_grid_centers_overlap_stride_denser():
    b = Bounds(0.0, 1000.0, 0.0, 1000.0)
    dense = grid_centers(b, tile_size_m=200.0, stride_m=100.0)
    sparse = grid_centers(b, tile_size_m=200.0, stride_m=200.0)
    assert len(dense) > len(sparse)                              # overlap => more tiles


def test_grid_centers_safe_margin_insets():
    b = Bounds(0.0, 1000.0, 0.0, 1000.0)
    centers = grid_centers(b, tile_size_m=200.0, stride_m=200.0, safe_margin_m=200.0)
    xs = sorted({cx for cx, _ in centers})
    assert min(xs) >= 300.0 and max(xs) <= 700.0                 # inset by margin+half


def test_grid_bboxes_are_boundingboxes_centred_on_grid():
    b = Bounds(0.0, 1000.0, 0.0, 1000.0)
    boxes = grid_bboxes(b, tile_size_m=200.0, stride_m=200.0)
    assert all(isinstance(x, BoundingBox) for x in boxes)
    first = boxes[0]
    assert (first.maxx - first.minx) == pytest.approx(200.0)     # tile side
    assert ((first.minx + first.maxx) / 2) == pytest.approx(100.0)


def test_grid_bboxes_aoi_filter_drops_outside_tiles():
    pytest.importorskip("shapely")
    from shapely.geometry import box as shapely_box
    b = Bounds(0.0, 1000.0, 0.0, 1000.0)
    aoi = shapely_box(0.0, 0.0, 400.0, 400.0)                    # bottom-left quadrant only
    kept = grid_bboxes(b, tile_size_m=200.0, stride_m=200.0, aoi_polygon=aoi, min_ratio=0.5)
    allb = grid_bboxes(b, tile_size_m=200.0, stride_m=200.0)
    assert 0 < len(kept) < len(allb)
    for x in kept:
        cx, cy = (x.minx + x.maxx) / 2, (x.miny + x.maxy) / 2
        assert cx <= 400.0 and cy <= 400.0
