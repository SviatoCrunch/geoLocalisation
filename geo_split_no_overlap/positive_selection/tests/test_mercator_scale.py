"""Regression: IoU strategies build TRUE-metre footprints in EPSG:3857 grid coords.

The gallery stores raw Web-Mercator coordinates (inflated by 1/cos(lat)). The 250 m /
1000 m footprints of ``pyramid_top_iou_250`` / ``tile_iou_1000`` are TRUE metres, so a
box must be built with grid side ``true_m / cos(lat)``. Without this scaling, at a real
latitude (~48.5°, cos≈0.66) every footprint shrinks by cos(lat) and a stride-250
gallery's central 250 m tops stop tiling the plane -> almost no positives ("МАЛО").

The other strategy tests sit at grid origin (lat 0, cos 1) where the scaling is a
no-op; these tests exercise a real latitude, so they FAIL against the old grid-unit code.
"""
import math

import pytest

from geo_split_no_overlap.positive_selection import create_positive_selector
from geo_split_no_overlap.positive_selection.strategies import _common as C
from ._ps_synth import gallery, tile, point, latlon_to_xy

LAT = 48.5
COS = math.cos(math.radians(LAT))
THIRD = 1.0 / 3.0


def test_true_m_to_grid_scales_by_inverse_cos():
    assert C.true_m_to_grid(250.0, LAT) == pytest.approx(250.0 / COS)
    assert C.true_m_to_grid(250.0, 0.0) == pytest.approx(250.0)          # equator: no-op
    assert C.true_m_to_grid(1000.0, LAT) == pytest.approx(1000.0 / COS)


def test_centered_square_true_has_true_metre_side_in_grid_units():
    x0, y0 = latlon_to_xy(LAT, 37.8)
    box = C.centered_square_true(x0, y0, LAT, 250.0)
    minx, _, maxx, _ = box.bounds
    assert (maxx - minx) == pytest.approx(250.0 / COS)                   # ~377 grid units


def _sel(strategy, threshold):
    return create_positive_selector({"strategy": strategy, "params": {"threshold": threshold}})


def test_pyramid_true_metre_offset_gives_expected_iou_at_lat48():
    # a TRUE 125 m offset (quarter of the 250 m top) -> IoU 1/3, independent of latitude.
    x0, y0 = latlon_to_xy(LAT, 37.8)
    g = gallery([tile("pyr", x0, y0, size=1000.0)])
    p = point("p", x0 + 125.0 / COS, y0)                                 # 125 TRUE m east
    m = _sel("pyramid_top_iou_250", 0.3).select(p, g)
    assert len(m) == 1 and m[0].tile_id == "pyr"
    assert m[0].score == pytest.approx(THIRD)


def test_pyramid_tops_tile_the_plane_at_true_stride_250():
    # two tiles 250 TRUE m apart; a query at the midpoint is 125 TRUE m from each top,
    # so IoU 1/3 for BOTH -> the 250 m tops cover the gap (the whole point of the fix).
    # Old grid-unit code: step≈377 grid, midpoint≈188.6 from each -> IoU≈0.14 -> {} .
    x0, y0 = latlon_to_xy(LAT, 37.8)
    step = 250.0 / COS
    g = gallery([tile("a", x0, y0, size=1000.0), tile("b", x0 + step, y0, size=1000.0)])
    p = point("p", x0 + step / 2.0, y0)
    m = _sel("pyramid_top_iou_250", 0.3).select(p, g)
    assert {x.tile_id for x in m} == {"a", "b"}
    for x in m:
        assert x.score == pytest.approx(THIRD)


def test_tile_iou_true_metre_offset_gives_expected_iou_at_lat48():
    # a TRUE 500 m offset (half of the 1000 m tile) -> IoU 1/3.
    x0, y0 = latlon_to_xy(LAT, 37.8)
    g = gallery([tile("t", x0, y0, size=1000.0)])
    p = point("p", x0 + 500.0 / COS, y0)                                 # 500 TRUE m east
    m = _sel("tile_iou_1000", 0.3).select(p, g)
    assert len(m) == 1 and m[0].tile_id == "t"
    assert m[0].score == pytest.approx(THIRD)


def test_reach_scales_with_latitude_so_candidates_are_not_missed():
    # a tile ~230 TRUE m east overlaps (tops share area) and MUST be returned; in grid
    # units that centre is ~347 units away — beyond the old reach of ~251 -> was missed.
    x0, y0 = latlon_to_xy(LAT, 37.8)
    g = gallery([tile("t", x0 + 230.0 / COS, y0, size=1000.0)])
    m = _sel("pyramid_top_iou_250", 0.01).select(point("p", x0, y0), g)
    assert [x.tile_id for x in m] == ["t"]
