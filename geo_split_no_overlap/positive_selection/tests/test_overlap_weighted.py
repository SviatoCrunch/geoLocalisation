"""overlap_weighted: intersection-ratio weights (Game4Loc style) over a 1000 m query."""
import pytest

pytest.importorskip("shapely")

from geo_split_no_overlap.positive_selection.models import GalleryIndex, GalleryTile, GeoPoint
from geo_split_no_overlap.positive_selection.registry import create_positive_selector
from geo_split_no_overlap.positive_selection.strategies.overlap_weighted import OverlapWeightedSelector

_COS = 0.657  # ~cos(48.9°); grid side = true_m / cos


def _gallery(tiles):
    return GalleryIndex(tiles, crs="EPSG:3857", area_units="m2", tile_size_fallback=1000.0)


def _tile(tid, x, y, lat=48.9):
    return GalleryTile(tile_id=tid, city="k", center_x=x, center_y=y, lat=lat, lon=37.5,
                       size_m=1000.0)


def test_exact_overlap_ratio_half():
    # tile offset by 500 true m in x -> grid 500/COS; a 1000 m query overlaps exactly half.
    off = 500.0 / _COS
    g = _gallery([_tile("k:0", 0.0, 0.0)])
    sel = OverlapWeightedSelector.from_params({"min_overlap": 0.0})
    p = GeoPoint(point_id="q", city="k", lat=48.9, lon=37.5, x=off, y=0.0)
    ms = sel.select(p, g)
    assert len(ms) == 1
    assert ms[0].reason == "overlap_ratio"
    assert ms[0].score == pytest.approx(0.5, abs=1e-3)          # half the query area covered


def test_checkerboard_ratios_sum_to_one():
    # 2x2 non-overlapping 1000 m cells; a query at the shared corner touches all 4, ratios sum to 1.
    s = 1000.0 / _COS                                           # cell pitch in grid units
    g = _gallery([_tile("k:0", 0.0, 0.0), _tile("k:1", s, 0.0),
                  _tile("k:2", 0.0, s), _tile("k:3", s, s)])
    sel = OverlapWeightedSelector.from_params({"min_overlap": 0.0})
    p = GeoPoint(point_id="q", city="k", lat=48.9, lon=37.5, x=s / 2, y=s / 2)  # centre of the 2x2
    ms = sel.select(p, g)
    assert len(ms) == 4
    assert sum(m.score for m in ms) == pytest.approx(1.0, abs=5e-3)   # _COS is approximate
    assert all(m.score == pytest.approx(0.25, abs=2e-3) for m in ms)


def test_min_overlap_drops_slivers_and_registered():
    off = 900.0 / _COS                                          # 10% overlap
    g = _gallery([_tile("k:0", 0.0, 0.0)])
    p = GeoPoint(point_id="q", city="k", lat=48.9, lon=37.5, x=off, y=0.0)
    assert OverlapWeightedSelector.from_params({"min_overlap": 0.2}).select(p, g) == []
    assert len(OverlapWeightedSelector.from_params({"min_overlap": 0.05}).select(p, g)) == 1
    sel = create_positive_selector({"strategy": "overlap_weighted", "params": {}})
    assert sel.name == "overlap_weighted"
    with pytest.raises(ValueError):
        OverlapWeightedSelector.from_params({"min_overlap": 1.5})
