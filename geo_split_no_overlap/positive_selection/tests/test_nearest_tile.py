"""nearest_tile strategy: exactly one positive = nearest tile centre; max_dist guard; same-city."""
import pytest

from geo_split_no_overlap.positive_selection.models import GalleryIndex, GalleryTile, GeoPoint
from geo_split_no_overlap.positive_selection.registry import create_positive_selector
from geo_split_no_overlap.positive_selection.strategies.nearest_tile import NearestTileSelector


def _gallery(tiles):
    return GalleryIndex(tiles, crs="EPSG:3857", area_units="m2", tile_size_fallback=1000.0)


def _tile(tid, city, x, y, lat=48.9):
    return GalleryTile(tile_id=tid, city=city, center_x=x, center_y=y, lat=lat, lon=37.5,
                       size_m=1000.0)


def test_returns_single_nearest():
    g = _gallery([_tile("k:0", "k", 0.0, 0.0), _tile("k:1", "k", 1000.0, 0.0),
                  _tile("k:2", "k", 0.0, 1000.0)])
    sel = NearestTileSelector.from_params({})
    p = GeoPoint(point_id="q", city="k", lat=48.9, lon=37.5, x=100.0, y=50.0)
    ms = sel.select(p, g)
    assert len(ms) == 1 and ms[0].tile_id == "k:0" and ms[0].reason == "nearest"


def test_max_dist_guard_drops_far_query():
    g = _gallery([_tile("k:0", "k", 0.0, 0.0)])
    sel = NearestTileSelector.from_params({"max_dist_m": 100.0})
    near = GeoPoint(point_id="a", city="k", lat=48.9, lon=37.5, x=50.0, y=0.0)   # 50 grid * .657 ≈ 33 m
    far = GeoPoint(point_id="b", city="k", lat=48.9, lon=37.5, x=5000.0, y=0.0)  # ~3.3 km
    assert len(sel.select(near, g)) == 1
    assert sel.select(far, g) == []


def test_same_city_only():
    g = _gallery([_tile("k:0", "k", 0.0, 0.0), _tile("u:0", "u", 10.0, 0.0)])
    p = GeoPoint(point_id="q", city="k", lat=48.9, lon=37.5, x=7.0, y=0.0)       # u:0 (d3) < k:0 (d7)
    same = NearestTileSelector.from_params({"same_city_only": True}).select(p, g)
    cross = NearestTileSelector.from_params({"same_city_only": False}).select(p, g)
    assert [m.tile_id for m in same] == ["k:0"]                                  # own city despite dist
    assert [m.tile_id for m in cross] == ["u:0"]                                 # global nearest


def test_registered_and_rejects_unknown_param():
    sel = create_positive_selector({"strategy": "nearest_tile", "params": {}})
    assert sel.name == "nearest_tile"
    with pytest.raises(ValueError):
        NearestTileSelector.from_params({"bogus": 1})
