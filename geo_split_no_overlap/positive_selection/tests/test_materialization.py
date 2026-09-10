import pytest

from geo_split_no_overlap.positive_selection import (
    create_positive_selector, materialize_positive_sets)
from ._ps_synth import gallery, tile, point


def _fixture():
    g = gallery([tile("t0", 0.0, 0.0), tile("t1", 1_000_000.0, 0.0)])
    pts = [point("b", 0.0, 0.0), point("a", 0.0, 0.0),           # both inside t0
           point("z", 5_000_000.0, 0.0)]                          # inside nothing
    return g, pts


def test_snapshot_is_immutable_and_sorted():
    g, pts = _fixture()
    sel = create_positive_selector({"strategy": "contains_point"})
    mps = materialize_positive_sets(pts, g, sel)
    assert isinstance(mps.point_to_tile_ids["a"], frozenset)
    # points_without_positives captured, not silently dropped
    assert mps.points_without_positives == ("z",)
    with pytest.raises(TypeError):
        mps.point_to_tile_ids["a"] = frozenset()                 # MappingProxy is read-only


def test_stats_are_reported():
    g, pts = _fixture()
    mps = materialize_positive_sets(pts, g, create_positive_selector({"strategy": "contains_point"}))
    assert mps.stats["n_points_with_positives"] == 2
    assert mps.stats["n_points_without_positives"] == 1
    assert mps.stats["unique_positive_tiles"] == 1


def test_fingerprint_stable_for_same_inputs():
    g, pts = _fixture()
    sel = create_positive_selector({"strategy": "contains_point"})
    a = materialize_positive_sets(pts, g, sel).fingerprint
    b = materialize_positive_sets(pts, g, sel).fingerprint
    assert a == b


def test_fingerprint_changes_with_rule_or_param():
    g, pts = _fixture()
    fp_contains = materialize_positive_sets(
        pts, g, create_positive_selector({"strategy": "contains_point"})).fingerprint
    fp_box0 = materialize_positive_sets(
        pts, g, create_positive_selector({"strategy": "current_rule",
                                          "params": {"query_size_m": 0.0}})).fingerprint
    fp_box50 = materialize_positive_sets(
        pts, g, create_positive_selector({"strategy": "current_rule",
                                          "params": {"query_size_m": 50.0}})).fingerprint
    # different strategy name -> different fingerprint (even if associations coincide)
    assert fp_contains != fp_box0
    # different parameter -> different fingerprint
    assert fp_box0 != fp_box50
