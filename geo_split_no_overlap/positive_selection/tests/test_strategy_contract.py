from geo_split_no_overlap.positive_selection import (
    PositiveSelector, create_positive_selector, materialize_positive_sets)
from geo_split_no_overlap.positive_selection.models import PositiveMatch
from ._ps_synth import gallery, tile, point


def _fixture():
    g = gallery([tile("t0", 0.0, 0.0, size=1000.0), tile("t1", 1500.0, 0.0, size=1000.0)])
    p = point("p", 300.0, 0.0)          # inside t0 (half=500); t1 only within a wide query box
    return g, p


def test_strategies_satisfy_protocol():
    for name in ("current_rule", "contains_point"):
        sel = create_positive_selector({"strategy": name})
        assert isinstance(sel, PositiveSelector)          # runtime_checkable Protocol
        assert isinstance(sel.name, str) and isinstance(sel.version, str)


def test_select_returns_positive_matches_and_is_pure():
    g, p = _fixture()
    sel = create_positive_selector({"strategy": "contains_point"})
    r1 = sel.select(p, g)
    r2 = sel.select(p, g)
    assert all(isinstance(m, PositiveMatch) for m in r1)
    assert [m.tile_id for m in r1] == [m.tile_id for m in r2]     # deterministic / pure


def test_two_strategies_differ_but_share_type():
    # point is inside t0 (contains_point -> {t0}); with a 2000 m query box current_rule
    # also reaches t1 -> {t0, t1}. Same MaterializedPositiveSets type downstream.
    g, p = _fixture()
    contains = materialize_positive_sets([p], g,
                                         create_positive_selector({"strategy": "contains_point"}))
    boxed = materialize_positive_sets([p], g, create_positive_selector(
        {"strategy": "current_rule", "params": {"query_size_m": 2000.0}}))
    assert contains.point_to_tile_ids["p"] == frozenset({"t0"})
    assert boxed.point_to_tile_ids["p"] == frozenset({"t0", "t1"})
    assert type(contains) is type(boxed)


def test_crs_is_checked_for_geometric_rules():
    import pytest
    g_bad = gallery([tile("t0", 0.0, 0.0)], crs="EPSG:4326")
    sel = create_positive_selector({"strategy": "contains_point"})
    with pytest.raises(ValueError):
        sel.select(point("p", 0.0, 0.0), g_bad)
