from pathlib import Path

from geo_split_no_overlap.positive_selection import (
    available_positive_selectors, create_positive_selector, materialize_positive_sets)
from ._ps_synth import gallery, tile, point


def _grid():
    tiles = [tile(f"t{i}", i * 250.0, 0.0, size=1000.0) for i in range(8)]
    pts = [point(f"p{k}", k * 250.0 + 10.0, 0.0) for k in range(8)]
    return gallery(tiles), pts


def test_both_registered():
    names = available_positive_selectors()
    assert "tile_iou_1000" in names and "pyramid_top_iou_250" in names


def test_created_via_registry_and_swappable_by_config():
    for strat in ("tile_iou_1000", "pyramid_top_iou_250"):
        sel = create_positive_selector({"strategy": strat, "params": {"threshold": 0.5}})
        assert sel.name == strat


def test_service_has_no_special_branches_for_these_names():
    src = (Path(__file__).resolve().parents[1] / "service.py").read_text(encoding="utf-8")
    assert "tile_iou_1000" not in src
    assert "pyramid_top_iou_250" not in src


def test_materialize_returns_frozen_snapshot_for_both():
    g, pts = _grid()
    for strat in ("tile_iou_1000", "pyramid_top_iou_250"):
        sel = create_positive_selector({"strategy": strat, "params": {"threshold": 0.3}})
        mps = materialize_positive_sets(pts, g, sel)
        assert mps.strategy_name == strat
        # frozen: values are frozensets, mapping is read-only
        import pytest
        assert all(isinstance(v, frozenset) for v in mps.point_to_tile_ids.values())
        with pytest.raises(TypeError):
            mps.point_to_tile_ids["x"] = frozenset()


def test_fingerprint_differs_between_strategies_same_inputs():
    g, pts = _grid()
    a = materialize_positive_sets(pts, g, create_positive_selector(
        {"strategy": "tile_iou_1000", "params": {"threshold": 0.3}})).fingerprint
    b = materialize_positive_sets(pts, g, create_positive_selector(
        {"strategy": "pyramid_top_iou_250", "params": {"threshold": 0.3}})).fingerprint
    assert a != b


def test_fingerprint_changes_with_threshold():
    g, pts = _grid()
    a = materialize_positive_sets(pts, g, create_positive_selector(
        {"strategy": "tile_iou_1000", "params": {"threshold": 0.3}})).fingerprint
    b = materialize_positive_sets(pts, g, create_positive_selector(
        {"strategy": "tile_iou_1000", "params": {"threshold": 0.6}})).fingerprint
    assert a != b


def test_rerun_same_config_is_reproducible():
    g, pts = _grid()
    sel1 = create_positive_selector({"strategy": "pyramid_top_iou_250", "params": {"threshold": 0.3}})
    sel2 = create_positive_selector({"strategy": "pyramid_top_iou_250", "params": {"threshold": 0.3}})
    m1 = materialize_positive_sets(pts, g, sel1)
    m2 = materialize_positive_sets(pts, g, sel2)
    assert m1.fingerprint == m2.fingerprint
    assert dict(m1.point_to_tile_ids) == dict(m2.point_to_tile_ids)


def test_tile_vs_pyramid_differ_on_same_point():
    # offset 200 m: full-tile IoU 0.667 (positive at 0.5) but top IoU ~0.111 (not).
    g = gallery([tile("t0", 0.0, 0.0, size=1000.0)])
    p = point("p", 200.0, 0.0)
    tile_pos = create_positive_selector(
        {"strategy": "tile_iou_1000", "params": {"threshold": 0.5}}).select(p, g)
    pyr_pos = create_positive_selector(
        {"strategy": "pyramid_top_iou_250", "params": {"threshold": 0.5}}).select(p, g)
    assert [m.tile_id for m in tile_pos] == ["t0"]
    assert pyr_pos == []
