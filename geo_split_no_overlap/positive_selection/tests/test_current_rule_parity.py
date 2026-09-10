"""Parity: the current_rule adapter must reproduce the production box-rule semantics.

The reference is computed independently here from the documented formula
``|dx|,|dy| <= (window_size_m + query_size_m)/2`` in EPSG:3857 (make_multicity_split's
own code path has a latent NameError and never ran, so the docstring formula is the
contract). If current_rule ever drifts from this, the test fails.
"""
import numpy as np

from geo_split_no_overlap.positive_selection import (
    create_positive_selector, materialize_positive_sets)
from ._ps_synth import gallery, tile, point, latlon_to_xy


def _reference_positives(points, tiles, query_size_m, fallback=1000.0):
    """Brute-force the documented rule directly from geometry (same-city)."""
    ref = {}
    for p in points:
        pos = set()
        for t in tiles:
            side = t.size_m if (t.size_m == t.size_m and t.size_m > 0) else fallback
            half = (side + query_size_m) / 2.0
            if t.city == p.city and abs(t.center_x - p.x) <= half and abs(t.center_y - p.y) <= half:
                pos.add(t.tile_id)
        ref[p.point_id] = frozenset(pos)
    return ref


def _grid():
    x0, y0 = latlon_to_xy(48.5, 37.8)
    tiles = [tile(f"t{i}_{j}", x0 + i * 250.0, y0 + j * 250.0, size=1000.0)  # stride-250 overlap
             for i in range(6) for j in range(6)]
    pts = [point(f"p{k}", x0 + (k * 137.0) % 1500.0, y0 + (k * 91.0) % 1500.0)
           for k in range(20)]
    return gallery(tiles), tiles, pts


def test_current_rule_matches_reference_query0():
    g, tiles, pts = _grid()
    mps = materialize_positive_sets(pts, g, create_positive_selector(
        {"strategy": "current_rule", "params": {"query_size_m": 0.0}}))
    ref = _reference_positives(pts, tiles, 0.0)
    got = {pid: mps.point_to_tile_ids.get(pid, frozenset()) for pid in ref}
    assert got == ref
    assert any(len(v) > 0 for v in ref.values())     # the fixture actually exercises the rule


def test_current_rule_matches_reference_query500():
    g, tiles, pts = _grid()
    mps = materialize_positive_sets(pts, g, create_positive_selector(
        {"strategy": "current_rule", "params": {"query_size_m": 500.0}}))
    ref = _reference_positives(pts, tiles, 500.0)
    got = {pid: mps.point_to_tile_ids.get(pid, frozenset()) for pid in ref}
    assert got == ref


def test_contains_point_equals_current_rule_query0():
    # contains_point == current_rule with query_size_m=0 (documented point-in-tile rule)
    g, tiles, pts = _grid()
    a = materialize_positive_sets(pts, g, create_positive_selector({"strategy": "contains_point"}))
    b = materialize_positive_sets(pts, g, create_positive_selector(
        {"strategy": "current_rule", "params": {"query_size_m": 0.0}}))
    assert dict(a.point_to_tile_ids) == dict(b.point_to_tile_ids)
