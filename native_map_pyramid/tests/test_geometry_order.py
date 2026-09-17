"""Geometric coverage (8.2) + n=8 cell ordering (8.3) for the native map source.

Run: cd geoLocalisation && python -m pytest native_map_pyramid/tests/test_geometry_order.py -q
"""
import pytest

from native_map_pyramid.geometry import (
    base_bounds, split_bounds, n8_flat_index, quadrant_token_slices,
    native_view_plan, N_VIEWS_PER_TILE, N_CELLS_TOTAL,
)

TOL = 1e-6


def _area(b):
    return (b.maxx - b.minx) * (b.maxy - b.miny)


def _overlap_area(a, b):
    dx = max(0.0, min(a.maxx, b.maxx) - max(a.minx, b.minx))
    dy = max(0.0, min(a.maxy, b.maxy) - max(a.miny, b.miny))
    return dx * dy


# ── 8.2 geometric coverage ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("n", [2, 4])
def test_children_exactly_cover_parent_no_gap_no_overlap(n):
    true_scale = 1.5455
    base = base_bounds(1_000_000.0, 6_500_000.0, 1000.0, true_scale)
    crops = split_bounds(base, n)
    assert len(crops) == n * n
    # total child area == parent area (no gaps, no overlaps)
    parent_area = (base[2] - base[0]) * (base[3] - base[1])
    assert abs(sum(_area(c) for c in crops) - parent_area) < TOL * parent_area
    # pairwise interiors disjoint
    for a in range(len(crops)):
        for b in range(a + 1, len(crops)):
            assert _overlap_area(crops[a], crops[b]) < TOL * parent_area
    # union bbox == parent bbox
    assert abs(min(c.minx for c in crops) - base[0]) < TOL
    assert abs(min(c.miny for c in crops) - base[1]) < TOL
    assert abs(max(c.maxx for c in crops) - base[2]) < TOL
    assert abs(max(c.maxy for c in crops) - base[3]) < TOL


def test_child_centers_are_correct():
    base = base_bounds(0.0, 0.0, 1000.0, 1.0)      # ±500 CRS
    crops = {(c.i, c.j): c for c in split_bounds(base, 2)}
    # i=0 is NORTH (top, +y); j=0 is WEST (left, -x). Cell half-side 250.
    assert crops[(0, 0)].center == pytest.approx((-250.0, 250.0))   # NW
    assert crops[(0, 1)].center == pytest.approx((250.0, 250.0))    # NE
    assert crops[(1, 0)].center == pytest.approx((-250.0, -250.0))  # SW
    assert crops[(1, 1)].center == pytest.approx((250.0, -250.0))   # SE


def test_flat_index_is_row_major():
    crops = split_bounds(base_bounds(0.0, 0.0, 1000.0, 1.0), 4)
    for k, c in enumerate(crops):
        assert c.flat == k == c.i * 4 + c.j


def test_view_plan_counts():
    plan = native_view_plan(0.0, 0.0, 1000.0, 1.4)
    assert len(plan[1000.0]) == 1 and len(plan[500.0]) == 4 and len(plan[250.0]) == 16
    assert sum(len(v) for v in plan.values()) == N_VIEWS_PER_TILE == 21
    assert N_CELLS_TOTAL == 85


# ── 8.3 n=8 ordering (parent quadrant → global 8×8 flat, row-major) ──────────────────────
def test_n8_mapping_exact():
    seen = {}
    for i in range(4):
        for j in range(4):
            for dy in (0, 1):
                for dx in (0, 1):
                    flat = n8_flat_index(i, j, dy, dx)
                    gr, gc = 2 * i + dy, 2 * j + dx
                    assert flat == gr * 8 + gc
                    seen[flat] = (i, j, dy, dx)
    assert sorted(seen) == list(range(64))          # bijection onto 0..63


def test_n8_mapping_breaks_under_transpose_and_flips():
    # A correct marker grid: cell flat index k holds value k. Any transpose/flip must differ.
    correct = {n8_flat_index(i, j, dy, dx): (2 * i + dy) * 8 + (2 * j + dx)
               for i in range(4) for j in range(4) for dy in (0, 1) for dx in (0, 1)}
    transpose = {k: (v % 8) * 8 + (v // 8) for k, v in correct.items()}
    vflip = {k: (7 - v // 8) * 8 + (v % 8) for k, v in correct.items()}
    hflip = {k: (v // 8) * 8 + (7 - v % 8) for k, v in correct.items()}
    assert transpose != correct and vflip != correct and hflip != correct


def test_quadrant_token_slices_60x60():
    quads = quadrant_token_slices(60, 60)
    assert len(quads) == 4
    covered = set()
    for dy, dx, rs, cs in quads:
        assert rs.stop - rs.start == 30 and cs.stop - cs.start == 30
        for r in range(rs.start, rs.stop):
            for c in range(cs.start, cs.stop):
                covered.add((r, c))
    assert len(covered) == 60 * 60          # exact, non-overlapping cover
    # dy=0 is north (top rows), dx=0 is west (left cols)
    assert quads[0][:2] == (0, 0) and quads[0][2].start == 0 and quads[0][3].start == 0


def test_quadrant_requires_even_dims():
    with pytest.raises(ValueError):
        quadrant_token_slices(61, 60)
