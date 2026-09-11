"""Fractional concentric masks: symmetric, exactly centred, nested, apex 250, deterministic.

Mask rule: w_i = L·max(0, min((i+1)/L, 0.5+r/2) − max(i/L, 0.5−r/2)), r = s/tile_size_m.
Fixes the previous round-crop half-token shift (odd sides 43,25,21,15 on a 60×60 grid).
"""
import numpy as np
import pytest

from siam_e2c_model.concentric_core import concentric_token_masks

FULL = (1000.0, 840.0, 710.0, 600.0, 500.0, 420.0, 350.0, 300.0, 250.0)
NOMINAL_60 = [60.0, 50.4, 42.6, 36.0, 30.0, 25.2, 21.0, 18.0, 15.0]   # r·60 per level


def test_full_tile_is_all_ones():
    specs = concentric_token_masks(60, 60, FULL, 1000.0)
    assert np.allclose(specs[0]["weight"], 1.0)
    assert specs[0]["effective_side_h"] == pytest.approx(60.0)


def test_effective_sides_equal_nominal_continuous_60():
    specs = concentric_token_masks(60, 60, FULL, 1000.0)
    assert [s["effective_side_h"] for s in specs] == pytest.approx(NOMINAL_60)
    assert [s["effective_side_w"] for s in specs] == pytest.approx(NOMINAL_60)


def test_weight_sum_equals_ratio_times_axis():
    for H, W in [(60, 60), (37, 41)]:
        for s in concentric_token_masks(H, W, FULL, 1000.0):
            assert s["effective_side_h"] == pytest.approx(s["ratio"] * H)
            assert s["effective_side_w"] == pytest.approx(s["ratio"] * W)
            assert s["weight_sum"] == pytest.approx(s["ratio"] * H * s["ratio"] * W)


@pytest.mark.parametrize("H,W", [(60, 60), (37, 37), (24, 30), (15, 15), (16, 16)])
def test_center_of_mass_is_exact_grid_center(H, W):
    # regression for the half-token shift: centre error ~0 for EVERY level, even + odd grids
    for s in concentric_token_masks(H, W, FULL, 1000.0):
        assert s["center_error_y"] < 1e-9
        assert s["center_error_x"] < 1e-9


def test_masks_are_nested_and_fractional_and_monotone():
    specs = concentric_token_masks(60, 60, FULL, 1000.0)
    ws = [s["weight"] for s in specs]
    for w in ws:
        assert w.min() >= 0.0 and w.max() <= 1.0 + 1e-12
    for i in range(len(ws) - 1):
        # nested in the fractional sense: larger level >= smaller level elementwise
        assert np.all(ws[i] + 1e-12 >= ws[i + 1])
        assert ws[i].sum() >= ws[i + 1].sum()               # token weight non-increasing
    # boundary tokens carry fractional weight at a non-integer nominal side (840 m → 50.4)
    assert 0.0 < ws[1].min() or (0.0 < ws[1]).any()
    assert not np.allclose(ws[1], np.round(ws[1]))          # genuinely fractional, not a hard crop


def test_apex_is_250_smallest_and_nonempty():
    specs = concentric_token_masks(60, 60, FULL, 1000.0)
    assert specs[-1]["size_m"] == 250.0
    assert specs[-1]["weight_sum"] == min(s["weight_sum"] for s in specs)
    assert all(s["weight_sum"] > 0 for s in specs)


def test_deterministic():
    a = concentric_token_masks(37, 37, FULL, 1000.0)
    b = concentric_token_masks(37, 37, FULL, 1000.0)
    for s1, s2 in zip(a, b):
        assert np.array_equal(s1["weight"], s2["weight"]) and s1["support_bounds"] == s2["support_bounds"]
