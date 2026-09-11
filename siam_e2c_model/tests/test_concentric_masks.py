"""Concentric central-crop masks: centred, nested, monotone, apex 250, deterministic.

The chosen deterministic rule is ``side = round((s/tile_size_m) * axis)`` central block. For a
60×60 grid + levels (1000,840,710,600,500,420,350,300,250) m the expected sides are
(60,50,43,36,30,25,21,18,15). This can differ by one token from a naive token-centre threshold
(e.g. 710 m → 43 here vs 42 by the strict centre test) — we assert the ACTUAL round-rule result.
"""
import numpy as np
import pytest

from siam_e2c_model.concentric_core import concentric_token_masks

FULL = (1000.0, 840.0, 710.0, 600.0, 500.0, 420.0, 350.0, 300.0, 250.0)


def _sides(H, W, sizes, tile=1000.0):
    specs = concentric_token_masks(H, W, sizes, tile)
    return specs


def test_60x60_mapping_matches_documented_round_rule():
    specs = _sides(60, 60, FULL)
    got = [(s["side_h"], s["side_w"]) for s in specs]
    exp = [(60, 60), (50, 50), (43, 43), (36, 36), (30, 30), (25, 25), (21, 21), (18, 18), (15, 15)]
    assert got == exp


def test_full_tile_covers_all_tokens():
    specs = _sides(60, 60, FULL)
    assert specs[0]["count"] == 60 * 60
    assert bool(specs[0]["mask"].all())


def test_apex_is_250_and_smallest():
    specs = _sides(60, 60, FULL)
    assert specs[-1]["size_m"] == 250.0
    assert specs[-1]["count"] == min(s["count"] for s in specs)


@pytest.mark.parametrize("H,W", [(60, 60), (37, 37), (24, 30), (15, 15), (16, 16)])
def test_centred_nested_monotone_nonempty_deterministic(H, W):
    specs = concentric_token_masks(H, W, FULL, 1000.0)
    counts = [s["count"] for s in specs]
    # monotone non-increasing as physical level shrinks
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1))
    # non-empty
    assert all(c >= 1 for c in counts)
    masks = [s["mask"].reshape(H, W) for s in specs]
    # full level covers everything
    assert masks[0].all()
    for i in range(len(masks) - 1):
        a, b = masks[i], masks[i + 1]
        # nested: smaller ⊆ larger
        assert bool((b & ~a).sum() == 0)
        # centred: symmetric row/col spans (same margin each side, ±1 for odd/even parity)
        ba, bb = specs[i]["bounds"], specs[i + 1]["bounds"]
        h0a, h1a, w0a, w1a = ba
        # margins on opposite sides differ by at most 1 (deterministic floor centring)
        assert abs((h0a) - (H - h1a)) <= 1 and abs((w0a) - (W - w1a)) <= 1
    # deterministic: rebuild → identical
    again = concentric_token_masks(H, W, FULL, 1000.0)
    for s1, s2 in zip(specs, again):
        assert np.array_equal(s1["mask"], s2["mask"]) and s1["bounds"] == s2["bounds"]


def test_center_does_not_drift_by_a_whole_token():
    # the block centre stays within half a token of the grid centre at every level
    specs = concentric_token_masks(60, 60, FULL, 1000.0)
    for s in specs:
        h0, h1, w0, w1 = s["bounds"]
        cy, cx = (h0 + h1 - 1) / 2.0, (w0 + w1 - 1) / 2.0
        assert abs(cy - 29.5) <= 0.5 and abs(cx - 29.5) <= 0.5
