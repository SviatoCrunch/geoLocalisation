import pytest

from geo_split_no_overlap.positive_selection import create_positive_selector
from geo_split_no_overlap.positive_selection.strategies import _common as C
from ._ps_synth import gallery, tile, point

THIRD = 1.0 / 3.0


def _sel(threshold):
    return create_positive_selector({"strategy": "tile_iou_1000",
                                     "params": {"threshold": threshold}})


def _score(threshold, dx):
    g = gallery([tile("t0", 0.0, 0.0, size=1000.0)])
    matches = _sel(threshold).select(point("p", dx, 0.0), g)
    return matches


def test_identical_squares_iou_1_positive_at_threshold_1():
    m = _score(1.0, 0.0)
    assert len(m) == 1 and m[0].tile_id == "t0"
    assert m[0].score == pytest.approx(1.0)
    assert m[0].reason == "iou_full_tile_1000"


def test_disjoint_squares_iou_zero():
    a = C.centered_square(0.0, 0.0, 1000.0)
    b = C.centered_square(5000.0, 0.0, 1000.0)
    assert C.compute_iou(a, b) == 0.0
    assert _score(0.01, 5000.0) == []          # not even a candidate / not positive


def test_touching_squares_iou_zero():
    a = C.centered_square(0.0, 0.0, 1000.0)
    b = C.centered_square(1000.0, 0.0, 1000.0)   # share an edge only
    assert C.compute_iou(a, b) == 0.0
    assert _score(0.01, 1000.0) == []


def test_offset_500_iou_is_one_third():
    m = _score(0.1, 500.0)
    assert len(m) == 1
    assert m[0].score == pytest.approx(THIRD)


def test_value_exactly_at_threshold_is_positive():
    m = _score(THIRD, 500.0)                     # iou == threshold -> positive (>=)
    assert len(m) == 1


def test_value_just_below_threshold_excluded():
    m = _score(THIRD + 1e-6, 500.0)
    assert m == []


def test_wrong_size_tile_rejected_not_rescaled():
    g = gallery([tile("t0", 0.0, 0.0, size=800.0)])   # not a 1000 m tile
    with pytest.raises(ValueError):
        _sel(0.5).select(point("p", 0.0, 0.0), g)


def test_iou_computes_full_tile_not_center_crop():
    # at offset 300 the FULL-tile IoU is (700*1000)/(2e6-7e5)=0.5384..., a value that a
    # centre-crop rule could not produce -> guards against accidental cropping.
    m = _score(0.1, 300.0)
    assert m[0].score == pytest.approx(700_000.0 / 1_300_000.0)
