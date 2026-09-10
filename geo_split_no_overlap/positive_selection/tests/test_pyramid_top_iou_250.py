"""Tests for pyramid_top_iou_250.

Concentricity policy (see the strategy docstring): the model carries only base
geometry, so the 250 m top is always derived concentric with the validated 1000 m
base centre — there is no independent explicit top to accept/reject. These tests
verify the top is centred on the base and that a malformed base is rejected (never
silently corrected).
"""
import pytest

from geo_split_no_overlap.positive_selection import create_positive_selector
from geo_split_no_overlap.positive_selection.strategies import _common as C
from ._ps_synth import gallery, tile, point

THIRD = 1.0 / 3.0


def _sel(threshold):
    return create_positive_selector({"strategy": "pyramid_top_iou_250",
                                     "params": {"threshold": threshold}})


def _run(threshold, dx, base_size=1000.0):
    g = gallery([tile("pyr0", 0.0, 0.0, size=base_size)])
    return _sel(threshold).select(point("p", dx, 0.0), g)


def test_concentric_top_and_query_iou_1_returns_pyramid_id():
    m = _run(1.0, 0.0)
    assert len(m) == 1
    assert m[0].tile_id == "pyr0"                # whole-pyramid id, not a synthetic top id
    assert m[0].score == pytest.approx(1.0)
    assert m[0].reason == "iou_pyramid_top_250"


def test_disjoint_tops_iou_zero():
    a = C.centered_square(0.0, 0.0, 250.0)
    b = C.centered_square(2000.0, 0.0, 250.0)
    assert C.compute_iou(a, b) == 0.0
    assert _run(0.01, 2000.0) == []


def test_touching_tops_iou_zero():
    a = C.centered_square(0.0, 0.0, 250.0)
    b = C.centered_square(250.0, 0.0, 250.0)
    assert C.compute_iou(a, b) == 0.0
    assert _run(0.01, 250.0) == []


def test_offset_125_iou_is_one_third():
    m = _run(0.1, 125.0)
    assert len(m) == 1
    assert m[0].score == pytest.approx(THIRD)


def test_value_exactly_at_threshold_is_positive():
    assert len(_run(THIRD, 125.0)) == 1


def test_base_overlap_but_top_below_threshold_is_not_positive():
    # at offset 200 the 1000 m bases overlap heavily (IoU 0.667) but the 250 m tops
    # barely overlap (IoU ~0.111) -> with threshold 0.7 the pyramid is NOT positive.
    m = _run(0.7, 200.0)
    assert m == []


def test_positivity_decided_by_top_only_regardless_of_base():
    # base size is validated but does NOT enter the IoU; two identical runs agree.
    a = _run(0.3, 100.0)          # top IoU at offset 100 is ~0.4286
    b = _run(0.3, 100.0)
    assert [(x.tile_id, x.score) for x in a] == [(x.tile_id, x.score) for x in b]
    # the score equals the 250-top IoU, independent of the 1000 base
    exp = C.compute_iou(C.centered_square(0.0, 0.0, 250.0),
                        C.centered_square(100.0, 0.0, 250.0))
    assert a[0].score == pytest.approx(exp)


def test_top_is_concentric_with_base_centre():
    # point exactly at the base centre -> top fully covers query -> IoU 1.0
    assert _run(1.0, 0.0)[0].score == pytest.approx(1.0)


def test_malformed_base_rejected_not_corrected():
    with pytest.raises(ValueError):
        _run(0.5, 0.0, base_size=1200.0)          # base not ~1000 m -> rejected
