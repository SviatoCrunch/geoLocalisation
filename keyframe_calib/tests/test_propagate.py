"""Keyframe/delta planning-helper tests — pure, no cv2/torch/GPU."""
from __future__ import annotations

from keyframe_calib.propagate import (
    bracket,
    masks_for_span,
    pick_delta_star,
    uniform_keyframes,
)


def test_uniform_keyframes_includes_endpoints():
    assert uniform_keyframes(101, 50) == [0, 50, 100]
    assert uniform_keyframes(100, 50) == [0, 50, 99]  # last frame appended
    assert uniform_keyframes(1, 50) == [0]


def test_bracket():
    keys = [0, 50, 100]
    assert bracket(keys, 25) == (0, 50)
    assert bracket(keys, 50) == (50, 100)  # on a key -> (self, next)
    assert bracket(keys, 75) == (50, 100)


def test_masks_for_span():
    assert masks_for_span(101, 50) == 3   # 0, 50, 100
    assert masks_for_span(1, 50) == 1
    assert masks_for_span(200, 50) == 5


def test_pick_delta_star():
    curve = [{"delta": 10, "mean_f1": 0.97}, {"delta": 50, "mean_f1": 0.93},
             {"delta": 100, "mean_f1": 0.80}]
    assert pick_delta_star(curve, 0.90) == 50
    assert pick_delta_star(curve, 0.99) is None
    assert pick_delta_star(curve, 0.90, "mean_iou") is None  # metric absent -> 0.0
