"""Keyframe / delta PLANNING helpers — pure numpy, no cv2/torch.

The actual label propagation is the faithful SegProp port in ``segprop.py``; this
module only holds the sampling-plan math the calibration sweep needs: build the
uniform keyframe grid for a spacing, bracket an interior frame between its two
keyframes, turn a chosen spacing into a mask budget, and pick the largest spacing
that clears a quality threshold. Kept import-light so the planning logic is unit-
testable without OpenCV / torch / a GPU.
"""
from __future__ import annotations

import math


def uniform_keyframes(n_frames: int, delta: int) -> list[int]:
    """Uniform keyframe grid over frame indices [0, n_frames): 0, delta, 2*delta,
    ... plus the last frame (n_frames-1) so every interior frame is bracketed by two
    keyframes and no more than ``delta`` from an anchor. Requires delta >= 1."""
    if delta < 1:
        raise ValueError("delta must be >= 1")
    if n_frames <= 1:
        return list(range(n_frames))
    keys = list(range(0, n_frames, delta))
    if keys[-1] != n_frames - 1:
        keys.append(n_frames - 1)
    return keys


def bracket(keys: list[int], t: int):
    """Nearest keyframe on each side of interior frame ``t`` -> (left, right).
    Assumes ``keys`` sorted and keys[0] <= t <= keys[-1]."""
    left = keys[0]
    for k in keys:
        if k <= t:
            left = k
        else:
            return left, k
    return left, keys[-1]


def masks_for_span(span_frames: int, delta_star: int) -> int:
    """Minimum manual masks for a single continuous shot of ``span_frames`` at the
    calibrated max spacing ``delta_star``: ceil((span-1)/delta*) intervals + 1
    closing endpoint. Sum this over shots for a whole video (each shot resets)."""
    if span_frames <= 1:
        return span_frames
    return math.ceil((span_frames - 1) / max(int(delta_star), 1)) + 1


def pick_delta_star(curve, threshold: float, metric: str = "mean_f1"):
    """Largest tested ``delta`` whose ``metric`` still clears ``threshold``.
    ``curve`` = list of dicts each with 'delta' + the metric key. None if even the
    densest spacing misses the bar (then label denser / improve flow / lower bar)."""
    ok = [c["delta"] for c in curve if c.get(metric, 0.0) >= threshold]
    return max(ok) if ok else None
