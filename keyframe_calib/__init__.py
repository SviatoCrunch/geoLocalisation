"""keyframe_calib — data-driven MINIMUM number of manual masks for mask propagation.

Answers "how few keyframes must I hand-label?" by measuring the SegProp degradation
curve F-measure(delta) on YOUR footage: densely label one short calibration segment,
then for each candidate keyframe spacing ``delta`` propagate the sparse keyframes'
masks with the FAITHFUL SegProp vote and score the filled-in frames against the dense
ground truth. The largest ``delta`` that still clears your threshold is the answer:
``N_min = masks_for_span(shot_len, delta*)`` masks per shot.

Pipeline: frames -> dense optical flow (RAFT / Farneback) -> SegProp vote (four
forward/backward key+cur projections, exp(-beta*delta) weighting, optional per-CC
homography votes) -> mean-F1 vs GT -> curve -> delta*.

Layers:
    segprop.py    faithful torch port of the original SegProp vote (parity-tested)
    propagate.py  pure keyframe/delta PLANNING math (testable, no cv2/torch)
    metrics.py    pure mean-IoU / mean-F1 (== SegProp per-class F-measure)
    flow.py       dense optical flow backends (lazy cv2 / torchvision RAFT)
    calibrate.py  runner + CLI: the F-measure(delta) sweep -> delta* -> mask budget
"""
from __future__ import annotations

from .metrics import confusion, evaluate, iou_f1_from_confusion
from .propagate import bracket, masks_for_span, pick_delta_star, uniform_keyframes

__all__ = [
    "uniform_keyframes",
    "bracket",
    "masks_for_span",
    "pick_delta_star",
    "confusion",
    "iou_f1_from_confusion",
    "evaluate",
]
