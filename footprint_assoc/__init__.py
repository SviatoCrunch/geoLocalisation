"""footprint_assoc — zero-shot UAV→pyramid footprint association testbed (isolated).

For each UAV frame with a KNOWN centre + a concentric satellite pyramid, estimate the
best level / footprint WITHOUT target training, to build overlap-aware positives + weights
for the main training. NOT global geolocation.

Iteration 1 estimators (data-ready, on frozen DINOv2 tokens): ``global_vlad`` (whole-image
cosine sweep) + ``patch_overlap`` (co-visible patch score). Fusion: ``cascade`` (default;
global→top, patch→order, agreement→confidence, refuse/soft when weak) + ``rank_vote``
(scale-invariant RRF experiment). Footprint→gallery association gives CQ/CT/IoU + pair
status. KMZ visualises everything (poor-man's oracle). Geometry(RANSAC)/semantic are later.

Fully self-contained: core is numpy-only; the heavy DINOv2/COG pieces are injected via
protocols (``extractors.base``) — no dependency on RevisitAnything or other packages.
"""
from __future__ import annotations

from .config import FootprintConfig, geometric_scales
from .schemas import (PyramidFeatures, QueryFeatures, LevelFeatures, LevelScore,
                      FrameEstimate, PairAssoc)
from .estimators import available_estimators, create_estimator, register_estimator
from .fusion import available_fusion, create_fusion, register_fusion
from .association import associate
from .pipeline import estimate_frame, estimate_and_associate
from . import pyramid, kmz

__all__ = [
    "FootprintConfig", "geometric_scales",
    "PyramidFeatures", "QueryFeatures", "LevelFeatures", "LevelScore", "FrameEstimate", "PairAssoc",
    "available_estimators", "create_estimator", "register_estimator",
    "available_fusion", "create_fusion", "register_fusion",
    "associate", "estimate_frame", "estimate_and_associate", "pyramid", "kmz",
]

__version__ = "0.1.0"
