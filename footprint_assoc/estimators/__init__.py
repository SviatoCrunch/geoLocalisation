"""Per-level zero-shot estimators (Global VLAD sweep, Patch overlap) + Registry."""
from __future__ import annotations

from .base import (LevelEstimator, available_estimators, create_estimator, register_estimator)
from .global_vlad import GlobalVladEstimator
from .patch_overlap import PatchOverlapEstimator

__all__ = ["LevelEstimator", "available_estimators", "create_estimator", "register_estimator",
           "GlobalVladEstimator", "PatchOverlapEstimator"]
