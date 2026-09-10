"""Injectable feature-extraction + satellite-crop interfaces.

The heavy/external pieces (a frozen backbone, a satellite COG reader) live BEHIND these
protocols so the estimators/fusion/KMZ core stays numpy-only and testable with fakes.
Any object implementing these can be passed to :func:`footprint_assoc.pyramid.build`.
"""
from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable

import numpy as np


@runtime_checkable
class FeatureExtractor(Protocol):
    def extract(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """image (H,W,3) → (global_vec (D,), patch_grid (h,w,D)). Frozen; no target-data training."""
        ...


@runtime_checkable
class CropSource(Protocol):
    def crop(self, center_lat: float, center_lon: float, size_m: float) -> np.ndarray:
        """Return a satellite image (H,W,3) for the concentric square of side ``size_m`` at the centre."""
        ...
