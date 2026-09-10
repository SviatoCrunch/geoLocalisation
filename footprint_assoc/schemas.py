"""Typed data structures for zero-shot UAV→pyramid footprint association.

Model-agnostic: features are plain numpy arrays (a global descriptor + a patch-token
grid) per level, so the estimators/fusion/KMZ never touch a backbone. Geometry is in the
EPSG:3857 grid CRS (consistent with the rest of geoLocalisation).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# pair status for footprint→gallery association
STRONG, PARTIAL, UNDETERMINED, NEGATIVE = "strong", "partial", "undetermined", "negative"

# frame estimate status
ACCEPT, SOFT, REFUSE = "accept", "soft", "refuse"


@dataclass(frozen=True)
class LevelFeatures:
    """Satellite concentric-crop features at one physical scale (metres)."""
    scale_m: float
    global_vec: np.ndarray          # (D,) L2-normalised descriptor
    patch_grid: np.ndarray          # (H, W, D) token grid (row-major)


@dataclass(frozen=True)
class QueryFeatures:
    """UAV frame features (the whole visible frame)."""
    global_vec: np.ndarray          # (D,)
    patch_grid: np.ndarray          # (Hq, Wq, D)


@dataclass(frozen=True)
class PyramidFeatures:
    """One UAV frame + its concentric satellite pyramid, all sharing centre (lat, lon)."""
    frame_id: str
    center_lat: float
    center_lon: float
    query: QueryFeatures
    levels: tuple                   # tuple[LevelFeatures] sorted by scale_m

    def scales(self) -> list:
        return [lv.scale_m for lv in self.levels]


@dataclass(frozen=True)
class LevelScore:
    """One method's score for one level (+ diagnostics)."""
    scale_m: float
    score: float
    aux: dict = field(default_factory=dict)


@dataclass
class FrameEstimate:
    """Fused per-frame result over the pyramid levels."""
    frame_id: str
    center_lat: float
    center_lon: float
    scales: list                    # level scales (m), ascending
    per_method: dict                # method_name -> list[LevelScore] (aligned to scales)
    soft: dict                      # scale_m -> probability (fused, sums to 1)
    best_scale: Optional[float]     # None if refused
    confidence: float
    status: str                     # ACCEPT | SOFT | REFUSE
    reason: str = ""


@dataclass(frozen=True)
class PairAssoc:
    """Footprint↔gallery-tile association metrics + status."""
    tile_id: str
    cq: float                       # area(P∩T)/area(P)  — fraction of footprint covered
    ct: float                       # area(P∩T)/area(T)  — fraction of tile seen
    iou: float
    weight: float                   # overlap weight for the loss
    status: str                     # STRONG | PARTIAL | UNDETERMINED | NEGATIVE
