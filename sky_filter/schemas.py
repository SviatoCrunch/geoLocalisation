"""Data structures for pre-computed sky filtering of UAV frames.

The mask is a PIXEL keep-mask (True = keep ground, False = sky), extractor-agnostic; a
reducer maps it to any token grid (see ``mask.py``). Numpy-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SkyMask:
    """Per-frame pixel keep-mask (True=keep ground). ``backend`` records who produced it."""
    frame_id: str
    keep: np.ndarray                # (H, W) bool  — True = ground (kept), False = sky
    backend: str                    # "precomputed" | "neural" | ...
    version: str = "1.0"
    meta: dict = field(default_factory=dict)

    @property
    def sky_fraction(self) -> float:
        return float(1.0 - self.keep.mean()) if self.keep.size else 0.0
