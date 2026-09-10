"""precomputed — read a ready-made mask from the store (no model, data-ready)."""
from __future__ import annotations

from typing import Optional

import numpy as np


class PrecomputedMasker:
    name = "precomputed"
    version = "1.0"

    def __init__(self, store):
        self._store = store

    def mask(self, frame_id: str, image=None) -> Optional[np.ndarray]:
        if self._store is None:
            return None
        return self._store.load(frame_id)          # keep (H,W) bool, or None if absent
