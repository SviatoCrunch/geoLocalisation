"""sky_filter — precompute sky keep-masks for UAV frames (the first drone-pipeline gate).

Before ANY operation on a drone frame, the sky is removed. Masks are computed ONCE
(precomputed file if present, else a neural segmentation fallback), cached per frame_id,
and reused everywhere (footprint estimation, feature extraction, query encoding) — one
consistent source. Sky is DROPPED (keep-mask; sky tokens removed before global/patch).

Isolated: numpy-only core (mask reduce/drop, store, resolver, precompute); the heavy
neural backend is behind a protocol and imported lazily (server-only). Applies to UAV
frames only — satellite crops have no sky.
"""
from __future__ import annotations

from .config import SkyFilterConfig
from .schemas import SkyMask
from .mask import reduce_to_grid, apply_drop
from .maskers import (SkyMasker, CascadingMasker, build_resolver, available_maskers,
                      register_masker)
from .store import MaskStore
from .precompute import precompute
from .ingest import ingest_sky_masks

__all__ = [
    "SkyFilterConfig", "SkyMask", "reduce_to_grid", "apply_drop",
    "SkyMasker", "CascadingMasker", "build_resolver", "available_maskers", "register_masker",
    "MaskStore", "precompute", "ingest_sky_masks",
]

__version__ = "0.1.0"
