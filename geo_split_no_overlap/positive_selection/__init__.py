"""positive_selection — pluggable positive-rule subsystem.

Public API only. The rest of geo_split_no_overlap must import from here and must NOT
reach into ``strategies/*`` or ``adapters/*``. The subsystem never imports the split
components / optimizer / audit (dependency flows one way: split -> here -> strategies).
"""
from __future__ import annotations

from .adapters import build_gallery_index, build_geo_points
from .config import PositiveSelectionConfig
from .models import (GalleryIndex, GalleryTile, GeoPoint, MaterializedPositiveSets,
                     PositiveMatch)
from .protocol import PositiveSelector
from .registry import (available_positive_selectors, create_positive_selector,
                       register_positive_selector)
from .service import materialize_positive_sets

__all__ = [
    # models / contract
    "GeoPoint", "GalleryTile", "GalleryIndex", "PositiveMatch",
    "MaterializedPositiveSets", "PositiveSelector", "PositiveSelectionConfig",
    # registry
    "register_positive_selector", "create_positive_selector",
    "available_positive_selectors",
    # service + loaders
    "materialize_positive_sets", "build_gallery_index", "build_geo_points",
]
