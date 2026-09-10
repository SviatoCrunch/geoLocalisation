"""Injectable extractor/crop interfaces (+ optional server-only DINOv2 adapter)."""
from __future__ import annotations

from .base import FeatureExtractor, CropSource

__all__ = ["FeatureExtractor", "CropSource"]

# DinoV2Extractor is imported lazily (it pulls torch): `from footprint_assoc.extractors.dinov2 import DinoV2Extractor`
