"""Project adapters — internal. Reach these only through the subsystem public API."""
from __future__ import annotations

from .project_rule import build_gallery_index, build_geo_points

__all__ = ["build_gallery_index", "build_geo_points"]
