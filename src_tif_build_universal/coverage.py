"""Coverage providers — determine which zoom level to use at each AOI location.

A CoverageProvider answers one question before downloading starts:
"At each point in the AOI, what is the best available zoom level?"

The answer is returned as {zoom: shapely_geometry} where geometries are
non-overlapping — every pixel belongs to exactly one zoom level.
Higher zoom levels must be queried / downloaded first so the mosaic
merge (first-valid-pixel-wins) correctly preserves higher resolution.

Implementations
---------------
GoogleCoverageProvider    — 1 Viewport API request; returns maxZoom per rect.
FixedZoomCoverageProvider — single zoom, full AOI (Esri or any fixed source).
MultiZoomCoverageProvider — multiple fixed zooms, full AOI each (no coverage API).

Adding a new source
-------------------
Subclass CoverageProvider and implement query().  The orchestrator
(mosaic.build_mosaic) calls nothing else on this object.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from .viewport import (
    apply_max_zoom_cap,
    build_effective_zoom_geometries,
    query_aoi_max_zoom_rects,
)


class CoverageProvider(ABC):
    """Determines effective zoom geometry per zoom level for an AOI.

    Contract for query():
    - Keys are integer zoom levels.
    - Values are valid Shapely geometries in (lon, lat) coordinates.
    - Geometries must be non-overlapping across zoom levels.
    - Higher zoom levels represent higher resolution and are downloaded first.
    """

    @abstractmethod
    def query(
        self,
        aoi_points: list,
        aoi_polygon,
    ) -> Dict[int, Any]:
        """Return {zoom: shapely_geometry} of non-overlapping coverage.

        Args:
            aoi_points: Ordered AOI vertices in (lat, lon) format.
            aoi_polygon: Valid Shapely polygon of the AOI in (lon, lat).
        """


class GoogleCoverageProvider(CoverageProvider):
    """Uses Google Viewport API (1 request) to find maxZoom per location.

    Google returns overlapping maxZoomRects — a low-zoom fallback may cover
    the same area as a higher-zoom local rect.  This provider resolves them
    into non-overlapping effective geometries:

        covered by z13, z14, z16, z19  →  use z19
        covered by z13, z14, z16       →  use z16

    Args:
        api_key: Google Cloud API key with Map Tiles API enabled.
        preferred_zoom: Zoom sent to the Viewport API query (default 20).
        max_zoom_to_use: Cap on downloaded zoom — rects above this are
            downgraded.  None means no cap.
        min_effective_area: Shapely area threshold in (lon, lat) units.
            Geometries smaller than this are dropped as boundary noise.
    """

    def __init__(
        self,
        api_key: str,
        preferred_zoom: int = 20,
        max_zoom_to_use: Optional[int] = None,
        min_effective_area: float = 1e-8,
    ) -> None:
        self._api_key = api_key
        self._preferred_zoom = preferred_zoom
        self._max_zoom_to_use = max_zoom_to_use
        self._min_effective_area = min_effective_area

    def query(self, aoi_points: list, aoi_polygon) -> Dict[int, Any]:
        logger.info("Querying Google Viewport API for max-zoom coverage ...")
        rects = query_aoi_max_zoom_rects(
            api_key=self._api_key,
            aoi_points=aoi_points,
            preferred_zoom=self._preferred_zoom,
        )

        if not rects:
            raise RuntimeError(
                "Google Viewport API returned no coverage rectangles for this AOI."
            )

        logger.info("Raw zoom levels from API : %s", sorted({r['maxZoom'] for r in rects}))

        rects = apply_max_zoom_cap(rects, self._max_zoom_to_use)
        logger.info("Effective zoom levels    : %s", sorted({r['maxZoom'] for r in rects}))

        geoms = build_effective_zoom_geometries(rects, aoi_polygon)

        return {
            zoom: geom
            for zoom, geom in geoms.items()
            if not geom.is_empty and geom.area > self._min_effective_area
        }


class FixedZoomCoverageProvider(CoverageProvider):
    """Returns a single zoom level covering the entire AOI.

    Use for sources that have no coverage API (e.g. Esri World Imagery).
    The full AOI is downloaded at the given zoom level.

    Args:
        zoom: The single zoom level to use.
    """

    def __init__(self, zoom: int) -> None:
        self._zoom = zoom

    def query(self, aoi_points: list, aoi_polygon) -> Dict[int, Any]:
        return {self._zoom: aoi_polygon}


class MultiZoomCoverageProvider(CoverageProvider):
    """Returns multiple fixed zoom levels, each covering the full AOI.

    Use when you want to blend several resolutions without a coverage API.
    Higher zoom levels are downloaded first and preserved by the mosaic
    merge; lower zoom levels fill any remaining empty pixels.

    Args:
        zooms: List of zoom levels to download (sorted internally high→low).
    """

    def __init__(self, zooms: List[int]) -> None:
        if not zooms:
            raise ValueError("zooms must contain at least one level")
        self._zooms = sorted(set(zooms))

    def query(self, aoi_points: list, aoi_polygon) -> Dict[int, Any]:
        return {z: aoi_polygon for z in self._zooms}
