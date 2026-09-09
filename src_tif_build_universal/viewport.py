"""Google Viewport API client and effective-zoom geometry computation.

No dependencies on other library modules or src_tif_build.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import requests
from shapely.geometry import box
from shapely.validation import make_valid

_SESSION_URL = "https://tile.googleapis.com/v1/createSession"
_VIEWPORT_URL = "https://tile.googleapis.com/tile/v1/viewport"
_VIEWPORT_TIMEOUT = 30  # seconds


def create_google_session(api_key: str) -> str:
    """Create a Google Map Tiles API satellite session token."""
    resp = requests.post(
        _SESSION_URL,
        params={"key": api_key},
        json={"mapType": "satellite", "language": "en-US", "region": "US"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["session"]


def query_aoi_max_zoom_rects(
    api_key: str,
    aoi_points: list,
    preferred_zoom: int = 20,
) -> List[Dict]:
    """Query Google Viewport API for AOI maxZoom rectangles.

    Returns local maxZoomRects only (global fallback rects that span
    ±89° latitude are filtered out).

    Args:
        api_key: Google Cloud API key with Map Tiles API enabled.
        aoi_points: AOI vertices in (lat, lon) format.
        preferred_zoom: Zoom level sent to the Viewport API.

    Returns:
        List of rect dicts with keys: north, south, east, west, maxZoom.
    """
    token = create_google_session(api_key)
    lats = [p[0] for p in aoi_points]
    lons = [p[1] for p in aoi_points]

    resp = requests.get(
        _VIEWPORT_URL,
        params={
            "session": token,
            "key": api_key,
            "zoom": preferred_zoom,
            "north": max(lats),
            "south": min(lats),
            "east": max(lons),
            "west": min(lons),
        },
        timeout=_VIEWPORT_TIMEOUT,
    )
    resp.raise_for_status()

    rects = resp.json().get("maxZoomRects", [])
    return [r for r in rects if not (r["north"] >= 89 and r["south"] <= -89)]


def apply_max_zoom_cap(
    rects: List[Dict],
    max_zoom: Optional[int],
) -> List[Dict]:
    """Downgrade rectangles whose maxZoom exceeds max_zoom.

    Args:
        rects: maxZoomRects from the Viewport API.
        max_zoom: Cap value. None = no cap.

    Returns:
        New list with maxZoom clamped to max_zoom.
    """
    if max_zoom is None:
        return list(rects)
    return [{**r, "maxZoom": min(r["maxZoom"], max_zoom)} for r in rects]


def build_effective_zoom_geometries(
    rects: List[Dict],
    aoi_polygon,
) -> Dict[int, object]:
    """Resolve overlapping maxZoomRects into non-overlapping effective geometries.

    Google rects overlap: a low-zoom fallback may cover the same area as a
    higher-zoom local rect.  This function clips each lower zoom by all
    higher zooms so that every pixel belongs to exactly one zoom level:

        covered by z13, z14, z16, z19  →  effective zoom = z19
        covered by z13, z14, z16       →  effective zoom = z16

    Args:
        rects: maxZoomRects (possibly after apply_max_zoom_cap).
        aoi_polygon: Valid Shapely polygon in (lon, lat) coordinates.

    Returns:
        {zoom: shapely_geometry} with non-overlapping geometries.
    """
    zoom_to_geom: Dict[int, object] = {}

    for rect in rects:
        zoom = rect["maxZoom"]
        rect_poly = box(rect["west"], rect["south"], rect["east"], rect["north"])
        clipped = aoi_polygon.intersection(rect_poly)

        if clipped.is_empty:
            continue
        if not clipped.is_valid:
            clipped = make_valid(clipped)

        if zoom not in zoom_to_geom:
            zoom_to_geom[zoom] = clipped
        else:
            zoom_to_geom[zoom] = zoom_to_geom[zoom].union(clipped)

    higher_geom = None
    effective: Dict[int, object] = {}

    for zoom in sorted(zoom_to_geom, reverse=True):
        geom = zoom_to_geom[zoom]

        if higher_geom is not None:
            geom = geom.difference(higher_geom)

        if geom.is_empty:
            higher_geom = (
                zoom_to_geom[zoom]
                if higher_geom is None
                else higher_geom.union(zoom_to_geom[zoom])
            )
            continue

        if not geom.is_valid:
            geom = make_valid(geom)

        effective[zoom] = geom
        higher_geom = (
            zoom_to_geom[zoom]
            if higher_geom is None
            else higher_geom.union(zoom_to_geom[zoom])
        )

    return effective
