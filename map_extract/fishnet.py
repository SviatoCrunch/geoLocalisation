"""Strict metric fishnet grid for EPSG:3857 rasters (vendored, torchgeo-free).

Byte-faithful port of RevisitAnything's ``src_tif_data/fishnet.py`` with the only change
being the removal of the ``torchgeo.datasets.BoundingBox`` dependency: it was used purely
as a 6-field container, so it is replaced by the local :class:`BoundingBox` namedtuple
(same field order ``minx, maxx, miny, maxy, mint, maxt`` and attribute access).

pyproj / shapely are imported lazily inside the functions that need them, so
``grid_centers`` / ``grid_bboxes`` (no AOI) work with only stdlib.
"""
from __future__ import annotations

import math
from collections import namedtuple
from typing import Any, List, Optional, Sequence, Tuple

# torchgeo's BoundingBox was only ever constructed as (minx, maxx, miny, maxy, 0, 1) and
# read via .minx/.maxx/.miny/.maxy — a plain namedtuple is a drop-in replacement.
BoundingBox = namedtuple("BoundingBox", ["minx", "maxx", "miny", "maxy", "mint", "maxt"])


def order_aoi_points(
    aoi_points: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Sort ``(lat, lon)`` points CCW by polar angle from their centroid."""
    pts = list(aoi_points)
    center_lat = sum(p[0] for p in pts) / len(pts)
    center_lon = sum(p[1] for p in pts) / len(pts)
    return sorted(
        pts,
        key=lambda p: math.atan2(p[0] - center_lat, p[1] - center_lon),
    )


def make_valid_aoi_polygon(
    aoi_points: Sequence[Tuple[float, float]],
) -> Tuple[Any, List[Tuple[float, float]]]:
    """Build a valid Shapely ``Polygon`` in EPSG:4326 from ``(lat, lon)`` points."""
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        from shapely.validation import make_valid
    except ImportError:
        raise ImportError("shapely is required: pip install shapely")

    ordered = order_aoi_points(aoi_points)
    polygon = Polygon([(lon, lat) for lat, lon in ordered])

    if not polygon.is_valid:
        polygon = make_valid(polygon)

    if polygon.geom_type == "GeometryCollection":
        parts = [g for g in polygon.geoms
                 if g.geom_type in ("Polygon", "MultiPolygon")]
        polygon = unary_union(parts)

    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda p: p.area)

    if polygon.geom_type != "Polygon":
        raise ValueError(f"Unsupported AOI geometry after repair: {polygon.geom_type}")
    if polygon.is_empty:
        raise ValueError("AOI polygon is empty after repair.")

    return polygon, ordered


def latlon_to_mercator_polygon(
    points_latlon: Sequence[Tuple[float, float]],
) -> Any:
    """Convert ``(lat, lon)`` points to a valid Shapely polygon in EPSG:3857."""
    try:
        from shapely.ops import transform as shapely_transform
    except ImportError:
        raise ImportError("shapely is required: pip install shapely")
    from pyproj import Transformer

    if hasattr(points_latlon, "tolist"):
        points_latlon = points_latlon.tolist()

    polygon_4326, _ = make_valid_aoi_polygon(points_latlon)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    return shapely_transform(tr.transform, polygon_4326)


def grid_centers(
    bounds,
    tile_size_m: float,
    stride_m: Optional[float] = None,
    safe_margin_m: float = 0.0,
) -> List[Tuple[float, float]]:
    """Return ``(cx, cy)`` centres of a strict metric grid in EPSG:3857.

    Centres sit on an exact ``stride_m``-spaced lattice within the inset extent
    (``bounds`` shrunk by ``safe_margin_m`` on every side). The last row/column is
    dropped when the next centre would exceed the inset boundary — no nudging, no
    overlap. ``bounds`` is any object with ``left/right/bottom/top`` (rasterio bounds).
    """
    if stride_m is None:
        stride_m = tile_size_m

    half = tile_size_m / 2.0
    min_cx = bounds.left + safe_margin_m + half
    max_cx = bounds.right - safe_margin_m - half
    min_cy = bounds.bottom + safe_margin_m + half
    max_cy = bounds.top - safe_margin_m - half

    centers: List[Tuple[float, float]] = []
    cy = min_cy
    while cy <= max_cy + 1e-6:
        cx = min_cx
        while cx <= max_cx + 1e-6:
            centers.append((cx, cy))
            cx += stride_m
        cy += stride_m
    return centers


def grid_bboxes(
    bounds,
    tile_size_m: float,
    stride_m: Optional[float] = None,
    safe_margin_m: float = 0.0,
    filter_window_m: Optional[float] = None,
    aoi_polygon=None,
    min_ratio: float = 0.3,
) -> List[BoundingBox]:
    """Return :class:`BoundingBox` objects on a strict metric grid.

    Wraps :func:`grid_centers` with an optional Shapely AOI filter. The AOI check uses a
    ``filter_window_m × filter_window_m`` box centred on each grid point (set it to the
    outermost pyramid level so every level of a kept tile overlaps the AOI).
    """
    if filter_window_m is None:
        filter_window_m = tile_size_m

    centers = grid_centers(bounds, tile_size_m, stride_m, safe_margin_m)
    half = tile_size_m / 2.0

    if aoi_polygon is None:
        return [BoundingBox(cx - half, cx + half, cy - half, cy + half, 0, 1)
                for cx, cy in centers]

    try:
        from shapely.geometry import box as shapely_box
    except ImportError:
        raise ImportError("shapely is required for AOI filtering: pip install shapely")

    filter_half = filter_window_m / 2.0
    result: List[BoundingBox] = []
    for cx, cy in centers:
        outer = shapely_box(cx - filter_half, cy - filter_half,
                            cx + filter_half, cy + filter_half)
        if outer.intersection(aoi_polygon).area / outer.area >= min_ratio:
            result.append(BoundingBox(cx - half, cx + half, cy - half, cy + half, 0, 1))
    return result
