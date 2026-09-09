"""Shared AOI / polygon geometry utilities.

No dependencies on other library modules or src_tif_build.
"""

from __future__ import annotations

import math
from typing import Generator, List, Tuple

from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

LatLon = Tuple[float, float]


def order_aoi_points(aoi_points: List[LatLon]) -> List[LatLon]:
    """Sort (lat, lon) points angularly around their centroid.

    Guarantees a non-self-intersecting polygon regardless of input order.
    """
    c_lat = sum(p[0] for p in aoi_points) / len(aoi_points)
    c_lon = sum(p[1] for p in aoi_points) / len(aoi_points)
    return sorted(
        aoi_points,
        key=lambda p: math.atan2(p[0] - c_lat, p[1] - c_lon),
    )


def make_aoi_polygon(
    aoi_points: List[LatLon],
) -> Tuple[Polygon, List[LatLon]]:
    """Build a valid Shapely polygon from AOI points.

    Orders points angularly, repairs self-intersections if any, and
    returns the largest polygon component if the result is multi-part.

    Returns:
        (polygon_lonlat, ordered_latlon_points)
        polygon uses (lon, lat) coordinates internally.
    """
    ordered = order_aoi_points(aoi_points)
    poly = Polygon([(lon, lat) for lat, lon in ordered])

    if not poly.is_valid:
        poly = make_valid(poly)

    if poly.geom_type == "GeometryCollection":
        parts = [g for g in poly.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        poly = unary_union(parts)

    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)

    if poly.is_empty or poly.geom_type != "Polygon":
        raise ValueError(f"Cannot build a valid AOI polygon: {poly.geom_type}")

    return poly, ordered


def iter_polygons(geom) -> Generator[Polygon, None, None]:
    """Yield all Polygon parts from any Shapely geometry."""
    if geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        for g in geom.geoms:
            if g.geom_type == "Polygon":
                yield g


def polygon_to_aoi_points(polygon: Polygon) -> List[LatLon]:
    """Convert a Shapely polygon exterior to (lat, lon) point list."""
    return [(lat, lon) for lon, lat in polygon.exterior.coords[:-1]]


def geometry_to_aoi_parts(geom) -> List[List[LatLon]]:
    """Convert any Shapely geometry to a list of (lat, lon) point lists.

    Each list corresponds to one polygon part (MultiPolygon is split).
    """
    return [
        polygon_to_aoi_points(poly)
        for poly in iter_polygons(geom)
        if not poly.is_empty
    ]
