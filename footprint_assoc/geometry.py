"""Concentric-square footprints + overlap metrics (EPSG:3857 grid).

Self-contained (numpy + math). The footprint of a chosen level is the concentric square
of side ``s`` around the frame centre; CQ/CT/IoU are area ratios (scale-invariant, so the
3857 area distortion cancels — no cos^2 needed for the ratios).
"""
from __future__ import annotations

import math

_R = 6378137.0  # EPSG:3857 spherical radius


def merc(lat: float, lon: float) -> tuple:
    x = _R * math.radians(lon)
    y = _R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def merc_inv(x: float, y: float) -> tuple:
    lon = math.degrees(x / _R)
    lat = math.degrees(2.0 * math.atan(math.exp(y / _R)) - math.pi / 2.0)
    return lon, lat


def square_bounds(cx: float, cy: float, side_m: float) -> tuple:
    """(minx, miny, maxx, maxy) for a square centred at (cx, cy) with the given side."""
    h = side_m / 2.0
    return (cx - h, cy - h, cx + h, cy + h)


def footprint_bounds(center_lat: float, center_lon: float, side_m: float) -> tuple:
    cx, cy = merc(center_lat, center_lon)
    return square_bounds(cx, cy, side_m)


def rect_intersection_area(a: tuple, b: tuple) -> float:
    """Intersection area of two (minx,miny,maxx,maxy) rects (0 for touch/disjoint)."""
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if dx <= 0.0 or dy <= 0.0:
        return 0.0
    return dx * dy


def rect_area(a: tuple) -> float:
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def overlap_metrics(footprint: tuple, tile: tuple) -> tuple:
    """Return (CQ, CT, IoU) between a footprint rect and a tile rect (area ratios)."""
    inter = rect_intersection_area(footprint, tile)
    ap, at = rect_area(footprint), rect_area(tile)
    if inter <= 0.0 or ap <= 0.0 or at <= 0.0:
        return 0.0, 0.0, 0.0
    union = ap + at - inter
    return inter / ap, inter / at, inter / union


def square_corners_lonlat(cx: float, cy: float, side_m: float) -> list:
    """4+1 corner ring (lon,lat) of a 3857 square — for KMZ polygons."""
    h = side_m / 2.0
    pts = [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h), (cx - h, cy - h)]
    return [merc_inv(x, y) for x, y in pts]
