"""Shared, self-contained geometry + validation helpers for built-in strategies.

Kept inside the strategies package so the subsystem has NO dependency on the split
module's ``spatial_conflicts`` (which would violate the dependency boundary). All box
math is in the grid CRS (EPSG:3857); areas/distances are converted to true metres via
the cos^2(lat) / cos(lat) conformal factors.
"""
from __future__ import annotations

import math
from typing import Mapping

from ..models import GalleryIndex, GalleryTile, GeoPoint

GRID_CRS = "EPSG:3857"


def ensure_grid_crs(gallery: GalleryIndex) -> None:
    """Geometric metric rules require the known grid CRS (explicit CRS check)."""
    if gallery.crs != GRID_CRS:
        raise ValueError(f"strategy expects gallery CRS {GRID_CRS}, got {gallery.crs!r}")


def tile_side(t: GalleryTile, fallback: float) -> float:
    return t.size_m if (t.size_m == t.size_m and t.size_m > 0) else fallback


def true_m2(area_grid: float, lat_deg: float) -> float:
    c = math.cos(math.radians(lat_deg))
    return area_grid * c * c


def true_dist_m(dist_grid: float, lat_deg: float) -> float:
    return dist_grid * math.cos(math.radians(lat_deg))


def rect_intersection_area(ax, ay, ah, bx, by, bh) -> float:
    """Intersection area of two axis-aligned squares (centre a=(ax,ay) half ah, etc.).

    0.0 for disjoint OR merely-touching squares (a bare boundary touch is not overlap).
    """
    dx = min(ax + ah, bx + bh) - max(ax - ah, bx - bh)
    dy = min(ay + ah, by + bh) - max(ay - ah, by - bh)
    if dx <= 0.0 or dy <= 0.0:
        return 0.0
    return dx * dy


def resolve_params(params: Mapping, defaults: Mapping, name: str) -> dict:
    """Reject unknown params, fill defaults. Value/type checks are per-strategy."""
    unknown = set(params) - set(defaults)
    if unknown:
        raise ValueError(f"{name}: unknown params {sorted(unknown)}; "
                         f"allowed {sorted(defaults)}")
    out = dict(defaults)
    out.update(params)
    return out


def require_number(v, name: str, *, positive=False, unit_interval=False) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{name} must be a number, got {v!r}")
    v = float(v)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v}")
    if positive and v <= 0:
        raise ValueError(f"{name} must be > 0, got {v}")
    if unit_interval and not (0.0 < v <= 1.0):
        raise ValueError(f"{name} must be in (0, 1], got {v}")
    return v


# ── shapely-based square + IoU helpers (shared by the IoU strategies) ──────────────
# IoU is a dimensionless ratio, so the cos^2(lat) distortion of EPSG:3857 cancels; the
# squares only need to be built in the SAME metric CRS the gallery/points live in.

def validate_metric_crs(crs: str) -> None:
    """Reject a missing CRS or a geographic (degree) CRS. IoU squares need metres."""
    if not crs:
        raise ValueError("gallery CRS is missing; IoU strategies require a metric CRS")
    c = str(crs).upper()
    if "4326" in c or c in ("WGS84", "CRS84", "OGC:CRS84", "EPSG:4979"):
        raise ValueError(f"gallery CRS {crs!r} is geographic (degrees); a metric CRS "
                         f"(e.g. {GRID_CRS}) is required — do not use degrees as metres")


def validate_square_size(actual_side: float, expected_side: float, tolerance_m: float,
                         what: str = "tile") -> None:
    """Fail (never silently rescale) if a geometry is not the expected square size."""
    if actual_side is None or not math.isfinite(actual_side) or actual_side <= 0:
        raise ValueError(f"{what} has invalid/zero side {actual_side!r}")
    if abs(actual_side - expected_side) > tolerance_m:
        raise ValueError(f"{what} side {actual_side} m != expected {expected_side} m "
                         f"(tolerance {tolerance_m} m); refusing to silently rescale")


def centered_square(x: float, y: float, side_m: float):
    """A square (shapely box) centred at (x, y) with the given side, in metric units."""
    from shapely.geometry import box
    if side_m is None or not math.isfinite(side_m) or side_m <= 0:
        raise ValueError(f"square side must be finite and > 0, got {side_m!r}")
    h = side_m / 2.0
    return box(x - h, y - h, x + h, y + h)


def compute_iou(a, b) -> float:
    """Standard IoU of two geometries: |A∩B| / (|A|+|B|-|A∩B|).

    Explicit edge-case handling; no unconditional ``buffer(0)`` repair. Result is
    clamped to [0, 1] against float noise. Touching-only geometries -> 0.0.
    """
    if a is None or b is None or a.is_empty or b.is_empty:
        raise ValueError("cannot compute IoU on empty/None geometry")
    if not a.is_valid or not b.is_valid:
        raise ValueError("invalid geometry passed to compute_iou (no silent buffer(0) repair)")
    area_a, area_b = a.area, b.area
    if area_a <= 0.0 or area_b <= 0.0:
        raise ValueError("cannot compute IoU on zero-area geometry")
    inter = a.intersection(b).area
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    iou = inter / union
    if iou < 0.0:
        return 0.0
    if iou > 1.0:
        return 1.0
    return iou
