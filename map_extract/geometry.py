"""Pure metric geometry for true-metre extraction (no torch / rasterio / GPU).

EPSG:3857 (Web-Mercator) is the CRS the tiles/grid live in. Its unit is a metre
inflated by ``1/cos(lat)``, so a footprint or stride specified in TRUE ground metres
must be scaled UP by ``1/cos(lat)`` before a grid/window of that many EPSG:3857 units is
built at those coordinates. Without the scale, ``--stride_m 250`` produces only a
~250·cos(lat) ≈ 162 m ground grid at lat 49.

Kept dependency-free so the unit tests can verify the scaling with just stdlib.
"""
from __future__ import annotations

import math


def mercator_true_scale(center_lat_deg: float) -> float:
    """EPSG:3857 grid units per one TRUE ground metre at ``center_lat_deg`` (= 1/cos)."""
    if center_lat_deg is None or not math.isfinite(center_lat_deg) \
            or not (-89.0 <= center_lat_deg <= 89.0):
        raise ValueError(f"latitude {center_lat_deg!r} out of range for metric scaling")
    c = math.cos(math.radians(center_lat_deg))
    if c <= 1e-9:
        raise ValueError(f"degenerate cos(lat) at latitude {center_lat_deg}")
    return 1.0 / c


def scaled_pyramid_config(tile_size_m: float, stride_m, levels: int,
                          scale_factor: float, output_size_px: int,
                          true_scale: float = 1.0):
    """Build a :class:`~map_extract.pyramid.PyramidConfig` for a given metre basis.

    ``true_scale == 1.0`` (default) leaves sizes as raw EPSG:3857 units — the legacy
    behaviour. With ``true_scale = mercator_true_scale(lat)`` the tile/stride become
    TRUE metres: they are multiplied into EPSG:3857 units so the ground footprint is
    exactly ``tile_size_m`` / ``stride_m`` metres. The stored ``window_size_m`` should
    then be divided back by ``true_scale`` to record the real footprint.
    """
    if not math.isfinite(true_scale) or true_scale <= 0:
        raise ValueError(f"true_scale must be > 0, got {true_scale!r}")
    from .pyramid import PyramidConfig
    return PyramidConfig(
        tile_size_m=tile_size_m * true_scale,
        stride_m=None if stride_m is None else stride_m * true_scale,
        levels=levels, scale_factor=scale_factor, output_size_px=output_size_px,
    )
