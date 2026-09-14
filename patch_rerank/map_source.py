"""Read arbitrary ground windows from the real EPSG:3857 GeoTIFF (for the sliding-window fine stage).

Windows are specified in TRUE metres and converted to CRS units (× 1/cos(lat)) so a "1000 m" crop is
1000 real metres — matching how the gallery was extracted (--true_meters). Mirrors
``map_extract.pyramid._read_from`` (rasterio window_from_bounds + bilinear resize)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

_R = 6378137.0                                    # EPSG:3857 sphere radius


def latlon_to_merc(lat: float, lon: float):
    return _R * math.radians(lon), _R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def merc_to_latlon(x: float, y: float):
    return (math.degrees(2 * math.atan(math.exp(y / _R)) - math.pi / 2), math.degrees(x / _R))


def true_m_to_crs(size_true_m: float, lat: float) -> float:
    """TRUE metres → EPSG:3857 CRS units at ``lat`` (Web-Mercator inflates by 1/cos(lat))."""
    return size_true_m / math.cos(math.radians(lat))


def resolve_maps(tif_dir, cities) -> dict:
    """city → first GeoTIFF in ``tif_dir`` whose name contains the city (``*<city>*.tif``)."""
    d = Path(tif_dir).expanduser()
    out = {}
    for c in cities:
        cands = sorted(d.glob(f"*{c}*.tif")) + sorted(d.glob(f"*{c}*.tiff"))
        if cands:
            out[c] = str(cands[0])
    return out


def open_src(path):
    import rasterio
    return rasterio.open(str(Path(path).expanduser()))


def read_window(src, cx: float, cy: float, size_crs_m: float, output_px: int) -> np.ndarray:
    """RGB (H,W,3) uint8 crop centred at merc (cx,cy), side ``size_crs_m`` CRS units, resized to px."""
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds
    h = size_crs_m / 2.0
    win = from_bounds(cx - h, cy - h, cx + h, cy + h, src.transform)
    arr = src.read([1, 2, 3], window=win, out_shape=(3, output_px, output_px),
                   resampling=Resampling.bilinear, boundless=True, fill_value=0)
    return arr.transpose(1, 2, 0).astype(np.uint8)


def read_pyramid_from_one(src, cx: float, cy: float, sizes_true_m, lat: float, output_px: int,
                          max_internal_px: int = 1536) -> dict:
    """Read the LARGEST level window ONCE (at high res) and derive every level by centre-crop +
    resize — cuts rasterio I/O ~N×. → {size_true_m: (output_px,output_px,3) uint8}. Concentric
    levels share the same centre, so smaller levels are exact centre crops of the big read."""
    import cv2
    sizes = list(sizes_true_m)
    Lmax, Lmin = max(sizes), min(sizes)
    internal = int(min(max_internal_px, max(output_px, round(output_px * Lmax / Lmin))))
    big = read_window(src, cx, cy, true_m_to_crs(Lmax, lat), internal)      # one read
    c = internal / 2.0
    out = {}
    for L in sizes:
        half = (L / Lmax) * internal / 2.0
        lo, hi = int(round(c - half)), int(round(c + half))
        crop = big[max(lo, 0):hi, max(lo, 0):hi]
        out[L] = cv2.resize(crop, (output_px, output_px), interpolation=cv2.INTER_LINEAR)
    return out
