"""Project adapters — the ONLY reach into pre-existing repo code.

Reuses ``make_multicity_split`` helpers (``_merc`` projection, frame parsing, city
resolution) and the ``multicity_vlad_*.h5`` gallery schema to build the subsystem's
domain objects. No other module imports this file directly; it is reached only via
the subsystem's public loaders.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from ..models import GalleryIndex, GalleryTile, GeoPoint

_AREA_UNITS = "true_m2 (EPSG:3857 area * cos^2(lat))"


def _import_reused():
    try:
        import make_multicity_split as mm
    except Exception as e:  # pragma: no cover - environment guard
        raise RuntimeError(
            "positive_selection requires the top-level 'make_multicity_split' module on "
            "sys.path (run from the repo root). Import failed: " + repr(e))
    for n in ("_merc", "_parse_stem", "_collect_frames", "_city_and_dir"):
        if not hasattr(mm, n):
            raise RuntimeError(f"make_multicity_split is missing reused helper {n!r}")
    return mm


def build_gallery_index(tiles_h5, grid_crs: str = "EPSG:3857",
                        tile_size_fallback: float = 1000.0) -> GalleryIndex:
    mm = _import_reused()
    import h5py

    p = Path(tiles_h5).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"tiles_h5 not found: {p}")
    with h5py.File(p, "r") as f:
        for req in ("lat", "lon", "city"):
            if req not in f:
                raise KeyError(f"{p} lacks dataset '{req}' (not a multicity gallery H5?)")
        lat = np.asarray(f["lat"], float)
        lon = np.asarray(f["lon"], float)
        city = [c.decode() if isinstance(c, bytes) else str(c) for c in f["city"][:]]
        n = len(lat)
        tid = ([t.decode() if isinstance(t, bytes) else str(t) for t in f["tile_id"][:]]
               if "tile_id" in f else [f"{city[i]}:row{i}" for i in range(n)])
        win = np.asarray(f["window_size_m"], float) if "window_size_m" in f else np.full(n, np.nan)
    mx, my = mm._merc(lat, lon)
    tiles = [GalleryTile(tile_id=tid[i], city=city[i], center_x=float(mx[i]),
                         center_y=float(my[i]), lat=float(lat[i]), lon=float(lon[i]),
                         size_m=float(win[i])) for i in range(n)]
    return GalleryIndex(tiles, crs=grid_crs, area_units=_AREA_UNITS,
                        tile_size_fallback=tile_size_fallback)


def build_geo_points(gt: Sequence[str], grid_crs: str = "EPSG:3857") -> Tuple[GeoPoint, ...]:
    mm = _import_reused()
    out = {}
    for arg in gt:
        city, root = mm._city_and_dir(arg)
        root = Path(root).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"gt root missing: {root} (city {city!r})")
        for fr in mm._collect_frames(root):
            pid = f"{city}:{fr['stem']}"
            x, y = mm._merc(np.array([fr["lat"]]), np.array([fr["lon"]]))
            out[pid] = GeoPoint(point_id=pid, city=city, lat=float(fr["lat"]),
                                lon=float(fr["lon"]), x=float(x[0]), y=float(y[0]))
    if not out:
        raise ValueError("no query points collected from the configured gt roots")
    return tuple(out[k] for k in sorted(out))
