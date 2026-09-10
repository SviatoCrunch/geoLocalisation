"""Synthetic fixtures for the geo_split_no_overlap tests (no real data needed)."""
from __future__ import annotations

import math
from pathlib import Path

from geo_split_no_overlap.config import Config
from geo_split_no_overlap.schemas import Component
from geo_split_no_overlap.positive_selection import GalleryIndex, GalleryTile, GeoPoint
from geo_split_no_overlap.positive_selection.fingerprint import compute_fingerprint
from geo_split_no_overlap.positive_selection.models import (MaterializedPositiveSets,
                                                            freeze_mapping)

_R = 6378137.0  # must match make_multicity_split / project_rule


def xy_to_latlon(x: float, y: float) -> tuple:
    lon = math.degrees(x / _R)
    lat = math.degrees(2.0 * math.atan(math.exp(y / _R)) - math.pi / 2.0)
    return lat, lon


def latlon_to_xy(lat: float, lon: float) -> tuple:
    x = _R * math.radians(lon)
    y = _R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def cfg(area_epsilon_m2=1.0, tile_size_m=1000.0, tol=0.05,
        ratios=(0.7, 0.15, 0.15), **kw) -> Config:
    return Config(area_epsilon_m2=area_epsilon_m2, tile_size_m=tile_size_m,
                  ratio_tolerance=tol, train_ratio=ratios[0], val_ratio=ratios[1],
                  test_ratio=ratios[2], **kw)


def tile(tid, x: float, y: float, size: float = 1000.0,
         city: str = "c", lat: float = 48.5) -> GalleryTile:
    return GalleryTile(tile_id=str(tid), city=city, center_x=x, center_y=y,
                       lat=lat, lon=37.8, size_m=size)


def make_gallery(tiles, crs="EPSG:3857", fallback=1000.0) -> GalleryIndex:
    return GalleryIndex(list(tiles), crs=crs, area_units="true_m2 (test)",
                        tile_size_fallback=fallback)


def geo_point(pid, x, y, city="c") -> GeoPoint:
    lat, lon = xy_to_latlon(x, y)
    return GeoPoint(point_id=pid, city=city, lat=lat, lon=lon, x=x, y=y)


def make_mps(point_to_tiles: dict, gallery: GalleryIndex,
             strategy_name="test", version="0", no_positive=()) -> MaterializedPositiveSets:
    p2t = {pid: frozenset(str(t) for t in tids) for pid, tids in point_to_tiles.items()}
    resolved = {"version": version}
    fp = compute_fingerprint(strategy_name, version, resolved, p2t, gallery)
    return MaterializedPositiveSets(
        point_to_tile_ids=freeze_mapping(p2t),
        matches=freeze_mapping({pid: () for pid in p2t}),
        points_without_positives=tuple(sorted(no_positive)),
        strategy_name=strategy_name, strategy_version=version,
        resolved_params=freeze_mapping(resolved), crs=gallery.crs,
        fingerprint=fp, stats=freeze_mapping({}))


def comps(sizes) -> list:
    """Build components with distinct point ids from a list of sizes."""
    out, k = [], 0
    for i, s in enumerate(sizes):
        pids = [f"p{k + j:04d}" for j in range(s)]
        k += s
        out.append(Component(component_id=i, point_ids=pids, tile_ids=[f"t{i}"], size=s))
    return out


def write_h5(path: Path, tiles):
    """tiles: list of (tid, city, lat, lon, size_m)."""
    import h5py
    import numpy as np
    sdt = h5py.string_dtype("utf-8")
    n = len(tiles)
    with h5py.File(path, "w") as f:
        f.create_dataset("lat", data=np.array([t[2] for t in tiles], float))
        f.create_dataset("lon", data=np.array([t[3] for t in tiles], float))
        f.create_dataset("window_size_m", data=np.array([t[4] for t in tiles], float))
        f.create_dataset("tile_id", data=np.array([t[0] for t in tiles], dtype=object), dtype=sdt)
        f.create_dataset("city", data=np.array([t[1] for t in tiles], dtype=object), dtype=sdt)
        f.create_dataset("vlad", data=np.zeros((n, 4), np.float16))


def write_gt(root: Path, frames):
    """frames: list of (id, lat, lon). Writes tiny placeholder .jpg files."""
    root.mkdir(parents=True, exist_ok=True)
    for fid, lat, lon in frames:
        (root / f"{fid}_{lat}_{lon}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
