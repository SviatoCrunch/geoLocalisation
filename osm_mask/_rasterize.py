"""Reproject and rasterize OSM geometries onto a GeoTIF grid."""
from __future__ import annotations

import logging

import numpy as np
from pyproj import Transformer
from rasterio.features import rasterize
from shapely.geometry import mapping
from shapely.ops import transform as shapely_transform

from ._tags import BAND_ORDER

logger = logging.getLogger(__name__)


def _make_projector(src_crs: str, dst_crs: str):
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    def _proj(x, y, z=None):
        return transformer.transform(x, y)

    return _proj


def _reproject(geoms: list, src_crs: str, dst_crs: str) -> list:
    if src_crs == dst_crs:
        return geoms
    proj = _make_projector(src_crs, dst_crs)
    return [shapely_transform(proj, g) for g in geoms]


def _burn(geoms: list, transform, shape: tuple[int, int]) -> np.ndarray:
    if not geoms:
        return np.zeros(shape, dtype=np.uint8)
    shapes = ((mapping(g), 1) for g in geoms)
    return rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )


def build_mask_array(
    categories: dict[str, list],
    transform,
    shape: tuple[int, int],
    tif_crs: str,
    mode: str,
    src_crs: str = "EPSG:4326",
) -> np.ndarray | None:
    reprojected = {}
    for tag, geoms in categories.items():
        if not geoms:
            continue
        reprojected[tag] = _reproject(geoms, src_crs, tif_crs)

    if not reprojected:
        return None

    if mode == "binary":
        all_geoms = [g for geoms in reprojected.values() for g in geoms]
        band = _burn(all_geoms, transform, shape)
        if not band.any():
            return None
        return band[np.newaxis, ...]

    bands = [_burn(reprojected.get(tag, []), transform, shape) for tag in BAND_ORDER]
    arr = np.stack(bands)
    if not arr.any():
        return None
    logger.info(
        "Non-empty bands: %s",
        [BAND_ORDER[i] for i, b in enumerate(bands) if b.any()],
    )
    return arr
