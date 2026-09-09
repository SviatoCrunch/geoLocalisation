"""Main API: build_osm_mask, OsmMaskConfig, MaskMode."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

from ._parse import parse_osm_response
from ._query import fetch_overpass
from ._rasterize import build_mask_array
from ._tags import BAND_ORDER, build_overpass_query

logger = logging.getLogger(__name__)


class MaskMode(str, Enum):
    BINARY = "binary"
    MULTI = "multi"


@dataclass
class OsmMaskConfig:
    """Configuration for build_osm_mask().

    Attributes:
        mode:              BINARY (1 band) or MULTI (12 bands).
        overpass_url:      Overpass API endpoint.
        overpass_timeout:  Query timeout passed to Overpass (seconds).
        http_timeout:      requests socket timeout (should exceed overpass_timeout).
        overwrite:         Re-download even if output or flag already exists.
    """

    mode: MaskMode = MaskMode.BINARY
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    overpass_timeout: int = 120
    http_timeout: int = 180
    overwrite: bool = False


def _write_empty_flag(out_path: Path, reason: str) -> None:
    flag = out_path.with_suffix(".flag")
    flag.write_text(f"empty: {reason}\n", encoding="utf-8")
    logger.info("No features → wrote %s  (%s)", flag.name, reason)


def _save_mask(arr: np.ndarray, out_path: Path, src_profile: dict) -> None:
    n_bands = arr.shape[0]
    profile = {
        **src_profile,
        "count": n_bands,
        "dtype": "uint8",
        "nodata": None,
        "compress": "deflate",
        "BIGTIFF": "YES",
        "SPARSE_OK": "TRUE",
    }
    profile.pop("photometric", None)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr)
        if n_bands > 1:
            for i, tag in enumerate(BAND_ORDER[:n_bands], start=1):
                dst.update_tags(i, name=tag)
    logger.info(
        "Saved mask: %s  bands=%d  shape=%s", out_path.name, n_bands, arr.shape[1:]
    )


def build_osm_mask(
    tif_path: Path | str,
    out_path: Path | str,
    config: Optional[OsmMaskConfig] = None,
) -> Optional[Path]:
    cfg = config if config is not None else OsmMaskConfig()
    tif_path = Path(tif_path)
    out_path = Path(out_path)
    flag_path = out_path.with_suffix(".flag")

    if not cfg.overwrite:
        if flag_path.exists():
            logger.info("Empty flag exists, skipping: %s", flag_path.name)
            return None
        if out_path.exists():
            logger.info("Mask exists, skipping: %s", out_path.name)
            return out_path

    with rasterio.open(tif_path) as src:
        tif_crs = src.crs.to_string()
        transform = src.transform
        shape = (src.height, src.width)
        src_profile = src.profile.copy()
        south, west, north, east = transform_bounds(
            src.crs, CRS.from_epsg(4326), *src.bounds
        )

    logger.info(
        "Mask for %s  mode=%s  bbox=S%.4f W%.4f N%.4f E%.4f",
        tif_path.name, cfg.mode.value, south, west, north, east,
    )

    query = build_overpass_query(south, west, north, east, timeout=cfg.overpass_timeout)
    data = fetch_overpass(query, url=cfg.overpass_url, http_timeout=cfg.http_timeout)

    ways = [e for e in data.get("elements", []) if e["type"] == "way"]
    if not ways:
        _write_empty_flag(out_path, "no OSM ways in bbox")
        return None

    categories = parse_osm_response(data)
    if not categories:
        _write_empty_flag(out_path, "no valid geometries after parsing")
        return None

    arr = build_mask_array(
        categories=categories,
        transform=transform,
        shape=shape,
        tif_crs=tif_crs,
        mode=cfg.mode.value,
    )
    if arr is None:
        _write_empty_flag(out_path, "geometries do not intersect pixel grid")
        return None

    _save_mask(arr, out_path, src_profile)
    return out_path
