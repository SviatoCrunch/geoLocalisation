from __future__ import annotations

import io
import logging
import math
import time

logger = logging.getLogger(__name__)

import numpy as np
import requests
from PIL import Image
from rasterio.transform import Affine
from rasterio.windows import Window

from .base import GridInfo, TileSource

ESRI_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"

_EARTH_R = 6378137.0
_ORIGIN_SHIFT = math.pi * _EARTH_R
_NATIVE_TILE_PX = 256

# Web-mercator ground resolution (m/px) per Esri LOD level, on native 256px tiles.
LOD_RES: dict[int, float] = {
    z: 2 * math.pi * _EARTH_R / (_NATIVE_TILE_PX * 2 ** z) for z in range(24)
}


def _lonlat_to_mercator(lon: float, lat: float) -> tuple[float, float]:
    x = lon * math.pi / 180 * _EARTH_R
    y = _EARTH_R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def _snap(value: float, step: float, origin: float, direction: str) -> float:
    offset = value - origin
    if direction == "floor":
        n = math.floor(offset / step)
    else:
        n = math.ceil(offset / step)
    return origin + n * step


def _fetch_esri_tile(
    x_min_m: float,
    y_min_m: float,
    x_max_m: float,
    y_max_m: float,
    tile_px: int,
    retries: int = 4,
    base_delay: float = 1.0,
) -> np.ndarray:
    """Fetch one tile from Esri World Imagery export endpoint.

    Returns (tile_px, tile_px, 3) uint8 RGB array.
    """
    params = {
        "bbox": f"{x_min_m},{y_min_m},{x_max_m},{y_max_m}",
        "bboxSR": 3857,
        "imageSR": 3857,
        "size": f"{tile_px},{tile_px}",
        "format": "png32",
        "transparent": "false",
        "f": "image",
    }
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(ESRI_URL, params=params, timeout=60)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            return np.array(img)
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                delay = base_delay * 1.7 ** attempt
                logger.warning(
                    "Esri retry: attempt=%d  err=%s  sleep=%.1fs",
                    attempt + 1, exc, delay,
                )
                time.sleep(delay)
    raise RuntimeError(f"Esri tile fetch failed after {retries} retries") from last_err


class EsriSource(TileSource):
    """Esri World Imagery tile source (ArcGIS MapServer export, no API key).

    Tiles are snapped to Esri's native LOD grid so the server returns
    pre-cached tiles rather than re-rendering them.

    Zoom / GSD reference:
        zoom 18 → 0.597 m/px  (default)
        zoom 19 → 0.299 m/px
        zoom 20 → 0.149 m/px
    """

    def __init__(
        self,
        zoom: int = 18,
        tile_px: int = 2048,
        retries: int = 4,
        tile_sleep: float = 0.35,
    ) -> None:
        if tile_px > 4096:
            raise ValueError(f"tile_px={tile_px} exceeds Esri server limit of 4096")
        self._zoom = zoom
        self._tile_px = tile_px
        self._retries = retries
        self._tile_sleep = tile_sleep
        self._grid_state = {}

    def build_grid(self, aoi_points: list) -> GridInfo:
        res_m = LOD_RES[self._zoom]
        step_m = self._tile_px * res_m
        ox = oy = -_ORIGIN_SHIFT

        xs, ys = zip(*[_lonlat_to_mercator(lon, lat) for lat, lon in aoi_points])

        x_min, y_min = min(xs), min(ys)
        x_max, y_max = max(xs), max(ys)

        x0 = _snap(x_min, step_m, ox, "floor")
        y0 = _snap(y_min, step_m, oy, "floor")
        x1 = _snap(x_max, step_m, ox, "ceil")
        y1 = _snap(y_max, step_m, oy, "ceil")

        cols = round((x1 - x0) / step_m)
        rows = round((y1 - y0) / step_m)

        transform = Affine(res_m, 0, x0, 0, -res_m, y1)

        self._grid_state = {
            "x0": x0,
            "y0": y0,
            "y1": y1,
            "cols": cols,
            "rows": rows,
            "step_m": step_m,
        }

        jobs = [(col, row) for row in range(rows) for col in range(cols)]

        return GridInfo(
            width_px=cols * self._tile_px,
            height_px=rows * self._tile_px,
            transform=transform,
            crs="EPSG:3857",
            tile_px=self._tile_px,
            jobs=jobs,
            meta={
                "source": "esri",
                "zoom": self._zoom,
                "tile_px": self._tile_px,
                "x0": x0,
                "y0": y0,
                "y1": y1,
                "cols": cols,
                "rows": rows,
                "step_m": step_m,
                "res_m": res_m,
            },
        )

    def fetch_tile(self, job) -> tuple:
        col, row = job
        g = self._grid_state
        x_min_m = g["x0"] + col * g["step_m"]
        x_max_m = x_min_m + g["step_m"]
        y_max_m = g["y1"] - row * g["step_m"]
        y_min_m = y_max_m - g["step_m"]

        arr_hwc = _fetch_esri_tile(
            x_min_m,
            y_min_m,
            x_max_m,
            y_max_m,
            tile_px=self._tile_px,
            retries=self._retries,
        )
        if self._tile_sleep > 0:
            time.sleep(self._tile_sleep)
        return job, arr_hwc.transpose(2, 0, 1)

    def job_to_window(self, job) -> Window:
        col, row = job
        return Window(
            col_off=col * self._tile_px,
            row_off=row * self._tile_px,
            width=self._tile_px,
            height=self._tile_px,
        )

    def load_done_jobs(self, meta: dict) -> set:
        return {tuple(t) for t in meta.get("done_tiles", [])}

    def serialise_done_jobs(self, done: set) -> list:
        return [list(t) for t in done]
