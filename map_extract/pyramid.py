"""Concentric context-pyramid sampler over a single EPSG:3857 GeoTIFF (vendored).

Trimmed port of RevisitAnything's ``src_tif_data/torchgeo_tif.py`` keeping only what the
extractor uses: :class:`PyramidConfig`, :class:`PyramidLevel`, :class:`PyramidTile`,
:class:`PyramidContextSampler`. The ``torchgeo`` import is gone (grid uses the local
:class:`~map_extract.fishnet.BoundingBox`); ``rasterio`` / ``pyproj`` are imported lazily
inside the sampler so ``PyramidConfig`` (a pure dataclass) imports with numpy + stdlib
only, which the unit tests rely on.

Sizes in the config are in the raster CRS units (EPSG:3857). To get TRUE-metre footprints
build the config via :func:`map_extract.geometry.scaled_pyramid_config`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .fishnet import grid_bboxes, latlon_to_mercator_polygon

_REMOTE_PREFIXES = ("http://", "https://", "s3://", "/vsicurl/", "/vsis3/")


def _is_remote(path: str) -> bool:
    return any(path.startswith(p) for p in _REMOTE_PREFIXES)


@dataclass
class PyramidConfig:
    """Grid + concentric-pyramid shape. ``tile_size_m``/``stride_m`` in raster CRS units.

    Each level is a concentric window scaled by ``scale_factor ** level`` relative to the
    base (level 0); every level is resized to ``output_size_px``.
    """

    tile_size_m: float
    stride_m: Optional[float] = None
    levels: int = 3
    scale_factor: float = 4.0
    output_size_px: int = 256

    @classmethod
    def from_apex(cls, apex_window_m: float, levels: int = 3, scale_factor: float = 2.0,
                  stride_m=None, output_size_px: int = 256) -> "PyramidConfig":
        """Config specified by the level-0 (smallest) window."""
        return cls(tile_size_m=apex_window_m, stride_m=stride_m, levels=levels,
                   scale_factor=scale_factor, output_size_px=output_size_px)

    @classmethod
    def from_base(cls, base_window_m: float, levels: int = 3, scale_factor: float = 2.0,
                  stride_m=None, output_size_px: int = 256) -> "PyramidConfig":
        """Config specified by the last (largest) window; back-computes tile_size_m."""
        return cls(tile_size_m=base_window_m / (scale_factor ** (levels - 1)),
                   stride_m=stride_m, levels=levels, scale_factor=scale_factor,
                   output_size_px=output_size_px)

    def window_size_m(self, level: int) -> float:
        """Window side length (raster CRS units) for *level*."""
        return self.tile_size_m * (self.scale_factor ** level)

    def all_window_sizes(self) -> List[float]:
        return [self.window_size_m(lvl) for lvl in range(self.levels)]

    @property
    def safe_margin_m(self) -> float:
        """Half the outermost window — inset so no level runs off the raster edge."""
        return self.window_size_m(self.levels - 1) / 2.0


@dataclass
class PyramidLevel:
    level: int
    window_size_m: float
    image: np.ndarray
    bbox_mercator: Dict[str, float]
    bbox_latlon: Dict[str, float]


@dataclass
class PyramidTile:
    index: int
    center_lat: float
    center_lon: float
    config: PyramidConfig
    levels: Dict[int, PyramidLevel] = field(default_factory=dict)
    pyr_idx: int = 0

    @property
    def center(self) -> Tuple[float, float]:
        return self.center_lat, self.center_lon

    def as_array(self) -> np.ndarray:
        order = sorted(self.levels.keys())
        return np.stack([self.levels[lvl].image for lvl in order], axis=0)


class PyramidContextSampler:
    """Tile a single EPSG:3857 GeoTIFF into concentric context pyramids.

    Tile centres are placed on an exact ``stride_m`` metric grid; any centre whose
    outermost window would leave the raster is dropped (never nudged), so tiles are free
    of black-border padding. Use as a context manager to keep one file handle open.
    """

    def __init__(self, geotiff_path: Union[str, Path], config: PyramidConfig,
                 filter_points_latlon: Optional[np.ndarray] = None,
                 min_intersection_ratio: float = 0.3) -> None:
        import rasterio
        from pyproj import Transformer

        self._raw_path = str(geotiff_path)
        self.geotiff_path = None if _is_remote(self._raw_path) else Path(geotiff_path)
        self.config = config
        self._filter_points_latlon = filter_points_latlon
        self._min_intersection_ratio = min_intersection_ratio
        self._src = None

        grid_size_m, stride_m = self._grid_size_and_stride_m(config)

        with rasterio.open(self._raw_path) as src:
            bounds = src.bounds

        aoi_polygon_m = (latlon_to_mercator_polygon(filter_points_latlon)
                         if filter_points_latlon is not None else None)

        self._bboxes = grid_bboxes(
            bounds, tile_size_m=grid_size_m, stride_m=stride_m,
            safe_margin_m=config.safe_margin_m, aoi_polygon=aoi_polygon_m,
            min_ratio=min_intersection_ratio,
        )

        self._to_latlon = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

        sizes = [f"{s:.0f} m" for s in config.all_window_sizes()]
        print(f"{type(self).__name__} ready: {len(self._bboxes)} tiles | "
              f"{config.levels} levels | windows: {sizes} | "
              f"grid: {grid_size_m:.0f} window, {stride_m:.0f} stride (CRS units) | "
              f"safe margin: {config.safe_margin_m:.0f}")

    def __enter__(self) -> "PyramidContextSampler":
        import rasterio
        self._src = rasterio.open(self._raw_path)
        return self

    def __exit__(self, *_) -> None:
        if self._src is not None:
            self._src.close()
            self._src = None

    def __len__(self) -> int:
        return len(self._bboxes)

    def __getitem__(self, index: int) -> PyramidTile:
        return self.get_tile(index)

    def get_tile(self, index: int, levels: Optional[List[int]] = None) -> PyramidTile:
        if not (0 <= index < len(self._bboxes)):
            raise IndexError(f"Index {index} out of range [0, {len(self._bboxes) - 1}]")

        base_bbox = self._bboxes[index]
        center_x = (base_bbox.minx + base_bbox.maxx) / 2
        center_y = (base_bbox.miny + base_bbox.maxy) / 2
        center_lon, center_lat = self._to_latlon.transform(center_x, center_y)

        if levels is None:
            levels = list(range(self.config.levels))

        tile = PyramidTile(index=index, center_lat=center_lat, center_lon=center_lon,
                           config=self.config)
        for lvl in levels:
            if not (0 <= lvl < self.config.levels):
                raise ValueError(f"Level {lvl} out of range [0, {self.config.levels - 1}]")
            tile.levels[lvl] = self._read_level(center_x, center_y, lvl)
        return tile

    def iter_tiles(self, levels: Optional[List[int]] = None, start: int = 0,
                   stop: Optional[int] = None):
        if stop is None:
            stop = len(self._bboxes)
        for i in range(start, stop):
            yield self.get_tile(i, levels=levels)

    # subclass hook (kept for parity with the source)
    def _grid_size_and_stride_m(self, config: PyramidConfig) -> Tuple[float, float]:
        grid_size_m = config.window_size_m(config.levels - 1)
        stride_m = config.stride_m if config.stride_m is not None else grid_size_m
        return grid_size_m, stride_m

    def _build_bbox_mercator(self, center_x: float, center_y: float,
                             level: int) -> Dict[str, float]:
        half = self.config.window_size_m(level) / 2.0
        return {"minx": center_x - half, "maxx": center_x + half,
                "miny": center_y - half, "maxy": center_y + half}

    def _bbox_to_latlon(self, bbox_mercator: Dict[str, float]) -> Dict[str, float]:
        west, south = self._to_latlon.transform(bbox_mercator["minx"], bbox_mercator["miny"])
        east, north = self._to_latlon.transform(bbox_mercator["maxx"], bbox_mercator["maxy"])
        return {"west": west, "east": east, "south": south, "north": north}

    def _read_level(self, center_x: float, center_y: float, level: int) -> PyramidLevel:
        import rasterio
        if self._src is not None:
            return self._read_from(self._src, center_x, center_y, level)
        with rasterio.open(self._raw_path) as src:
            return self._read_from(src, center_x, center_y, level)

    def _read_from(self, src, center_x: float, center_y: float, level: int) -> PyramidLevel:
        from rasterio.enums import Resampling
        from rasterio.windows import from_bounds as window_from_bounds

        bbox_mercator = self._build_bbox_mercator(center_x, center_y, level)
        px = self.config.output_size_px
        window = window_from_bounds(
            bbox_mercator["minx"], bbox_mercator["miny"],
            bbox_mercator["maxx"], bbox_mercator["maxy"], src.transform,
        )
        arr = src.read([1, 2, 3], window=window, out_shape=(3, px, px),
                       resampling=Resampling.bilinear, boundless=True, fill_value=0)
        image = arr.transpose(1, 2, 0).astype(np.uint8)
        return PyramidLevel(level=level, window_size_m=self.config.window_size_m(level),
                            image=image, bbox_mercator=bbox_mercator,
                            bbox_latlon=self._bbox_to_latlon(bbox_mercator))
