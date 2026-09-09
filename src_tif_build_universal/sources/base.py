from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List

import numpy as np
from rasterio.transform import Affine
from rasterio.windows import Window


@dataclass
class GridInfo:
    width_px: int
    height_px: int
    transform: Affine
    crs: str
    tile_px: int
    jobs: List[Any] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


class TileSource(ABC):
    """Abstract base for tile data sources.

    Implementors supply grid computation and tile fetching.
    The threading loop, buffer, and mosaic logic are source-agnostic.
    """

    @abstractmethod
    def build_grid(self, aoi_points: list) -> GridInfo:
        """Compute the tile grid for aoi_points and return it.

        May store internal state (e.g. grid origin) needed by
        fetch_tile and job_to_window.
        """

    @abstractmethod
    def fetch_tile(self, job: Any) -> tuple[Any, np.ndarray]:
        """Fetch one tile.

        Returns (job, rgb_array) where rgb_array is (3, H, W) uint8.
        Must be safe to call from multiple threads.
        """

    @abstractmethod
    def job_to_window(self, job: Any) -> Window:
        """Convert job to a rasterio Window in the output TIF."""

    @abstractmethod
    def load_done_jobs(self, meta: dict) -> set:
        """Deserialise done job IDs from sidecar metadata dict."""

    @abstractmethod
    def serialise_done_jobs(self, done: set) -> list:
        """Serialise done job IDs for JSON sidecar metadata."""
