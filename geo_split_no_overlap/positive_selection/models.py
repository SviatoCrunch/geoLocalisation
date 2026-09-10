"""Domain models for the positive-selection subsystem.

These are the ONLY types crossing the subsystem boundary. The split orchestration
(components / optimizer / audit) consumes :class:`MaterializedPositiveSets` and
:class:`GalleryIndex`; it never sees a concrete strategy. Geometry is carried in the
tile-grid CRS (EPSG:3857): tile/point coordinates are pre-projected so strategies do
no projection themselves.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class GeoPoint:
    """A query point with a stable id, pre-projected into the grid CRS."""
    point_id: str
    city: str
    lat: float
    lon: float
    x: float                 # grid-CRS easting (EPSG:3857)
    y: float                 # grid-CRS northing


@dataclass(frozen=True)
class GalleryTile:
    """One gallery tile footprint, pre-projected into the grid CRS."""
    tile_id: str
    city: str
    center_x: float
    center_y: float
    lat: float
    lon: float
    size_m: float            # square side in grid units (may be NaN -> fallback used)


@dataclass(frozen=True)
class PositiveMatch:
    """A single (point, tile) positive with optional diagnostics."""
    tile_id: str
    score: Optional[float] = None
    reason: Optional[str] = None


class GalleryIndex:
    """Immutable, queryable view over gallery tiles.

    Exposes the CRS + area units so geometric strategies can *verify* them, and a
    cheap axis-aligned bbox candidate query (per city) so strategies avoid scanning
    the whole gallery. Tiles are keyed by ``tile_id``.
    """

    def __init__(self, tiles: Iterable[GalleryTile], crs: str, area_units: str,
                 tile_size_fallback: float):
        tiles = list(tiles)
        self._tiles = {t.tile_id: t for t in tiles}
        if len(self._tiles) != len(tiles):
            raise ValueError("duplicate tile_id in gallery")
        self.crs = str(crs)
        self.area_units = str(area_units)
        self.tile_size_fallback = float(tile_size_fallback)

        self._by_city = defaultdict(list)
        for t in tiles:
            self._by_city[t.city].append(t)
        # per-city numpy arrays for fast bbox candidate queries
        self._city_xy = {}
        for city, ts in self._by_city.items():
            self._city_xy[city] = (
                [t.tile_id for t in ts],
                np.array([t.center_x for t in ts], float),
                np.array([t.center_y for t in ts], float),
            )
        sizes = [t.size_m for t in tiles if t.size_m == t.size_m and t.size_m > 0]
        self.max_tile_size = max(sizes) if sizes else float(tile_size_fallback)

    def __len__(self) -> int:
        return len(self._tiles)

    def __contains__(self, tile_id: str) -> bool:
        return tile_id in self._tiles

    def get(self, tile_id: str) -> GalleryTile:
        return self._tiles[tile_id]

    def all_tiles(self) -> Tuple[GalleryTile, ...]:
        return tuple(self._tiles[k] for k in sorted(self._tiles))

    def tiles_for_city(self, city: str) -> Tuple[GalleryTile, ...]:
        return tuple(self._by_city.get(city, ()))

    def cities(self) -> Tuple[str, ...]:
        return tuple(sorted(self._by_city))

    def candidates_box(self, city, x: float, y: float, reach: float) -> Tuple[GalleryTile, ...]:
        """Tiles whose centre lies within [x±reach, y±reach]. ``city=None`` -> all cities."""
        cities = [city] if city is not None else list(self._city_xy)
        out = []
        for c in cities:
            entry = self._city_xy.get(c)
            if entry is None:
                continue
            ids, cx, cy = entry
            m = (np.abs(cx - x) <= reach) & (np.abs(cy - y) <= reach)
            for i in np.nonzero(m)[0]:
                out.append(self._tiles[ids[int(i)]])
        return tuple(out)


@dataclass(frozen=True)
class MaterializedPositiveSets:
    """Immutable snapshot of P(q) — the ONLY thing the split algorithm consumes.

    ``point_to_tile_ids`` holds only points with >=1 positive; points with none are
    listed in ``points_without_positives`` (the split layer applies its policy). Once
    built, this is frozen: components / optimizer / audit read it, never re-run a
    strategy, so P(q) cannot change mid-run.
    """
    point_to_tile_ids: Mapping[str, frozenset]
    matches: Mapping[str, Tuple[PositiveMatch, ...]]
    points_without_positives: Tuple[str, ...]
    strategy_name: str
    strategy_version: str
    resolved_params: Mapping[str, object]
    crs: str
    fingerprint: str
    stats: Mapping[str, object]

    def tile_ids(self) -> frozenset:
        """Union of all positive tile ids across every point."""
        out = set()
        for s in self.point_to_tile_ids.values():
            out |= s
        return frozenset(out)

    def point_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self.point_to_tile_ids))


def freeze_mapping(d: dict) -> Mapping:
    """Wrap a dict read-only (defensive immutability for the snapshot)."""
    return MappingProxyType(dict(d))
