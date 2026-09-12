"""nearest_tile — a query's positive is the SINGLE gallery tile whose centre is nearest
the point (same-city by default).

On a non-overlapping "checkerboard" gallery (one tile per grid cell) the tile centres
tessellate the plane, so the nearest centre IS the cell the query falls in. Unlike a
footprint ``contains_point`` test, this is robust to sub-cell centre offsets (a kept tile
may sit up to half a source-stride off its cell centre after sub-sampling), which would
otherwise make ``contains_point`` miss queries near a cell boundary. Exactly one positive
per point — or none, when ``max_dist_m`` is set and the nearest tile is farther than that
(a guard for queries outside the mapped area).
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

from ..models import GalleryIndex, GeoPoint, PositiveMatch
from . import _common as C

_DEFAULTS = {"same_city_only": True, "max_dist_m": None}


class NearestTileSelector:
    name = "nearest_tile"
    version = "1.0"

    def __init__(self, same_city_only: bool, max_dist_m: Optional[float]):
        self._same_city_only = bool(same_city_only)
        self._max_dist_m = max_dist_m

    @classmethod
    def from_params(cls, params: Mapping) -> "NearestTileSelector":
        p = C.resolve_params(params, _DEFAULTS, cls.name)
        if not isinstance(p["same_city_only"], bool):
            raise ValueError("same_city_only must be a bool")
        md = p["max_dist_m"]
        if md is not None:
            md = C.require_number(md, "max_dist_m", positive=True)
        return cls(same_city_only=p["same_city_only"], max_dist_m=md)

    def select(self, point: GeoPoint, gallery: GalleryIndex) -> Sequence[PositiveMatch]:
        C.ensure_grid_crs(gallery)                       # nearest-centre is a metric rule
        tiles = (gallery.tiles_for_city(point.city) if self._same_city_only
                 else gallery.all_tiles())
        best = None                                      # (dist_grid, tile_id)
        for t in tiles:
            dx = t.center_x - point.x
            dy = t.center_y - point.y
            d = (dx * dx + dy * dy) ** 0.5
            cand = (d, t.tile_id)
            if best is None or cand < best:              # ties broken by tile_id (deterministic)
                best = cand
        if best is None:
            return []
        d_true = C.true_dist_m(best[0], point.lat)
        if self._max_dist_m is not None and d_true > self._max_dist_m:
            return []
        return [PositiveMatch(tile_id=best[1], score=-d_true, reason="nearest")]

    def resolved_config(self) -> Mapping[str, object]:
        return {"same_city_only": self._same_city_only, "max_dist_m": self._max_dist_m,
                "version": self.version, "crs_expected": C.GRID_CRS,
                "threshold_units": "true_metres"}
