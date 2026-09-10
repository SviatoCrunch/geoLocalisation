"""contains_point — a query is positive for a tile iff the point lies inside the tile
square (``|dx|,|dy| <= window_size_m/2``). A minimal, real geometric rule; it doubles
as the simple alternative used to prove that swapping strategies via the Registry
changes P(q) without touching any split code.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from ..models import GalleryIndex, GeoPoint, PositiveMatch
from . import _common as C

_DEFAULTS = {"same_city_only": True}


class ContainsPointSelector:
    name = "contains_point"
    version = "1.0"

    def __init__(self, same_city_only: bool):
        self._same_city_only = bool(same_city_only)

    @classmethod
    def from_params(cls, params: Mapping) -> "ContainsPointSelector":
        p = C.resolve_params(params, _DEFAULTS, cls.name)
        if not isinstance(p["same_city_only"], bool):
            raise ValueError("same_city_only must be a bool")
        return cls(same_city_only=p["same_city_only"])

    def select(self, point: GeoPoint, gallery: GalleryIndex) -> Sequence[PositiveMatch]:
        C.ensure_grid_crs(gallery)          # explicit CRS check for a geometric rule
        reach = gallery.max_tile_size / 2.0
        city = point.city if self._same_city_only else None
        out = []
        for t in gallery.candidates_box(city, point.x, point.y, reach):
            half = C.tile_side(t, gallery.tile_size_fallback) / 2.0
            if abs(t.center_x - point.x) <= half and abs(t.center_y - point.y) <= half:
                out.append(PositiveMatch(tile_id=t.tile_id, score=None, reason="contains"))
        return out

    def resolved_config(self) -> Mapping[str, object]:
        return {"same_city_only": self._same_city_only, "version": self.version,
                "crs_expected": C.GRID_CRS, "threshold_units": "grid_metres"}
