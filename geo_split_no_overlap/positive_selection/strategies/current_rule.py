"""current_rule — adapter for the project's production positive rule.

Reproduces the semantics documented in ``make_multicity_split``: a query is positive
for a tile iff it lies inside the tile square, generalised to
``|dx|,|dy| <= (window_size_m + query_size_m)/2`` in EPSG:3857. ``query_size_m=0``
gives the exact "point inside tile square" rule. Matching is per-city by default.

This is the first Strategy and the one covered by the parity test — its math must not
drift from the production computation.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from ..models import GalleryIndex, GeoPoint, PositiveMatch
from . import _common as C

_DEFAULTS = {"query_size_m": 0.0, "same_city_only": True}


class CurrentRuleSelector:
    name = "current_rule"
    version = "1.0"

    def __init__(self, query_size_m: float, same_city_only: bool):
        self._query_size_m = float(query_size_m)
        self._same_city_only = bool(same_city_only)

    @classmethod
    def from_params(cls, params: Mapping) -> "CurrentRuleSelector":
        p = C.resolve_params(params, _DEFAULTS, cls.name)
        q = C.require_number(p["query_size_m"], "query_size_m")
        if q < 0:
            raise ValueError("query_size_m must be >= 0")
        if not isinstance(p["same_city_only"], bool):
            raise ValueError("same_city_only must be a bool")
        return cls(query_size_m=q, same_city_only=p["same_city_only"])

    def select(self, point: GeoPoint, gallery: GalleryIndex) -> Sequence[PositiveMatch]:
        C.ensure_grid_crs(gallery)          # explicit CRS check for a geometric rule
        reach = (gallery.max_tile_size + self._query_size_m) / 2.0
        city = point.city if self._same_city_only else None
        out = []
        for t in gallery.candidates_box(city, point.x, point.y, reach):
            half = (C.tile_side(t, gallery.tile_size_fallback) + self._query_size_m) / 2.0
            if abs(t.center_x - point.x) <= half and abs(t.center_y - point.y) <= half:
                out.append(PositiveMatch(tile_id=t.tile_id, score=None, reason="box"))
        return out

    def resolved_config(self) -> Mapping[str, object]:
        return {"query_size_m": self._query_size_m, "same_city_only": self._same_city_only,
                "version": self.version, "crs_expected": C.GRID_CRS,
                "threshold_units": "grid_metres"}
