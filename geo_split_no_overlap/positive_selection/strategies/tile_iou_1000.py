"""tile_iou_1000 — positive by IoU of the FULL 1000x1000 tile vs a 1000x1000 query box.

Every gallery item is a plain 1000 x 1000 m square tile. Around each point a reference
1000 x 1000 m square is built. The tile is positive iff the IoU of the two squares
reaches ``threshold``:

    query_box = centered_square(center=q, side_m=1000)
    score     = IoU(tile_1000, query_box)
    positive iff score >= threshold

No pyramid, no sub-levels, no tile-centre crop — the full tile geometry is compared.
The 1000 m sizes are FIXED semantics of this rule (recorded in resolved_config +
fingerprint). Tiles that are not ~1000 m squares are rejected, never silently rescaled.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from ..models import GalleryIndex, GeoPoint, PositiveMatch
from . import _common as C

TILE_SIZE_M = 1000.0
QUERY_SIZE_M = 1000.0

_DEFAULTS = {"threshold": None, "geometry_tolerance_m": 1.0, "same_city_only": True}


class TileIoU1000Selector:
    name = "tile_iou_1000"
    version = "1.0"

    def __init__(self, threshold: float, geometry_tolerance_m: float, same_city_only: bool):
        self._threshold = float(threshold)
        self._tol = float(geometry_tolerance_m)
        self._same_city_only = bool(same_city_only)

    @classmethod
    def from_params(cls, params: Mapping) -> "TileIoU1000Selector":
        p = C.resolve_params(params, _DEFAULTS, cls.name)
        if p["threshold"] is None:
            raise ValueError(f"{cls.name}: 'threshold' is required")
        threshold = C.require_number(p["threshold"], "threshold", unit_interval=True)
        tol = C.require_number(p["geometry_tolerance_m"], "geometry_tolerance_m", positive=True)
        if not isinstance(p["same_city_only"], bool):
            raise ValueError("same_city_only must be a bool")
        return cls(threshold=threshold, geometry_tolerance_m=tol,
                   same_city_only=p["same_city_only"])

    def select(self, point: GeoPoint, gallery: GalleryIndex) -> Sequence[PositiveMatch]:
        C.validate_metric_crs(gallery.crs)
        query_box = C.centered_square(point.x, point.y, QUERY_SIZE_M)
        reach = (TILE_SIZE_M + QUERY_SIZE_M) / 2.0 + self._tol
        city = point.city if self._same_city_only else None

        out = []
        for t in gallery.candidates_box(city, point.x, point.y, reach):
            side = C.tile_side(t, gallery.tile_size_fallback)
            C.validate_square_size(side, TILE_SIZE_M, self._tol, what=f"tile {t.tile_id}")
            tile_box = C.centered_square(t.center_x, t.center_y, side)
            iou = C.compute_iou(tile_box, query_box)
            if iou >= self._threshold:
                out.append(PositiveMatch(tile_id=t.tile_id, score=iou, reason="iou_full_tile_1000"))
        return sorted(out, key=lambda m: m.tile_id)

    def resolved_config(self) -> Mapping[str, object]:
        return {"threshold": self._threshold, "tile_size_m": TILE_SIZE_M,
                "query_size_m": QUERY_SIZE_M, "geometry_tolerance_m": self._tol,
                "same_city_only": self._same_city_only, "version": self.version,
                "crs_expected": C.GRID_CRS, "score": "iou", "threshold_rule": ">="}
