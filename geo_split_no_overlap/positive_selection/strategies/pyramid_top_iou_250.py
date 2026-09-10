"""pyramid_top_iou_250 — positive by IoU of the pyramid TOP (250x250) vs a 250x250 box.

One gallery item represents a whole concentric-square pyramid: base 1000 x 1000 m,
top 250 x 250 m, all levels sharing one centre. Positivity is decided ONLY by the
250 m top (intermediate levels are irrelevant and are not generated):

    query_box   = centered_square(center=q,             side_m=250)
    pyramid_top = centered_square(center=pyramid.centre, side_m=250)
    score       = IoU(pyramid_top, query_box)
    positive iff score >= threshold

On a positive, the id of the WHOLE pyramid (the gallery ``tile_id``) is returned — the
top is only a probe geometry. IoU is NOT taken against the 1000 m base, any middle
level, base-vs-250, or as a max over levels.

Concentricity policy: this gallery model (``GalleryTile``) stores only the base
geometry, so the 250 m top is always DERIVED concentric with the validated 1000 m base
centre. There is no independent/explicit top geometry to accept; if one were added to
the model it would have to be concentricity- and size-checked before use (never
silently corrected). The base is validated to be a ~1000 m square first.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from ..models import GalleryIndex, GeoPoint, PositiveMatch
from . import _common as C

PYRAMID_BASE_SIZE_M = 1000.0
PYRAMID_TOP_SIZE_M = 250.0
QUERY_SIZE_M = 250.0

_DEFAULTS = {"threshold": None, "geometry_tolerance_m": 1.0, "same_city_only": True}


class PyramidTopIoU250Selector:
    name = "pyramid_top_iou_250"
    version = "1.0"

    def __init__(self, threshold: float, geometry_tolerance_m: float, same_city_only: bool):
        self._threshold = float(threshold)
        self._tol = float(geometry_tolerance_m)
        self._same_city_only = bool(same_city_only)

    @classmethod
    def from_params(cls, params: Mapping) -> "PyramidTopIoU250Selector":
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
        reach = (PYRAMID_TOP_SIZE_M + QUERY_SIZE_M) / 2.0 + self._tol
        city = point.city if self._same_city_only else None

        out = []
        for t in gallery.candidates_box(city, point.x, point.y, reach):
            base_side = C.tile_side(t, gallery.tile_size_fallback)
            C.validate_square_size(base_side, PYRAMID_BASE_SIZE_M, self._tol,
                                   what=f"pyramid base {t.tile_id}")
            # top derived concentric with the validated base centre
            top_box = C.centered_square(t.center_x, t.center_y, PYRAMID_TOP_SIZE_M)
            iou = C.compute_iou(top_box, query_box)
            if iou >= self._threshold:
                out.append(PositiveMatch(tile_id=t.tile_id, score=iou,
                                         reason="iou_pyramid_top_250"))
        return sorted(out, key=lambda m: m.tile_id)

    def resolved_config(self) -> Mapping[str, object]:
        return {"threshold": self._threshold,
                "pyramid_base_size_m": PYRAMID_BASE_SIZE_M,
                "pyramid_top_size_m": PYRAMID_TOP_SIZE_M,
                "query_size_m": QUERY_SIZE_M, "geometry_tolerance_m": self._tol,
                "same_city_only": self._same_city_only, "version": self.version,
                "crs_expected": C.GRID_CRS, "score": "iou", "threshold_rule": ">="}
