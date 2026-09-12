"""overlap_weighted — soft, weighted positives by ground-area overlap of a 1000 m query.

Around the query a ``query_size_m`` square is built; every gallery tile whose footprint
overlaps it by more than ``min_overlap`` (fraction of the QUERY area) is a positive, with

    score = |tile ∩ query| / |query|        (the "intersection ratio")

This is the weight label used by Game4Loc's weighted-InfoNCE for partial UAV↔map matches
(Ji et al., AAAI 2025): the fraction of the query's ground area covered by the tile. On a
non-overlapping checkerboard gallery the query overlaps 1–4 cells and these ratios sum to
1 over the cells it touches (a natural target distribution); the loss normalises anyway,
so the raw ratio is what we emit here (any sharpening is a loss-side hyper-parameter).

Units: 1000 is TRUE metres; centres/points live in raw EPSG:3857 (inflated by 1/cos(lat)),
so boxes are built with grid side ``true_m / cos(lat)`` (:func:`centered_square_true`).
"""
from __future__ import annotations

from typing import Mapping, Sequence

from ..models import GalleryIndex, GeoPoint, PositiveMatch
from . import _common as C

QUERY_SIZE_M = 1000.0

_DEFAULTS = {"min_overlap": 0.05, "query_size_m": QUERY_SIZE_M, "same_city_only": True,
             "geometry_tolerance_m": 1.0}


class OverlapWeightedSelector:
    name = "overlap_weighted"
    version = "1.0"

    def __init__(self, min_overlap: float, query_size_m: float, same_city_only: bool,
                 geometry_tolerance_m: float):
        self._min_overlap = float(min_overlap)
        self._query_size_m = float(query_size_m)
        self._same_city_only = bool(same_city_only)
        self._tol = float(geometry_tolerance_m)

    @classmethod
    def from_params(cls, params: Mapping) -> "OverlapWeightedSelector":
        p = C.resolve_params(params, _DEFAULTS, cls.name)
        mo = C.require_number(p["min_overlap"], "min_overlap")
        if not (0.0 <= mo < 1.0):
            raise ValueError(f"min_overlap must be in [0, 1), got {mo}")
        qs = C.require_number(p["query_size_m"], "query_size_m", positive=True)
        tol = C.require_number(p["geometry_tolerance_m"], "geometry_tolerance_m", positive=True)
        if not isinstance(p["same_city_only"], bool):
            raise ValueError("same_city_only must be a bool")
        return cls(min_overlap=mo, query_size_m=qs, same_city_only=p["same_city_only"],
                   geometry_tolerance_m=tol)

    def select(self, point: GeoPoint, gallery: GalleryIndex) -> Sequence[PositiveMatch]:
        C.validate_metric_crs(gallery.crs)
        query_box = C.centered_square_true(point.x, point.y, point.lat, self._query_size_m)
        qa = query_box.area
        reach = C.true_m_to_grid((gallery.max_tile_size + self._query_size_m) / 2.0,
                                 point.lat) + self._tol
        city = point.city if self._same_city_only else None

        out = []
        for t in gallery.candidates_box(city, point.x, point.y, reach):
            side = C.tile_side(t, gallery.tile_size_fallback)
            tile_box = C.centered_square_true(t.center_x, t.center_y, t.lat, side)
            inter = query_box.intersection(tile_box).area
            ratio = inter / qa if qa > 0 else 0.0
            if ratio > self._min_overlap:
                out.append(PositiveMatch(tile_id=t.tile_id, score=ratio, reason="overlap_ratio"))
        return sorted(out, key=lambda m: m.tile_id)

    def resolved_config(self) -> Mapping[str, object]:
        return {"min_overlap": self._min_overlap, "query_size_m": self._query_size_m,
                "same_city_only": self._same_city_only, "geometry_tolerance_m": self._tol,
                "version": self.version, "crs_expected": C.GRID_CRS,
                "score": "intersection_ratio (|tile∩query|/|query|)", "threshold_rule": ">",
                "footprint_units": "true_metres (EPSG:3857 side scaled by 1/cos(lat))"}
