"""Materialization service: apply a Strategy to every point -> a frozen snapshot.

The snapshot is stably sorted and immutable. Downstream (components / optimizer /
audit) uses ONLY this snapshot, never the strategy, so P(q) is fixed for the run.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

from .fingerprint import compute_fingerprint
from .models import (GalleryIndex, GeoPoint, MaterializedPositiveSets, PositiveMatch,
                     freeze_mapping)
from .protocol import PositiveSelector


def materialize_positive_sets(points: Iterable[GeoPoint], gallery: GalleryIndex,
                              selector: PositiveSelector) -> MaterializedPositiveSets:
    pts = sorted(points, key=lambda p: p.point_id)

    point_to_tile_ids = {}
    matches = {}
    no_positive = []
    counts = []

    for p in pts:
        raw = selector.select(p, gallery)
        ms = _normalize(raw, gallery, selector, p)
        if ms:
            point_to_tile_ids[p.point_id] = frozenset(m.tile_id for m in ms)
            matches[p.point_id] = ms
            counts.append(len(ms))
        else:
            no_positive.append(p.point_id)

    resolved = dict(selector.resolved_config())
    fp = compute_fingerprint(selector.name, selector.version, resolved,
                             point_to_tile_ids, gallery)

    hist = dict(sorted(Counter(counts).items()))
    stats = {
        "n_points_with_positives": len(point_to_tile_ids),
        "n_points_without_positives": len(no_positive),
        "n_positive_links": int(sum(counts)),
        "unique_positive_tiles": len({t for s in point_to_tile_ids.values() for t in s}),
        "positives_per_point_hist": hist,
        "mean_positives_per_point": (sum(counts) / len(counts)) if counts else 0.0,
    }

    return MaterializedPositiveSets(
        point_to_tile_ids=freeze_mapping(point_to_tile_ids),
        matches=freeze_mapping(matches),
        points_without_positives=tuple(sorted(no_positive)),
        strategy_name=selector.name, strategy_version=selector.version,
        resolved_params=freeze_mapping(resolved), crs=gallery.crs,
        fingerprint=fp, stats=freeze_mapping(stats))


def _normalize(raw: Sequence[PositiveMatch], gallery: GalleryIndex,
               selector: PositiveSelector, point: GeoPoint):
    """Validate + stably sort a strategy's matches; dedupe by tile_id (keep best score)."""
    best = {}
    for m in raw:
        if not isinstance(m, PositiveMatch):
            raise TypeError(f"{selector.name}.select must return PositiveMatch, got {type(m)}")
        if m.tile_id not in gallery:
            raise ValueError(f"{selector.name}: tile_id {m.tile_id!r} not in gallery "
                             f"(point {point.point_id})")
        prev = best.get(m.tile_id)
        if prev is None or _score_key(m) > _score_key(prev):
            best[m.tile_id] = m
    # stable order: by tile_id (service-side deterministic sort)
    return tuple(best[t] for t in sorted(best))


def _score_key(m: PositiveMatch):
    return (m.score if m.score is not None else float("-inf"),)
