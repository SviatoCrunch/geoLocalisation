"""Integration with ``geo_split_no_overlap``: build a training RelevanceTable from a split.

This is the ONLY place that imports the split package. It reuses that package's
positive-selection subsystem so training **positives are identical to the split's**
(same config, same strategy, fingerprint-consistent), then reads ``split.json`` to pick
the ``train``/``val``/``test`` queries. Safe negatives (and ignore) are geometry —
the one thing the split doesn't provide:

    positives  = split snapshot (point_id -> tile_ids), mapped to tile rows
    safe        = tiles whose footprint does NOT overlap the query footprint (inter <= eps)
    ignore      = overlaps but not a positive (the rest)

Nothing here touches a model. The result feeds ``batching`` + ``loss`` unchanged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..relevance import ExplicitRelevanceTable


@dataclass
class SplitRelevance:
    which: str                 # "train" | "val" | "test"
    relevance: ExplicitRelevanceTable
    query_ids: list            # order matches q_xy rows
    q_xy: np.ndarray           # (Nq, 2) EPSG:3857
    tile_ids: list             # tile-row order (row -> tile_id)
    tile_xy: np.ndarray        # (M, 2) EPSG:3857
    gallery: object            # geo_split_no_overlap GalleryIndex
    snapshot: object           # MaterializedPositiveSets
    fingerprint: str


def _load_inputs(cfg):
    """(gallery, points, snapshot) for single-gallery or per-city merged mode."""
    from geo_split_no_overlap.positive_selection import (build_gallery_index, build_geo_points,
                                                         create_positive_selector,
                                                         materialize_positive_sets)
    if cfg.is_per_city():
        from geo_split_no_overlap.multicity import load_merged
        return load_merged(cfg)
    gallery = build_gallery_index(cfg.tiles_h5, cfg.grid_crs, cfg.tile_size_m)
    points = build_geo_points(cfg.gt, cfg.grid_crs)
    selector = create_positive_selector(cfg.positive_selection_config())
    mps = materialize_positive_sets(points, gallery, selector)
    return gallery, points, mps


def build_split_relevance(config_path, split_json_path, which: str = "train", *,
                          query_size_m: float = 1000.0, safe_eps_area: float = 0.0) -> SplitRelevance:
    """Build a :class:`SplitRelevance` for one split by reusing the split's positives."""
    from geo_split_no_overlap.config import load_config

    cfg = load_config(config_path)
    gallery, points, mps = _load_inputs(cfg)

    tiles = gallery.all_tiles()                       # sorted by tile_id -> stable row order
    tile_ids = [t.tile_id for t in tiles]
    row_of = {tid: i for i, tid in enumerate(tile_ids)}
    tile_xy = np.array([[t.center_x, t.center_y] for t in tiles], float).reshape(-1, 2)
    tile_side = np.array([t.size_m if (t.size_m == t.size_m and t.size_m > 0) else cfg.tile_size_m
                          for t in tiles], float)
    half = (tile_side + float(query_size_m)) / 2.0    # per-tile overlap half-extent

    data = json.loads(Path(split_json_path).expanduser().read_text(encoding="utf-8"))
    want = set(data.get(which, []))
    pt_by_id = {p.point_id: p for p in points}

    q_ids, q_xy, pos_rows, safe_rows, pos_weights = [], [], [], [], []
    any_weight = False                                  # carry weights only if the strategy scored
    for pid in sorted(want):
        if pid not in mps.point_to_tile_ids:          # no positive -> not a training query
            continue
        p = pt_by_id[pid]
        tiles_sorted = sorted(mps.point_to_tile_ids[pid], key=lambda t: row_of[t])
        pr = np.array([row_of[t] for t in tiles_sorted], np.int64)
        score_by_tile = {m.tile_id: m.score for m in mps.matches.get(pid, ())}
        wr = np.array([1.0 if score_by_tile.get(t) is None else float(score_by_tile[t])
                       for t in tiles_sorted], np.float64)
        if any(score_by_tile.get(t) is not None for t in tiles_sorted):
            any_weight = True
        dx = np.abs(tile_xy[:, 0] - p.x)
        dy = np.abs(tile_xy[:, 1] - p.y)
        inter = np.clip(half - dx, 0.0, None) * np.clip(half - dy, 0.0, None)
        safe = np.nonzero(inter <= safe_eps_area)[0].astype(np.int64)
        safe = np.setdiff1d(safe, pr, assume_unique=False)   # pos wins over safe
        q_ids.append(pid)
        q_xy.append([p.x, p.y])
        pos_rows.append(pr)
        pos_weights.append(wr)
        safe_rows.append(safe)

    q_xy = np.array(q_xy, float).reshape(-1, 2)
    rel = ExplicitRelevanceTable(q_ids, pos_rows, safe_rows,
                                 pos_weights=(pos_weights if any_weight else None))
    return SplitRelevance(which=which, relevance=rel, query_ids=q_ids, q_xy=q_xy,
                          tile_ids=tile_ids, tile_xy=tile_xy, gallery=gallery, snapshot=mps,
                          fingerprint=mps.fingerprint)


def build_pairs_from_split(sr: SplitRelevance):
    """Convenience: canonical query→tile pairs for a :class:`SplitRelevance`."""
    from ..batching import build_pair_pool
    return build_pair_pool(sr.relevance, sr.q_xy, sr.tile_xy)
