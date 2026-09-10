"""Geometric conflict detection between positive gallery tiles.

Two tiles conflict when their square footprints overlap with true area
> ``area_epsilon_m2``. Footprints are axis-aligned squares in the grid CRS
(EPSG:3857); intersection area is exact for axis-aligned rectangles and converted to
true m^2 with the cos^2(lat) conformal factor. An STRtree prunes candidate pairs.

Consumes tile geometry via a ``positive_selection.GalleryIndex`` (public model) — NOT
any concrete strategy. Same-tile sharing needs no geometry (captured because both
points reference the same tile_id downstream).
"""
from __future__ import annotations

import math

from .positive_selection import GalleryIndex
from .union_find import UnionFind


def tile_bounds(tile, tile_size_fallback: float) -> tuple:
    """(minx, miny, maxx, maxy) square footprint in grid units."""
    s = tile.size_m if (tile.size_m == tile.size_m and tile.size_m > 0) else tile_size_fallback
    h = s / 2.0
    return (tile.center_x - h, tile.center_y - h, tile.center_x + h, tile.center_y + h)


def rect_intersection_area(a: tuple, b: tuple) -> float:
    """Intersection area of two (minx,miny,maxx,maxy) rectangles (0 for touch/disjoint)."""
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if dx <= 0.0 or dy <= 0.0:
        return 0.0
    return dx * dy


def true_m2(area_grid: float, lat_deg: float) -> float:
    c = math.cos(math.radians(lat_deg))
    return area_grid * c * c


def overlap_true_m2(a, b, tile_size_fallback: float) -> float:
    area = rect_intersection_area(tile_bounds(a, tile_size_fallback),
                                  tile_bounds(b, tile_size_fallback))
    if area <= 0.0:
        return 0.0
    return true_m2(area, 0.5 * (a.lat + b.lat))


def build_tile_clusters(gallery: GalleryIndex, tile_ids, area_epsilon_m2: float,
                        tile_size_fallback: float) -> tuple:
    """Union-find over the given tile ids by geometric overlap > area_epsilon.

    Returns ``(cluster_of, n_clusters, n_overlap_edges)`` with ``cluster_of`` mapping
    each tile_id -> a stable 0-based cluster id.
    """
    ids = sorted(tile_ids)
    idx_of = {t: i for i, t in enumerate(ids)}
    uf = UnionFind(len(ids))
    n_edges = 0

    tiles = {t: gallery.get(t) for t in ids}
    bounds = {t: tile_bounds(tiles[t], tile_size_fallback) for t in ids}

    for t1, t2 in _candidate_pairs(ids, bounds):
        if overlap_true_m2(tiles[t1], tiles[t2], tile_size_fallback) > area_epsilon_m2:
            uf.union(idx_of[t1], idx_of[t2])
            n_edges += 1

    comps = uf.components()
    cluster_of = {}
    for cid, members in enumerate(comps):
        for i in members:
            cluster_of[ids[i]] = cid
    return cluster_of, len(comps), n_edges


def _candidate_pairs(ids, bounds):
    """Yield unordered candidate (id1<id2) pairs whose bboxes may overlap (STRtree)."""
    try:
        from shapely import STRtree, box
    except Exception:                            # pragma: no cover - shapely is declared
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                yield ids[i], ids[j]
        return

    geoms = [box(*bounds[t]) for t in ids]
    tree = STRtree(geoms)
    seen = set()
    for i, g in enumerate(geoms):
        for j in tree.query(g):
            j = int(j)
            if j == i:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            yield ids[a], ids[b]
