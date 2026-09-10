"""Lift tile-overlap clusters to indivisible point components.

Consumes the frozen :class:`MaterializedPositiveSets` snapshot + a ``GalleryIndex``
(both public models of the positive subsystem) — never a strategy. Two points
conflict iff they touch the same tile cluster (same tile, or geometrically
overlapping tiles); a point spanning several clusters bridges them. Transitivity is
automatic (single union-find over points).
"""
from __future__ import annotations

from collections import defaultdict

from .positive_selection import GalleryIndex, MaterializedPositiveSets
from .schemas import Component
from .spatial_conflicts import build_tile_clusters
from .union_find import UnionFind


def build_components(gallery: GalleryIndex, mps: MaterializedPositiveSets,
                    area_epsilon_m2: float, tile_size_fallback: float) -> list:
    cluster_of, _, _ = build_tile_clusters(gallery, mps.tile_ids(),
                                           area_epsilon_m2, tile_size_fallback)

    point_ids = sorted(mps.point_to_tile_ids)
    idx_of = {pid: i for i, pid in enumerate(point_ids)}
    uf = UnionFind(len(point_ids))

    cluster_to_points = defaultdict(list)
    for pid in point_ids:
        for cl in {cluster_of[t] for t in mps.point_to_tile_ids[pid]}:
            cluster_to_points[cl].append(idx_of[pid])
    for members in cluster_to_points.values():
        first = members[0]
        for other in members[1:]:
            uf.union(first, other)

    components = []
    for cid, member_idx in enumerate(uf.components()):
        pids = sorted(point_ids[i] for i in member_idx)
        tile_ids = sorted({t for pid in pids for t in mps.point_to_tile_ids[pid]})
        components.append(Component(component_id=cid, point_ids=pids,
                                    tile_ids=tile_ids, size=len(pids)))
    return components


def size_histogram(components) -> dict:
    hist = defaultdict(int)
    for c in components:
        hist[c.size] += 1
    return dict(sorted(hist.items()))
