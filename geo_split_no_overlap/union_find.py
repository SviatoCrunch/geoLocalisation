"""Disjoint-set union (union-find) with path compression + deterministic roots.

Mirrors the DSU already used in ``make_multicity_split._components`` and
``siam_model_stage2/split_crop_geometry.connected_components`` (path compression,
``parent[max]=min`` so the root is always the smallest index in a set). Kept local
so the new module has no hidden dependency on a stage pipeline.
"""
from __future__ import annotations

from collections import defaultdict


class UnionFind:
    def __init__(self, n: int):
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        p = self._parent
        while p[x] != x:
            p[x] = p[p[x]]          # path halving
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[max(ra, rb)] = min(ra, rb)   # smaller index becomes root

    def components(self) -> list:
        """Return list of member-index lists, each sorted ascending, order-stable.

        Groups are ordered by their smallest member so the output is deterministic
        for identical input regardless of insertion order.
        """
        groups = defaultdict(list)
        for x in range(len(self._parent)):
            groups[self.find(x)].append(x)
        return [sorted(members) for _, members in sorted(groups.items())]


def connected_components(n: int, edges) -> list:
    """Convenience: components of an undirected graph given as an iterable of (i, j)."""
    uf = UnionFind(n)
    for i, j in edges:
        uf.union(int(i), int(j))
    return uf.components()
