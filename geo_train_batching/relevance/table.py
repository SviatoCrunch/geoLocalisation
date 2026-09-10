"""Relevance tables: per-query positive / safe-negative tiles (ignore = the rest).

A ``RelevanceTable`` is the only geometry contract the batcher + loss need:
``pos_of(i)`` and ``safe_of(i)`` return tile-row index arrays; ``ignore`` is implicit
(neither pos nor safe). Two providers:

* :class:`GeometryRelevanceTable` — pos = IoU ≥ threshold, safe = IoU ≤ eps (ported
  verbatim from the e2c ``IoURelevanceTable``; axis-aligned EPSG:3857 footprints).
* :class:`ExplicitRelevanceTable` — pos/safe supplied directly (used by the split
  adapter, which takes positives from the ``geo_split_no_overlap`` snapshot).
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class RelevanceTable(Protocol):
    @property
    def n_queries(self) -> int: ...

    @property
    def query_ids(self) -> Sequence[str]: ...

    def pos_of(self, i: int) -> np.ndarray: ...

    def safe_of(self, i: int) -> np.ndarray: ...


class GeometryRelevanceTable:
    """IoU relevance between each query's footprint and every tile footprint.

    pos = IoU ≥ ``pos_iou``; safe = IoU ≤ ``iou_eps`` (no overlap); else ignore.
    """

    def __init__(self, q_xy, tile_xy, query_ids, *, tile_size_m: float = 1000.0,
                 query_size_m: float = 1000.0, pos_iou: float = 0.25, iou_eps: float = 1e-9):
        q_xy = np.asarray(q_xy, np.float64).reshape(-1, 2)
        tile_xy = np.asarray(tile_xy, np.float64).reshape(-1, 2)
        self._ids = [str(x) for x in query_ids]
        if len(self._ids) != len(q_xy):
            raise ValueError("query_ids length must match q_xy")
        half = (tile_size_m + query_size_m) / 2.0
        area = tile_size_m * tile_size_m + query_size_m * query_size_m
        self._pos, self._safe = [], []
        for i in range(len(q_xy)):
            ox = np.clip(half - np.abs(tile_xy[:, 0] - q_xy[i, 0]), 0.0, None)
            oy = np.clip(half - np.abs(tile_xy[:, 1] - q_xy[i, 1]), 0.0, None)
            inter = ox * oy
            iou = inter / (area - inter)
            self._pos.append(np.nonzero(iou >= pos_iou)[0].astype(np.int64))
            self._safe.append(np.nonzero(iou <= iou_eps)[0].astype(np.int64))

    @property
    def n_queries(self) -> int:
        return len(self._ids)

    @property
    def query_ids(self):
        return list(self._ids)

    def pos_of(self, i: int) -> np.ndarray:
        return self._pos[int(i)]

    def safe_of(self, i: int) -> np.ndarray:
        return self._safe[int(i)]


class ExplicitRelevanceTable:
    """Relevance from explicit per-query pos/safe row lists (positives may come from an
    external source, e.g. the split's positive-selection snapshot)."""

    def __init__(self, query_ids, pos_rows, safe_rows):
        self._ids = [str(x) for x in query_ids]
        self._pos = [np.asarray(p, np.int64).reshape(-1) for p in pos_rows]
        self._safe = [np.asarray(s, np.int64).reshape(-1) for s in safe_rows]
        if not (len(self._ids) == len(self._pos) == len(self._safe)):
            raise ValueError("query_ids, pos_rows, safe_rows must be equal length")

    @property
    def n_queries(self) -> int:
        return len(self._ids)

    @property
    def query_ids(self):
        return list(self._ids)

    def pos_of(self, i: int) -> np.ndarray:
        return self._pos[int(i)]

    def safe_of(self, i: int) -> np.ndarray:
        return self._safe[int(i)]
