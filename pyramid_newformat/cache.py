"""Bounded cross-query cell cache (spec §3).

A shortlist batch (and repeated searches) reference the same cells across queries; this caches whole
cell blocks so a shared cell is read ONCE. The cache key includes object identity — ``(file,
version_or_etag, local_cell_idx)`` — so a replaced/reuploaded shard can never serve stale bytes under
the same manifest offset. Eviction is LRU bounded by a byte budget (default a few cells); set
``max_bytes=0`` to disable. Stores read-only views; callers must not mutate returned arrays.

This is deliberately NOT wired into the single-pass ``search_shard`` (which already reads each union
cell once via its per-cell→queries map). It is for the loader / repeated-search paths where the same
cell recurs across calls.
"""
from __future__ import annotations

from collections import OrderedDict


class CellCache:
    def __init__(self, max_bytes: int = 0):
        self.max_bytes = int(max_bytes)
        self._d = OrderedDict()                   # key -> ndarray (one cell block)
        self._bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @staticmethod
    def key(file: str, identity, local_cell_idx: int):
        """``identity`` = version_id or etag (whichever pins the object); None for unversioned local."""
        return (file, identity, int(local_cell_idx))

    def get(self, key):
        v = self._d.get(key)
        if v is None:
            self.misses += 1
            return None
        self._d.move_to_end(key)                  # LRU: mark most-recently-used
        self.hits += 1
        return v

    def put(self, key, arr):
        if self.max_bytes <= 0:
            return
        nb = int(arr.nbytes)
        if nb > self.max_bytes:                   # single cell bigger than budget → don't cache
            return
        if key in self._d:
            self._bytes -= int(self._d[key].nbytes)
            self._d.move_to_end(key)
        self._d[key] = arr
        self._bytes += nb
        while self._bytes > self.max_bytes and len(self._d) > 1:
            _, old = self._d.popitem(last=False)  # evict least-recently-used
            self._bytes -= int(old.nbytes)
            self.evictions += 1

    def __contains__(self, key):
        return key in self._d

    def __len__(self):
        return len(self._d)

    @property
    def nbytes(self):
        return self._bytes

    def stats(self):
        return {"hits": self.hits, "misses": self.misses, "evictions": self.evictions,
                "n_cached": len(self._d), "bytes": self._bytes, "max_bytes": self.max_bytes}
