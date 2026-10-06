"""Local-dir per-cell pyramid store reader (no S3) — mirrors :class:`CellStoreS3`'s interface.

Same layout ``build_cell_store_s3`` writes and ``CellStoreS3`` fetches into:
  <root>/_index.json            manifest {cells: {cell_id: {key, lat, lon, …}}, config: {...}}
  <root>/cells/<safe_id>.h5     one H5 per cell (``key`` in the manifest, usually ``cells/<safe>.h5``)

Use it to read a store that is already synced to local disk (e.g. ``kram_store_local``) so the search
does ZERO S3 downloads and needs no cell cache. Drop-in for ``CellStoreS3`` in ``search_pyramid_s3``
(same ``has`` / ``cell`` / ``config`` / ``n_downloads`` / ``close`` surface; ``n_downloads`` stays 0).
"""
from __future__ import annotations

import json
from pathlib import Path

from .cell_store_s3 import CellData


def _safe(cell_id: str) -> str:
    return cell_id.replace(":", "_").replace("/", "_")


class CellStoreLocal:
    def __init__(self, root: str):
        self.root = Path(root).expanduser()
        self.index = json.loads((self.root / "_index.json").read_text())
        self.cells = self.index["cells"]
        self.config = self.index.get("config", {})
        self._open: dict = {}
        self.n_downloads = 0                                  # always 0 — kept for API parity

    def has(self, cell_id: str) -> bool:
        return cell_id in self.cells

    def _path(self, cell_id: str) -> Path:
        rel = self.cells[cell_id].get("key")                  # manifest key is relative to the store root
        if rel:
            p = self.root / rel
            if p.exists():
                return p
        return self.root / "cells" / (_safe(cell_id) + ".h5")  # fallback to the standard filename

    def cell(self, cell_id: str) -> CellData:
        import h5py
        if cell_id not in self._open:
            self._open[cell_id] = h5py.File(self._path(cell_id), "r")
        return CellData(cell_id, self._open[cell_id])

    def close(self):
        for f in self._open.values():
            f.close()
        self._open.clear()
