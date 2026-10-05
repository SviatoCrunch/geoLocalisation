"""Local (offline) per-cell pyramid store — drop-in for CellStoreS3, no S3 at runtime.

Reads ``_index.json`` + the ``cells/*.h5`` from a local directory (an ``aws s3 sync`` of the S3
store prefix), so a run touches zero network: cleaner, reproducible timing. Mirrors the CellStoreS3
interface used by evaluate/search: ``has / cell / config / n_downloads / close``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class LocalCellStore:
    def __init__(self, root: str):
        self.root = Path(root).expanduser()
        self.index = json.loads((self.root / "_index.json").read_text())
        self.cells = self.index["cells"]                 # cell_id -> {key, lat, lon, ...}
        self.config = self.index.get("config", {})
        self.n_downloads = 0                              # always 0 (local)
        self._open: dict = {}

    def has(self, cell_id: str) -> bool:
        return cell_id in self.cells

    def cell(self, cell_id: str):
        import h5py
        from patch_rerank.cell_store_s3 import CellData
        if cell_id not in self._open:
            key = self.cells[cell_id]["key"]
            self._open[cell_id] = h5py.File(self.root / key, "r")
        return CellData(cell_id, self._open[cell_id])

    def close(self):
        for f in self._open.values():
            f.close()
        self._open.clear()
