"""Packed (GPU) store adapter — drop-in for service.rerank.CellStore, backed by the packed
contiguous ``/features`` new-format (NewFormatReader) instead of per-cell H5 objects.

Same duck-typed interface the reranker uses: ``.has(cell_id) / .cell(cell_id) / .config / .cells``,
and ``.cell()`` returns a CellData-compatible object (``.n_pos / .levels / .grid(i, L) / .lat / .lon``).
So ``reranker.rerank`` runs over this unchanged — only the bytes come from one contiguous read per cell
instead of one object per cell. Local dir reads TRUE per-position coords from the shard's
position_lat/position_lon datasets (manifest coords optional).
"""
from __future__ import annotations

import numpy as np

from .reader import NewFormatReader


class NewFormatCellData:
    """CellData-compatible view of one packed cell block (P, L, H, W, D) float16."""

    def __init__(self, cell_id, block, pos_lat, pos_lon, levels_m):
        self.cell_id = cell_id
        self._block = np.asarray(block)                      # (P, L, H, W, D) fp16
        self.lat = np.asarray(pos_lat, np.float64)
        self.lon = np.asarray(pos_lon, np.float64)
        self.levels = [int(x) for x in levels_m]
        self.n_pos = int(self._block.shape[0])
        self._li = {L: i for i, L in enumerate(self.levels)}

    def grid(self, i: int, L: int):
        import torch
        return torch.from_numpy(self._block[i, self._li[int(L)]].astype(np.float32))


class NewFormatCellStore:
    """``.cell(cid)`` reads the cell's contiguous block (one slice / one Range GET) + its true
    per-position coords and returns a NewFormatCellData. Interchangeable with CellStore."""

    def __init__(self, dataset_dir: str):
        self.r = NewFormatReader(dataset_dir)
        self.config = self.r.manifest.get("config", {})
        self.levels_m = self.r.levels_m
        self.cells = self.r.cells                            # cell_id -> meta (for coverage logging)
        self.n_downloads = 0
        self._by_gidx = {int(c["global_cell_idx"]): cid for cid, c in self.r.cells.items()}

    @property
    def is_s3(self) -> bool:
        return self.r._is_s3

    def resolve(self, index):
        if isinstance(index, (int, np.integer)) or (isinstance(index, str) and index.lstrip("-").isdigit()):
            return self._by_gidx[int(index)]
        return index

    def has(self, cell_id) -> bool:
        return cell_id in self.r.cells

    def cell(self, cell_id) -> NewFormatCellData:
        block = self.r.read_cell(cell_id)
        plat, plon = self.r.read_cell_coords(cell_id)
        return NewFormatCellData(cell_id, block, plat, plon, self.levels_m)

    def close(self):
        self.r.close()
