"""GPU-facing access to the packed new-format store, by EXTERNAL cell indices, in CHUNKS.

Inputs the search tool receives from outside: a dataset DIRECTORY (new-format) and a list of cell
INDICES (e.g. the coarse top-100 per query) — either ``cell_id`` strings or integer
``global_cell_idx``. This module turns those into GPU-ready batches:

  ChunkedCellLoader(dataset_dir).iter_chunks(indices)   -> (cell_ids_chunk, tensor (C,P,L,H,W,D) on GPU)

and provides a drop-in store so the existing faithful-MAGSAC++ reranker
(``magsacpp_torch.search.search``) runs on this format unchanged:

  NewFormatCellStore(dataset_dir)   # .has / .cell / .config / .n_downloads / .close
"""
from __future__ import annotations

import numpy as np

from .reader import NewFormatReader


def _is_int(x):
    return isinstance(x, (int, np.integer)) or (isinstance(x, str) and x.lstrip("-").isdigit())


class ChunkedCellLoader:
    """Fetch cells by index in chunks of coalesced contiguous reads, each moved to the GPU once.

    ``chunk_cells`` cells per batch bounds the per-step work tensor; consecutive local cells inside a
    chunk are read as ONE slice (local) / ONE Range GET (S3) by the reader. Reads land in a CPU buffer;
    the GPU transfer is the explicit ``.to(device)`` here (optionally via pinned memory)."""

    def __init__(self, dataset_dir, device="cuda", chunk_cells=8, to_device=True, pin_memory=False):
        self.r = NewFormatReader(dataset_dir)
        self.device = device
        self.chunk = int(chunk_cells)
        self.to_device = to_device
        self.pin = pin_memory
        self._by_gidx = {int(c["global_cell_idx"]): cid for cid, c in self.r.cells.items()}
        self.block_shape = self.r.block_shape          # (P, L, H, W, D)
        self.levels_m = self.r.levels_m

    def resolve(self, indices) -> list:
        """Map a mixed list of global_cell_idx (int/str digits) and/or cell_id strings -> cell_ids."""
        out = []
        for x in indices:
            if _is_int(x):
                gi = int(x)
                if gi not in self._by_gidx:
                    raise KeyError(f"global_cell_idx {gi} out of range")
                out.append(self._by_gidx[gi])
            elif x in self.r.cells:
                out.append(x)
            else:
                raise KeyError(f"index {x!r} is neither a global_cell_idx nor a known cell_id")
        return out

    def iter_chunks(self, indices):
        """Yield (cell_ids_chunk, data). data = torch (C,P,L,H,W,D) float16 on ``device`` (to_device)
        else numpy. Order within a chunk follows the request order."""
        cids = self.resolve(indices)
        for i in range(0, len(cids), self.chunk):
            sub = cids[i:i + self.chunk]
            arr = np.stack(self.r.read_cells(sub))          # (C,P,L,H,W,D) fp16, coalesced reads
            if not self.to_device:
                yield sub, arr
                continue
            import torch
            t = torch.from_numpy(arr)
            if self.pin and t.device.type == "cpu":
                t = t.pin_memory()
            yield sub, t.to(self.device, non_blocking=self.pin)

    def close(self):
        self.r.close()


def cell_to_crops(block, levels_m, device=None):
    """new-format cell block (P,L,H,W,D) -> (grids (P*L, H*W, D) torch float32, keys[(pos,level_m)], kp).
    Mirrors ``search_pyramid_s3._load_cell`` output so a custom loop can match/verify these crops."""
    import torch

    from patch_rerank.matcher import grid_keypoints
    P, L, H, W, D = block.shape
    grids = torch.as_tensor(np.asarray(block)).reshape(P * L, H * W, D).float()
    if device is not None:
        grids = grids.to(device)
    keys = [(pi, int(levels_m[li])) for pi in range(P) for li in range(L)]
    return grids, keys, grid_keypoints(H, W)


class NewFormatCellData:
    """CellData-compatible view of one new-format cell block (so _load_cell / search see the old API)."""

    def __init__(self, cell_id, block, pos_lat, pos_lon, levels_m):
        self.cell_id = cell_id
        self._block = np.asarray(block)                     # (P,L,H,W,D) fp16
        self.lat = np.asarray(pos_lat, np.float64)
        self.lon = np.asarray(pos_lon, np.float64)
        self.levels = [int(x) for x in levels_m]
        self.n_pos = int(self._block.shape[0])
        self._li = {L: i for i, L in enumerate(self.levels)}

    def grid(self, i: int, L: int):
        import torch
        return torch.from_numpy(self._block[i, self._li[int(L)]].astype(np.float32))


class NewFormatCellStore:
    """Drop-in for ``cell_store_s3.CellStoreS3`` backed by the packed new format (local or S3).

    ``.cell(cid)`` reads the cell's contiguous block (one slice / one Range GET) and returns a
    :class:`NewFormatCellData`. Lets ``magsacpp_torch.search.search`` (the GPU reranker) run on this
    store by swapping only the store object — same scoring/aggregation."""

    def __init__(self, dataset_dir):
        self.r = NewFormatReader(dataset_dir)
        self.config = self.r.manifest.get("config", {})
        self.levels_m = self.r.levels_m
        self.n_downloads = 0
        self._by_gidx = {int(c["global_cell_idx"]): cid for cid, c in self.r.cells.items()}

    def resolve(self, index):
        if _is_int(index):
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
