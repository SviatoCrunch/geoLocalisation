"""S3-resident per-cell pyramid store reader: fetch-by-index + bounded local LRU cache.

Companion to ``build_cell_store_s3``. Loads the ``_index.json`` manifest (cell_id → S3 key + geo +
config fingerprint), downloads a cell's H5 on demand into a capped local cache, and exposes its
pyramid token grids. The search fetches only the coarse shortlist's top-N cells; neighbouring queries
share cells via the cache. No whole-city download.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import torch


class CellData:
    """One opened cell H5: per-position geo + pyramid grids ``p{i}/l{L}`` (all output_px/patch tokens)."""

    def __init__(self, cell_id: str, f):
        self.cell_id = cell_id
        self.f = f
        self.lat = np.asarray(f["lat"][:], float)
        self.lon = np.asarray(f["lon"][:], float)
        self.levels = [int(x) for x in f.attrs["levels_m"]]
        self.n_pos = int(f.attrs.get("n_positions", len(self.lat)))

    def grid(self, i: int, L: int) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.f[f"p{i}/l{int(L)}"]).astype(np.float32))


class CellStoreS3:
    def __init__(self, index_uri: str, cache_dir: str = "/tmp/cellcache", cache_cap: int = 256,
                 s3_client=None):
        from s3_gt_sync.core import make_s3_client, parse_s3_uri
        self.bucket, key = parse_s3_uri(index_uri)            # index_uri = s3://…/<city>/_index.json
        self.s3 = s3_client if s3_client is not None else make_s3_client()
        body = self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        self.index = json.loads(body)
        self.cells = self.index["cells"]                      # cell_id -> {key, lat, lon, …}
        self.config = self.index.get("config", {})
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_cap = int(cache_cap)
        self._lru: "collections.OrderedDict[str, Path]" = collections.OrderedDict()
        self._open: dict = {}
        self.n_downloads = 0                                  # for speed accounting

    def has(self, cell_id: str) -> bool:
        return cell_id in self.cells

    def _local(self, cell_id: str) -> Path:
        if cell_id in self._lru:
            self._lru.move_to_end(cell_id)
            return self._lru[cell_id]
        key = self.cells[cell_id]["key"]
        p = self.cache_dir / (cell_id.replace(":", "_").replace("/", "_") + ".h5")
        if not p.exists():
            self.s3.download_file(self.bucket, key, str(p))
            self.n_downloads += 1
        self._lru[cell_id] = p
        while len(self._lru) > self.cache_cap:                # evict LRU
            old, oldp = self._lru.popitem(last=False)
            if old in self._open:
                self._open.pop(old).close()
            try:
                oldp.unlink()
            except OSError:
                pass
        return p

    def cell(self, cell_id: str) -> CellData:
        import h5py
        if cell_id not in self._open:
            self._open[cell_id] = h5py.File(self._local(cell_id), "r")
        return CellData(cell_id, self._open[cell_id])

    def close(self):
        for f in self._open.values():
            f.close()
        self._open.clear()
