"""Per-cell pyramid store reader — LOCAL dir (default, prod) or S3 — vendored for the service.

Store layout (built by geoLocalisation/patch_rerank/build_cell_store_s3.py):
  <root>/_index.json              manifest: {cells: {cell_id: {key, lat, lon, …}}, config: {...}}
  <root>/cells/<safe_id>.h5       one H5 per cell: lat/lon (per position) + p{i}/l{L} token grids

LOCAL mode (``source`` is a directory, e.g. /home/ubuntu/work/kup-files/pyramid/kup): reads the H5
straight off disk — fastest, no network, fits once the city's store (~90 GB for kup 32×32) is synced.
S3 mode (``source`` is s3://…/_index.json): downloads each cell on demand into a bounded local cache
(fallback for cities too big to hold locally). cell_id → file name ``<cell_id ':'/'/'→'_'>.h5``.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import torch


def _safe(cell_id: str) -> str:
    return cell_id.replace(":", "_").replace("/", "_")


class CellData:
    """One opened cell H5: per-position lat/lon + pyramid grids ``p{i}/l{L}``."""

    def __init__(self, cell_id: str, f):
        self.cell_id = cell_id
        self.f = f
        self.lat = np.asarray(f["lat"][:], float)
        self.lon = np.asarray(f["lon"][:], float)
        self.levels = [int(x) for x in f.attrs["levels_m"]]
        self.n_pos = int(f.attrs.get("n_positions", len(self.lat)))

    def grid(self, i: int, L: int) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.f[f"p{i}/l{int(L)}"]).astype(np.float32))


class CellStore:
    def __init__(self, source: str, cache_dir: str = "/tmp/cellcache", cache_cap: int = 256,
                 s3_client=None):
        self.is_s3 = source.startswith("s3://")
        if self.is_s3:
            self.bucket, _, key = source[5:].partition("/")      # s3://<bucket>/<…/_index.json>
            if s3_client is not None:
                self.s3 = s3_client
            else:
                import boto3
                self.s3 = boto3.client("s3")
            self.index = json.loads(self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read())
            ns = key[:-len("/_index.json")] if key.endswith("/_index.json") else key
            ns = (self.bucket + "_" + ns).replace("/", "_").replace(":", "_")
            self.cache_dir = Path(cache_dir).expanduser() / ns
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_cap = int(cache_cap)
            self._lru: "collections.OrderedDict[str, Path]" = collections.OrderedDict()
        else:
            self.root = Path(source).expanduser()                # dir with _index.json + cells/
            self.index = json.loads((self.root / "_index.json").read_text())
        self.cells = self.index["cells"]
        self.config = self.index.get("config", {})
        self._open: dict = {}
        self.n_fetched = 0

    def has(self, cell_id: str) -> bool:
        return cell_id in self.cells

    def _path(self, cell_id: str) -> Path:
        if not self.is_s3:
            return self.root / "cells" / (_safe(cell_id) + ".h5")
        if cell_id in self._lru:                                 # S3 LRU cache
            self._lru.move_to_end(cell_id)
            return self._lru[cell_id]
        key = self.cells[cell_id]["key"]
        p = self.cache_dir / (_safe(cell_id) + ".h5")
        if not p.exists():
            self.s3.download_file(self.bucket, key, str(p))
            self.n_fetched += 1
        self._lru[cell_id] = p
        while len(self._lru) > self.cache_cap:
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
            self._open[cell_id] = h5py.File(self._path(cell_id), "r")
        return CellData(cell_id, self._open[cell_id])

    def close(self):
        for f in self._open.values():
            f.close()
        self._open.clear()
