"""Local per-cell pyramid store reader — the isolated, no-S3 counterpart of ``CellStoreS3``.

Reads cell pyramids that already live on local disk (the production case: the map is downloaded
once, not fetched per query). Same on-disk cell H5 schema as the S3 store so the reranker scoring is
byte-identical:

  * each cell H5 has datasets ``lat`` / ``lon`` (per sub-position), attrs ``levels_m`` (list) and
    ``n_positions``, and the pyramid token grids ``p{i}/l{L}`` (``(h, w, D)`` float);
  * cells are addressed by ``cell_id`` (the shortlist's ids, e.g. ``kup:73_lvl0``).

Two ways to map a ``cell_id`` to a file:
  * with an ``_index.json`` manifest (``{"cells": {cell_id: {"key": "...", "lat":…}}, "config": {…}}``,
    the same file ``build_cell_store_s3`` writes) → resolve by the key's basename under ``cells_dir``;
  * without a manifest → discover ``*.h5`` under the root (or ``root/cells/``), ``cell_id`` = filename
    stem.

NO network, NO boto3. h5py reads each ``p{i}/l{L}`` dataset lazily on access (only the requested
grids touch disk), and open file handles are kept in a small LRU so thousands of cells don't exhaust
file descriptors. Deliberately mirrors the ``CellStoreS3`` API (``has`` / ``cell`` / ``cells`` /
``config`` / ``close`` / ``n_downloads``) so the same scoring loop drives either store.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import torch


class CellData:
    """One opened cell H5: per-position geo + pyramid grids ``p{i}/l{L}`` (lazy per-dataset read).

    Identical contract to ``cell_store_s3.CellData`` so ``rerank_local`` scores it unchanged."""

    def __init__(self, cell_id: str, f):
        self.cell_id = cell_id
        self.f = f
        self.lat = np.asarray(f["lat"][:], float)
        self.lon = np.asarray(f["lon"][:], float)
        self.levels = [int(x) for x in f.attrs["levels_m"]]
        self.n_pos = int(f.attrs.get("n_positions", len(self.lat)))

    def grid(self, i: int, L: int) -> torch.Tensor:
        """Read ONE (position, level) token grid from disk → ``(h, w, D)`` float32 tensor."""
        return torch.from_numpy(np.asarray(self.f[f"p{i}/l{int(L)}"]).astype(np.float32))


class LocalCellStore:
    """Open-on-demand reader of local cell H5s with a bounded handle cache (no download, no S3)."""

    def __init__(self, cells_dir: str, index_path: str | None = None, open_cap: int = 16):
        import h5py  # noqa: F401  (import-time check that h5py is present)

        self.root = Path(cells_dir).expanduser()
        if not self.root.exists():
            raise FileNotFoundError(f"cells_dir does not exist: {self.root}")
        ip = Path(index_path).expanduser() if index_path else (self.root / "_index.json")
        if ip.exists():
            idx = json.loads(ip.read_text(encoding="utf-8"))
            self.cells = idx["cells"]                 # cell_id -> {key, lat, lon, …}
            self.config = idx.get("config", {})
            self._has_index = True
        else:
            # discover: cells under root/cells/ (preferred) or root/, cell_id = filename stem
            base = self.root / "cells" if (self.root / "cells").is_dir() else self.root
            self.cells = {p.stem: {"key": str(p)} for p in sorted(base.glob("*.h5"))}
            self.config = {}
            self._has_index = False
            if not self.cells:
                raise FileNotFoundError(f"no _index.json and no *.h5 under {base}")
        self.open_cap = int(open_cap)
        self._lru: "collections.OrderedDict[str, bool]" = collections.OrderedDict()
        self._open: dict = {}
        self.n_downloads = 0                          # always 0 (local) — kept for API parity

    def has(self, cell_id: str) -> bool:
        return cell_id in self.cells

    def _path(self, cell_id: str) -> Path:
        key = self.cells[cell_id].get("key", "")
        base = key.split("/")[-1] if key else ""      # S3-key basename, if any
        cand = []
        if key:
            cand.append(Path(key))                    # key may already be a local absolute path
        if base:
            cand += [self.root / "cells" / base, self.root / base]
        safe = cell_id.replace(":", "_").replace("/", "_") + ".h5"
        cand += [self.root / "cells" / safe, self.root / safe]
        for c in cand:
            if c.exists():
                return c
        raise FileNotFoundError(f"cell {cell_id!r}: no local file (tried {[str(c) for c in cand]})")

    def cell(self, cell_id: str) -> CellData:
        import h5py
        if cell_id not in self._open:
            self._open[cell_id] = h5py.File(self._path(cell_id), "r")
            self._lru[cell_id] = True
            while len(self._lru) > self.open_cap:     # bound open file descriptors
                old, _ = self._lru.popitem(last=False)
                if old in self._open:
                    self._open.pop(old).close()
        else:
            self._lru.move_to_end(cell_id)
        return CellData(cell_id, self._open[cell_id])

    def close(self):
        for f in self._open.values():
            f.close()
        self._open.clear()
        self._lru.clear()
