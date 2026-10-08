"""Reader for the packed contiguous ``/features`` format (schema.py), local files or S3.

API (same for single-file or sharded; files resolved via manifest.json):
    r = NewFormatReader(dest_prefix)
    r.read_cell(cell_id)            -> (P, L, H, W, D) float16
    r.read_cell_level(cell_id, Lm)  -> (P, H, W, D)    float16
    r.read_cells(cell_ids)          -> list aligned to the request order

Local: keeps file handles open across calls; coalesces consecutive local cells into one contiguous
slice (``read_direct`` into a reusable buffer). S3: reads a cell's contiguous block via one Range GET
at its manifest byte-offset (offset bound to the object VersionId captured at init); consecutive cells
in one shard are fetched in a single range. Reading non-consecutive cells does NOT promise one GET.
Transfer to GPU is a separate step (reads land in a CPU buffer).
"""
from __future__ import annotations

import numpy as np

from . import s3io
from .schema import DS_FEATURES, MANIFEST_NAME


class NewFormatReader:
    def __init__(self, dest_prefix: str, manifest: dict | None = None):
        self.dst = dest_prefix.rstrip("/")
        self.manifest = manifest or s3io.read_json(s3io.join(self.dst, MANIFEST_NAME))
        m = self.manifest
        self.levels_m = [int(x) for x in m["levels_m"]]
        self.P = int(m["n_positions"]); self.H, self.W = m["grid_hw"]; self.D = int(m["feature_dim"])
        self.L = len(self.levels_m)
        self.block_shape = (self.P, self.L, self.H, self.W, self.D)
        self.files = {f["name"]: f for f in m["files"]}
        self.cells = m["cells"]
        self._is_s3 = s3io.is_s3(self.dst)
        self._fh = {}                                        # local h5py handles
        self._ver = {}                                       # s3 file -> VersionId (immutability pin)
        self._dtype = np.dtype("<f2")                        # x86/ARM little-endian

    # ---------- resolution ----------
    def _loc(self, cell_id):
        c = self.cells[cell_id]
        return c["file"], int(c["local_cell_idx"])

    def _file_uri(self, fname):
        return s3io.join(self.dst, fname)

    def _handle(self, fname):
        if fname not in self._fh:
            import h5py
            self._fh[fname] = h5py.File(self._file_uri(fname), "r")
        return self._fh[fname]

    def _version(self, fname):
        if fname not in self._ver:
            self._ver[fname] = s3io.version_id(self._file_uri(fname))
        return self._ver[fname]

    # ---------- single cell ----------
    def read_cell(self, cell_id, out=None):
        fname, li = self._loc(cell_id)
        if self._is_s3:
            fm = self.files[fname]
            off = fm["features_byte_offset"] + li * fm["bytes_per_cell"]
            raw = s3io.range_get(self._file_uri(fname), off, fm["bytes_per_cell"], self._version(fname))
            arr = np.frombuffer(raw, self._dtype).reshape(self.block_shape)
            return arr.copy() if out is None else np.copyto(out, arr) or out
        ds = self._handle(fname)[DS_FEATURES]
        if out is not None:
            ds.read_direct(out, np.s_[li], np.s_[:])
            return out
        return ds[li]

    def read_cell_level(self, cell_id, level_m, out=None):
        if int(level_m) not in self.levels_m:
            raise KeyError(f"level {level_m} not in {self.levels_m}")
        li_lvl = self.levels_m.index(int(level_m))
        if self._is_s3:
            return self.read_cell(cell_id)[:, li_lvl]          # S3 block is cell-contiguous; slice in RAM
        fname, li = self._loc(cell_id)
        ds = self._handle(fname)[DS_FEATURES]
        if out is not None:
            ds.read_direct(out, np.s_[li, :, li_lvl], np.s_[:]); return out
        return ds[li, :, li_lvl]

    # ---------- many cells (coalesced) ----------
    def read_cells(self, cell_ids):
        out = [None] * len(cell_ids)
        by_file = {}
        for req_i, cid in enumerate(cell_ids):
            fname, li = self._loc(cid)
            by_file.setdefault(fname, []).append((li, req_i))
        for fname, items in by_file.items():
            items.sort()                                       # by local_idx
            runs = []                                          # [(lo, hi, [req_i...]), ...] hi inclusive
            for li, req_i in items:
                if runs and li == runs[-1][1] + 1:
                    runs[-1][1] = li; runs[-1][2].append(req_i)
                else:
                    runs.append([li, li, [req_i]])
            if self._is_s3:
                fm = self.files[fname]; uri = self._file_uri(fname); ver = self._version(fname)
                for lo, hi, reqs in runs:
                    off = fm["features_byte_offset"] + lo * fm["bytes_per_cell"]
                    raw = s3io.range_get(uri, off, (hi - lo + 1) * fm["bytes_per_cell"], ver)
                    blk = np.frombuffer(raw, self._dtype).reshape((hi - lo + 1,) + self.block_shape)
                    for k, req_i in enumerate(reqs):
                        out[req_i] = blk[k].copy()
            else:
                ds = self._handle(fname)[DS_FEATURES]
                for lo, hi, reqs in runs:
                    chunk = ds[lo:hi + 1]                       # one contiguous slice
                    for k, req_i in enumerate(reqs):
                        out[req_i] = chunk[k]
        return out

    def read_cell_coords(self, cell_id):
        """(pos_lat (P,), pos_lon (P,)) for a cell. Local: from position_lat/lon datasets; S3: falls
        back to the cell-level lat/lon broadcast (per-position arrays aren't range-addressable)."""
        from .schema import DS_POS_LAT, DS_POS_LON
        fname, li = self._loc(cell_id)
        if self._is_s3:
            c = self.cells[cell_id]
            return (np.full(self.P, c.get("lat") if c.get("lat") is not None else np.nan),
                    np.full(self.P, c.get("lon") if c.get("lon") is not None else np.nan))
        f = self._handle(fname)
        return np.asarray(f[DS_POS_LAT][li], np.float64), np.asarray(f[DS_POS_LON][li], np.float64)

    def close(self):
        for f in self._fh.values():
            f.close()
        self._fh.clear()
