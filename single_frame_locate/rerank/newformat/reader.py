"""Reader for the packed contiguous ``/features`` format (schema.py), local files or S3.

Design notes tied to the docs:
  * ``/features`` is contiguous + uncompressed, so a cell is one physical span; a run of consecutive
    cells is one contiguous slice (HDF5 chunking docs — layout matches access, so no chunk cache).
  * ``read_cells_into(ids, out)`` fills a caller-owned buffer via ``Dataset.read_direct`` (h5py docs)
    to avoid the ``list→np.stack`` second big copy. It returns the cell order it wrote (``(file,
    local_cell_idx)`` sorted) because read_direct wants a contiguous dest; callers key results by
    cell_id, not position, so order is explicit, not assumed.
  * S3 reads one contiguous cell/run via a single Range GET at the manifest ``features_byte_offset``
    (S3 GetObject: one GET = one range). Object identity is pinned by the manifest's recorded
    ``version_id`` (or ``etag`` via IfMatch) — NOT a per-open HEAD. S3 cannot zero-copy (the transport
    yields bytes); we do exactly one copy into ``out``.
  * A level is axis-2 of the cell block → a single level is STRIDED across positions (gap of L between
    positions); there is no cheaper physical read for one level than the whole cell.
"""
from __future__ import annotations

import numpy as np

from . import s3io
from .schema import DS_FEATURES, DS_POS_LAT, DS_POS_LON, MANIFEST_NAME


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

    def _pin(self, fname):
        """Object identity from the MANIFEST (recorded at convert-time), not a per-open HEAD.
        Returns (version_id, etag) — either may be None (unversioned / not recorded)."""
        fm = self.files[fname]
        return fm.get("version_id"), fm.get("etag")

    # ---------- runs over local indices ----------
    @staticmethod
    def _runs(sorted_lis):
        runs = []                                            # [[lo, hi]] inclusive, consecutive
        for li in sorted_lis:
            if runs and li == runs[-1][1] + 1:
                runs[-1][1] = li
            else:
                runs.append([li, li])
        return runs

    # ---------- single cell ----------
    def read_cell(self, cell_id, out=None):
        fname, li = self._loc(cell_id)
        if self._is_s3:
            fm = self.files[fname]
            off = fm["features_byte_offset"] + li * fm["bytes_per_cell"]
            ver, etag = self._pin(fname)
            raw = s3io.range_get(self._file_uri(fname), off, fm["bytes_per_cell"], version=ver, etag=etag)
            arr = np.frombuffer(raw, self._dtype).reshape(self.block_shape)
            if out is None:
                return arr.copy()
            np.copyto(out, arr); return out
        ds = self._handle(fname)[DS_FEATURES]
        if out is not None:
            ds.read_direct(out, np.s_[li], np.s_[:])           # straight into caller buffer, no temp
            return out
        return ds[li]

    def read_cell_level(self, cell_id, level_m, out=None):
        """(P, H, W, D) for one level. The level is STRIDED within the cell (gap L between positions),
        so there is no cheaper physical read than the cell; S3 reads the whole cell once then slices."""
        if int(level_m) not in self.levels_m:
            raise KeyError(f"level {level_m} not in {self.levels_m}")
        j = self.levels_m.index(int(level_m))
        if self._is_s3:
            sl = self.read_cell(cell_id)[:, j]
            if out is None:
                return sl
            np.copyto(out, sl); return out
        fname, li = self._loc(cell_id)
        ds = self._handle(fname)[DS_FEATURES]
        if out is not None:
            ds.read_direct(out, np.s_[li, :, j], np.s_[:]); return out
        return ds[li, :, j]

    # ---------- many cells: dedup + coalesced + read into a provided buffer ----------
    def read_cells_into(self, cell_ids, out=None):
        """Read the UNIQUE cells of ``cell_ids`` (duplicates read once) in (file, local_cell_idx) order
        into ``out`` (shape ``(n_unique, *block_shape)``, created if None) and return the ordered unique
        cell_id list aligned to ``out`` rows. Contiguous runs = one read_direct (local) / one Range GET
        (S3). No ``np.stack`` second copy. Callers map by cell_id (see :meth:`read_cells` for req order)."""
        uniq = list(dict.fromkeys(cell_ids))                  # dedup, keep first-seen order
        loc = {c: self._loc(c) for c in uniq}
        order = sorted(uniq, key=lambda c: (loc[c][0], loc[c][1]))   # (file, local_idx) physical order
        n = len(order)
        if out is None:
            out = np.empty((n,) + self.block_shape, self._dtype)
        elif out.shape[0] < n:
            raise ValueError(f"out has {out.shape[0]} rows < {n} unique cells")
        pos = {c: i for i, c in enumerate(order)}             # cell -> out row
        by_file = {}
        for c in order:
            by_file.setdefault(loc[c][0], []).append(loc[c][1])
        for fname, lis in by_file.items():
            lis.sort()
            li2row = {}                                       # local_idx -> out row (via cell)
            for c in order:
                if loc[c][0] == fname:
                    li2row[loc[c][1]] = pos[c]
            if self._is_s3:
                fm = self.files[fname]; uri = self._file_uri(fname); ver, etag = self._pin(fname)
                for lo, hi in self._runs(lis):
                    raw = s3io.range_get(uri, fm["features_byte_offset"] + lo * fm["bytes_per_cell"],
                                         (hi - lo + 1) * fm["bytes_per_cell"], version=ver, etag=etag)
                    blk = np.frombuffer(raw, self._dtype).reshape((hi - lo + 1,) + self.block_shape)
                    for k, li in enumerate(range(lo, hi + 1)):
                        np.copyto(out[li2row[li]], blk[k])    # single copy; no stack
            else:
                ds = self._handle(fname)[DS_FEATURES]
                for lo, hi in self._runs(lis):
                    rows = [li2row[li] for li in range(lo, hi + 1)]
                    if rows == list(range(rows[0], rows[0] + len(rows))):     # contiguous dest
                        ds.read_direct(out, np.s_[lo:hi + 1], np.s_[rows[0]:rows[0] + len(rows)])
                    else:
                        chunk = ds[lo:hi + 1]
                        for k, rrow in enumerate(rows):
                            out[rrow] = chunk[k]
        return out, order

    def read_cells(self, cell_ids):
        """Back-compat: list aligned to REQUEST order (duplicates share one physical read)."""
        out, order = self.read_cells_into(cell_ids)
        row = {c: i for i, c in enumerate(order)}
        return [out[row[c]] for c in cell_ids]

    # ---------- per-position coords (true, from manifest; one read) ----------
    def read_cell_coords(self, cell_id):
        """(pos_lat (P,), pos_lon (P,)) — TRUE per-position coords from the manifest (written at
        convert-time). Falls back to the local position_lat/lon datasets if the manifest lacks them."""
        c = self.cells[cell_id]
        if c.get("pos_lat") is not None and c.get("pos_lon") is not None:
            return np.asarray(c["pos_lat"], np.float64), np.asarray(c["pos_lon"], np.float64)
        if not self._is_s3:                                   # fallback: read datasets once
            fname, li = self._loc(cell_id)
            f = self._handle(fname)
            return np.asarray(f[DS_POS_LAT][li], np.float64), np.asarray(f[DS_POS_LON][li], np.float64)
        raise KeyError(f"{cell_id}: no per-position coords in manifest (re-run convert to record them)")

    def close(self):
        for f in self._fh.values():
            f.close()
        self._fh.clear()
