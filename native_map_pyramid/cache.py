"""On-disk cache of the FROZEN per-cell per-group residual sums ``S`` for the native map source.

Cache boundary (audited): everything up to and including the frozen VLAD residual accumulation
``S[cell,k] = Σ_{tok∈cell} α_k·value`` is cached; the trainable ``PerGroupProjection`` / ``map_head``
run every step and are NEVER cached. Storage per tile = ``(85, K, d_value)`` fp16
(= 85·32·1024·2 B ≈ 5.31 MiB); ~13.6 GiB for 2615 tiles.

Row layout (canonical, matches the legacy ``region_ids`` cell order): the 85 rows are the four
scales concatenated in the order (8, 4, 2, 1) — n=8 rows [0:64), n=4 [64:80), n=2 [80:84),
n=1 [84:85) — each row-major (i=north, j=west) within its scale.

A STRICT fingerprint over the feature-identity fields is stored; loading a cache whose fingerprint
does not match the current run raises :class:`CacheIncompatibleError` (never a silent read).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

SCHEMA_VERSION = 1
SCALES = (8, 4, 2, 1)                       # canonical stored scale order
CELL_COUNTS = {8: 64, 4: 16, 2: 4, 1: 1}
N_CELLS_TOTAL = sum(CELL_COUNTS.values())   # 85
# constant string pinning the cell/token ordering convention (part of the fingerprint)
TOKEN_ORDER = ("cells=row_major(i=north,j=west); scales=(8,4,2,1) concatenated; "
               "n8=quadrant(global_row=2i+dy,global_col=2j+dx) of the 250m grid")

# identity fields that MUST match between a cache and the run that consumes it
_FINGERPRINT_FIELDS = (
    "schema_version", "map_pyramid_source", "backbone", "dino_layer", "dino_facet",
    "output_px", "tile_size_m", "patch_size", "projection", "projection_seed",
    "projection_in_dim", "projection_out_dim", "n_groups", "d_value",
    "scales", "cell_counts", "crs", "resampling", "token_order", "vlad_dict_id",
    "crop_footprints_m",
)


class CacheIncompatibleError(RuntimeError):
    """Raised when a native cache's fingerprint does not match the current run."""


def scale_row_slices() -> dict:
    """{n: slice} into the 85-row axis, in the canonical (8,4,2,1) order."""
    out, off = {}, 0
    for n in SCALES:
        out[n] = slice(off, off + CELL_COUNTS[n])
        off += CELL_COUNTS[n]
    return out


def vlad_dict_id(assign_weight) -> str:
    """Stable id of the frozen VLAD dictionary = sha256 of the fp32 assign_weight bytes."""
    import torch
    a = assign_weight.detach().cpu().to(torch.float32).contiguous().numpy() \
        if hasattr(assign_weight, "detach") else np.asarray(assign_weight, dtype=np.float32)
    return hashlib.sha256(a.tobytes()).hexdigest()


def build_fingerprint(*, backbone: str, dino_layer, dino_facet: str, output_px: int,
                      tile_size_m: float, patch_size: int, projection: str,
                      projection_seed: int, projection_in_dim: int, projection_out_dim: int,
                      n_groups: int, d_value: int, vlad_dict_id: str,
                      crs: str = "EPSG:3857", resampling: str = "bilinear",
                      crop_footprints_m=(1000.0, 500.0, 250.0)) -> dict:
    """Assemble the identity dict + its sha256 digest under key ``fingerprint``."""
    ident = {
        "schema_version": SCHEMA_VERSION,
        "map_pyramid_source": "native_hierarchical",
        "backbone": str(backbone),
        "dino_layer": (-1 if dino_layer is None else int(dino_layer)),
        "dino_facet": str(dino_facet),
        "output_px": int(output_px),
        "tile_size_m": float(tile_size_m),
        "patch_size": int(patch_size),
        "projection": str(projection),
        "projection_seed": int(projection_seed),
        "projection_in_dim": int(projection_in_dim),
        "projection_out_dim": int(projection_out_dim),
        "n_groups": int(n_groups),
        "d_value": int(d_value),
        "scales": list(SCALES),
        "cell_counts": [CELL_COUNTS[n] for n in SCALES],
        "crs": str(crs),
        "resampling": str(resampling),
        "token_order": TOKEN_ORDER,
        "vlad_dict_id": str(vlad_dict_id),
        "crop_footprints_m": [float(x) for x in crop_footprints_m],
    }
    ident["fingerprint"] = _digest(ident)
    return ident


def _digest(ident: dict) -> str:
    payload = {k: ident[k] for k in _FINGERPRINT_FIELDS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def assert_compatible(cache_ident: dict, expected_ident: dict) -> None:
    """Raise :class:`CacheIncompatibleError` naming the FIRST mismatching identity field.

    Compares field-by-field (not just the digest) so the error message is actionable."""
    for f in _FINGERPRINT_FIELDS:
        cv, ev = cache_ident.get(f), expected_ident.get(f)
        if cv != ev:
            raise CacheIncompatibleError(
                f"native cache incompatible on {f!r}: cache={cv!r} vs run={ev!r}. "
                f"Rebuild the cache with matching settings (or point --native-cache at the right file).")
    if cache_ident.get("fingerprint") != _digest(expected_ident):
        raise CacheIncompatibleError(
            "native cache fingerprint digest mismatch despite equal fields (schema drift?).")


def write_cache(path, S_all, tile_ids, ident: dict, extra_meta: dict | None = None) -> None:
    """Write ``S_all`` (M,85,K,Dv) fp16 + tile ids + identity/fingerprint attrs to an H5 cache."""
    import h5py
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)          # ensure parent dir exists
    S_all = np.asarray(S_all)
    if S_all.ndim != 4 or S_all.shape[1] != N_CELLS_TOTAL:
        raise ValueError(f"S_all must be (M,{N_CELLS_TOTAL},K,Dv), got {S_all.shape}")
    if S_all.shape[0] != len(tile_ids):
        raise ValueError(f"S_all rows {S_all.shape[0]} != len(tile_ids) {len(tile_ids)}")
    with h5py.File(path, "w") as f:
        f.create_dataset("S", data=S_all.astype(np.float16),
                         chunks=(1,) + S_all.shape[1:], compression="gzip", compression_opts=4)
        f.create_dataset("tile_id", data=np.array(list(tile_ids), dtype=object),
                         dtype=h5py.string_dtype())
        f.attrs["fingerprint"] = ident["fingerprint"]
        f.attrs["identity_json"] = json.dumps(ident, sort_keys=True)
        f.attrs["meta_json"] = json.dumps(extra_meta or {}, sort_keys=True)


class NativeCacheWriter:
    """Incremental, resumable writer for the native ``S`` cache.

    Pre-creates the full ``S (M,85,K,Dv)`` dataset + a ``done (M,)`` bool mask and writes one tile
    at a time (flushing periodically), so a crash/interrupt loses at most the un-flushed tail. On
    re-open, if the file exists with the SAME identity + tile_ids + shape, completed rows are kept
    and skipped (resume); otherwise a fresh file is created (unless it exists but is incompatible,
    which raises unless ``fresh=True``)."""

    def __init__(self, path, tile_ids, n_groups, d_value, ident: dict,
                 extra_meta: dict | None = None, *, fresh: bool = False, flush_every: int = 25):
        import os
        import h5py
        from pathlib import Path
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._ids = [str(t) for t in tile_ids]
        M = len(self._ids)
        self.flush_every = int(flush_every)
        self.resumed = False

        can_resume = False
        if os.path.exists(self.path) and not fresh:
            try:
                with h5py.File(self.path, "r") as f:
                    prev = json.loads(f.attrs["identity_json"])
                    ids_ok = [t.decode() if isinstance(t, bytes) else str(t)
                              for t in f["tile_id"][:]] == self._ids
                    shape_ok = tuple(f["S"].shape) == (M, N_CELLS_TOTAL, n_groups, d_value)
                    same = (prev.get("fingerprint") == ident["fingerprint"]) and ids_ok and shape_ok
                if same:
                    can_resume = True
                elif not fresh:
                    raise CacheIncompatibleError(
                        f"{self.path} exists but does not match this run (identity/tile_ids/shape). "
                        f"Pass fresh=True (--fresh) to overwrite, or point --out at a new path.")
            except (KeyError, OSError):
                can_resume = False

        if can_resume:
            self._f = h5py.File(self.path, "r+")
            self.resumed = True
        else:
            self._f = h5py.File(self.path, "w")
            self._f.create_dataset("S", shape=(M, N_CELLS_TOTAL, n_groups, d_value), dtype="float16",
                                   chunks=(1, N_CELLS_TOTAL, n_groups, d_value),
                                   compression="gzip", compression_opts=4)
            self._f.create_dataset("tile_id", data=np.array(self._ids, dtype=object),
                                   dtype=h5py.string_dtype())
            self._f.create_dataset("done", data=np.zeros(M, dtype=bool))
            self._f.attrs["fingerprint"] = ident["fingerprint"]
            self._f.attrs["identity_json"] = json.dumps(ident, sort_keys=True)
            self._f.attrs["meta_json"] = json.dumps(extra_meta or {}, sort_keys=True)
        self._done = np.asarray(self._f["done"][:], dtype=bool)

    def is_done(self, row: int) -> bool:
        return bool(self._done[row])

    def n_done(self) -> int:
        return int(self._done.sum())

    def __len__(self) -> int:
        return len(self._ids)

    def write(self, row: int, S_row) -> None:
        self._f["S"][row] = np.asarray(S_row, dtype=np.float16)
        self._f["done"][row] = True
        self._done[row] = True
        if self.n_done() % self.flush_every == 0:
            self._f.flush()

    def finalize(self) -> int:
        """Flush + close; return the number of tiles still missing (0 = complete)."""
        self._f.flush()
        missing = int((~self._done).sum())
        self._f.close()
        return missing


class NativeCellCache:
    """Reader for a native ``S`` cache. Use :meth:`assert_compatible` before :meth:`load`."""

    def __init__(self, path):
        import h5py
        self._f = h5py.File(path, "r")
        self._S = self._f["S"]                                   # (M,85,K,Dv) fp16, lazy
        self.tile_ids = [t.decode() if isinstance(t, bytes) else str(t)
                         for t in self._f["tile_id"][:]]
        self._row = {t: i for i, t in enumerate(self.tile_ids)}
        self.identity = json.loads(self._f.attrs["identity_json"])
        self.meta = json.loads(self._f.attrs.get("meta_json", "{}"))
        self.fingerprint = self._f.attrs["fingerprint"]
        self._slices = scale_row_slices()

    @property
    def shape(self):
        return tuple(self._S.shape)

    def assert_compatible(self, expected_ident: dict) -> None:
        assert_compatible(self.identity, expected_ident)

    def has(self, tile_id: str) -> bool:
        return tile_id in self._row

    def load(self, tile_ids):
        """Requested tile ids → {n: torch.FloatTensor (m, n², K, Dv)} in the given order."""
        import torch
        missing = [t for t in tile_ids if t not in self._row]
        if missing:
            raise KeyError(f"{len(missing)} tile_id(s) not in native cache, e.g. {missing[:3]}")
        rows = np.array([self._row[t] for t in tile_ids])
        order = np.argsort(rows, kind="stable")                  # h5py fancy-index needs ascending
        arr = self._S[rows[order].tolist()]                      # (m,85,K,Dv) fp16
        inv = np.argsort(order, kind="stable")
        arr = arr[inv]                                           # restore requested order
        t = torch.from_numpy(arr.astype(np.float32))
        return {n: t[:, self._slices[n], :, :].contiguous() for n in SCALES}

    def close(self):
        self._f.close()
