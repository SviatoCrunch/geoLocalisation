"""Convert the per-cell pyramid-embedding store (build_cell_store_s3 output) into the packed
contiguous ``/features`` format (schema.py) in a sibling S3 prefix. Streaming, bounded RAM/disk,
resumable, with on-write bitwise verification. The source store is never modified.

Flow (spec §6): read+validate source index → size/disk estimate → per cell: download → validate
(groups ^p\\d+$ = p0..p{P-1}, levels == config, grid shape, fp16) → assemble (P,L,H,W,D) block in
numeric-position / explicit-level order → write into the shard's contiguous /features[local] → flush
→ read-back bitwise check → checkpoint → delete download. Shard full → sha256 + upload. manifest LAST.

Run: see README. CLI at bottom.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import time
from pathlib import Path

import numpy as np

from . import s3io
from .schema import (AXES, CHECKPOINT_NAME, DS_CELL_IDS, DS_CELL_LAT, DS_CELL_LON, DS_FEATURES,
                     DS_LEVELS, DS_POS_LAT, DS_POS_LON, FEATURES_DTYPE, MANIFEST_NAME, POS_GROUP_RE,
                     SCHEMA_VERSION)

_POS = re.compile(POS_GROUP_RE)


def sorted_cell_ids(index: dict) -> list:
    """Deterministic cell order: numeric by the <N> in '<city>:<N>_lvl0' (fallback: string)."""
    def key(cid):
        m = re.search(r":(\d+)_", cid) or re.search(r"(\d+)", cid)
        return (0, int(m.group(1))) if m else (1, cid)
    return sorted(index["cells"], key=key)


def _pos_groups(f) -> list:
    """HDF5 groups whose name matches ^p\\d+$, sorted NUMERICALLY (p2 < p10). Rejects p_meta etc."""
    gs = [k for k in f.keys() if _POS.match(k) and hasattr(f[k], "keys")]
    return sorted(gs, key=lambda p: int(p[1:]))


def read_source_cell(local_h5: str, levels_m, expect_pos=None):
    """Open a source cell H5, validate structure, return the assembled block + geometry + inventory.

    block: (P, L, H, W, D) float16 (position numeric order; level == ``levels_m`` order).
    Raises ValueError on: wrong position set, missing level, inconsistent grid/dtype, pos-count mismatch.
    """
    import h5py
    with h5py.File(local_h5, "r") as f:
        pos = _pos_groups(f)
        P = len(pos)
        exp = [f"p{i}" for i in range(P)]
        if pos != exp:
            raise ValueError(f"{local_h5}: position groups {pos[:5]}… != contiguous p0..p{P-1}")
        if expect_pos is not None and P != expect_pos:
            raise ValueError(f"{local_h5}: n_positions {P} != expected {expect_pos}")
        lvl_names = [f"l{int(L)}" for L in levels_m]
        g0 = f[pos[0]]
        for ln in lvl_names:
            if ln not in g0:
                raise ValueError(f"{local_h5}/{pos[0]}: missing level {ln} (have {list(g0.keys())})")
        H, W, D = g0[lvl_names[0]].shape
        if str(g0[lvl_names[0]].dtype) != FEATURES_DTYPE:
            raise ValueError(f"{local_h5}: dtype {g0[lvl_names[0]].dtype} != {FEATURES_DTYPE}")
        L = len(lvl_names)
        block = np.empty((P, L, H, W, D), np.float16)
        for pi, pg in enumerate(pos):
            for li, ln in enumerate(lvl_names):
                d = f[pg][ln]
                if d.shape != (H, W, D):
                    raise ValueError(f"{local_h5}/{pg}/{ln}: shape {d.shape} != {(H, W, D)}")
                block[pi, li] = d[()]                       # verbatim fp16, no recompute
        lat = np.asarray(f["lat"][:], np.float64) if "lat" in f else np.full(P, np.nan)
        lon = np.asarray(f["lon"][:], np.float64) if "lon" in f else np.full(P, np.nan)
        inv = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in f.attrs.items()}
        extra = {k: f[k].shape for k in f.keys() if not _POS.match(k)}   # non-position datasets (px/py/lat/lon)
    return block, lat.astype(np.float64), lon.astype(np.float64), (P, L, H, W, D), inv, extra


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _params_fp(city, source_prefix, cell_order, dims, levels_m, cps) -> str:
    import json
    payload = {"city": city, "src": source_prefix, "order": cell_order, "dims": list(dims),
               "levels_m": [int(x) for x in levels_m], "cells_per_shard": cps, "schema": SCHEMA_VERSION}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


class Converter:
    def __init__(self, source_prefix, dest_prefix, work_dir, cells_per_shard=0, prefetch=0,
                 verify=True):
        self.src = source_prefix.rstrip("/")
        self.dst = dest_prefix.rstrip("/")
        self.work = Path(work_dir).expanduser(); self.work.mkdir(parents=True, exist_ok=True)
        self.cps = int(cells_per_shard)
        self.prefetch = int(prefetch)
        self.verify = verify
        self.index = s3io.read_json(s3io.join(self.src, "_index.json"))
        self.config = self.index["config"]
        self.levels_m = [int(x) for x in self.config["levels_m"]]

    # ---- cell download (sequential, optional bounded prefetch) ----
    def _src_key(self, cid):
        return self.index["cells"][cid]["key"]

    def _download(self, cid) -> str:
        local = self.work / "_src" / (cid.replace(":", "_").replace("/", "_") + ".h5")
        s3io.download(self._resolve_src(self._src_key(cid)), str(local))
        return str(local)

    def _resolve_src(self, key: str) -> str:
        """Source cell 'key' in _index.json is the FULL object key (e.g. embeddings/kup/.../cells/x.h5).
        Resolve against the bucket (S3) or the local root that contains the 'embeddings/' tree."""
        if s3io.is_s3(self.src):
            bucket = s3io.split_s3(self.src)[0]
            return f"s3://{bucket}/{key}"
        # local: self.src ends with .../embeddings/<city>/<variant>/<city>; strip to the dir holding 'embeddings'
        root = self.src
        marker = "/embeddings/"
        root = root[: root.index(marker)] if marker in root else str(Path(self.src).parents[3])
        return str(Path(root) / key)

    def plan(self, max_cells=0):
        order = sorted_cell_ids(self.index)
        if max_cells:
            order = order[:max_cells]
        return order

    def estimate(self, order):
        """Peek the FIRST cell for dims, estimate /features bytes (does not trust name-based D)."""
        c0 = order[0]
        local = self._download(c0)
        _, _, _, dims, inv, extra = read_source_cell(local, self.levels_m)
        Path(local).unlink(missing_ok=True)
        P, L, H, W, D = dims
        bpc = P * L * H * W * D * 2
        total = bpc * len(order)
        return dims, bpc, total, inv, extra

    def run(self, order, resume=True, dry_run=False, manifest_only_last=True):
        import h5py
        dims, bpc, total, inv0, extra0 = self.estimate(order)
        P, L, H, W, D = dims
        n = len(order)
        cps = self.cps if self.cps > 0 else n
        n_shards = (n + cps - 1) // cps
        print(f"[plan] cells={n} dims={dims} bytes/cell={bpc/1e6:.1f}MB "
              f"/features total={total/2**30:.2f}GiB shards={n_shards} (cells_per_shard={cps})", flush=True)
        if dry_run:
            free = __import__("shutil").disk_usage(self.work).free
            print(f"[dry-run] work-dir free={free/2**30:.2f}GiB  need≈{(total/n_shards + bpc)/2**30:.2f}GiB/shard peak", flush=True)
            return {"dry_run": True, "dims": dims, "bytes_per_cell": bpc, "total_bytes": total,
                    "n_shards": n_shards}

        fp = _params_fp(self.index["city"], self.src, order, dims, self.levels_m, cps)
        ckpt_path = self.work / CHECKPOINT_NAME
        ckpt = {"fp": fp, "done": {}, "files": {}}
        if resume and ckpt_path.exists():
            prev = s3io.read_json(str(ckpt_path))
            if prev.get("fp") != fp:
                raise SystemExit("[resume] checkpoint fingerprint mismatch (params/order/source changed); "
                                 "use a fresh --work-dir or drop --resume")
            ckpt = prev
            print(f"[resume] {len(ckpt['done'])}/{n} cells already done", flush=True)

        cell_map = {}                                        # cell_id -> (file, global_idx, local_idx)
        files_meta = {}
        for si in range(n_shards):
            lo, hi = si * cps, min((si + 1) * cps, n)
            nl = hi - lo
            fname = f"features_{si:05d}.h5" if n_shards > 1 else "features.h5"
            fpath = self.work / fname
            shard_cells = order[lo:hi]
            done_here = all(order[lo + j] in ckpt["done"] for j in range(nl)) and fname in ckpt["files"]
            if done_here:
                for j, cid in enumerate(shard_cells):
                    cell_map[cid] = (fname, lo + j, j)
                files_meta[fname] = ckpt["files"][fname]
                print(f"[shard {si}] {fname}: already complete ({nl} cells)", flush=True)
                continue
            self._write_shard(fpath, fname, si, lo, shard_cells, dims, bpc, ckpt, ckpt_path, cell_map, files_meta)

        # ---- manifest LAST ----
        manifest = {
            "schema_version": SCHEMA_VERSION, "city": self.index["city"], "source_prefix": self.src,
            "source_index_key": s3io.join(self.src, "_index.json"), "config": self.config,
            "axes": list(AXES), "levels_m": self.levels_m, "dtype": FEATURES_DTYPE,
            "grid_hw": [H, W], "feature_dim": D, "n_positions": P, "n_cells": n,
            "cells_per_shard": cps, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files": [files_meta[f] for f in sorted(files_meta)],
            "cells": {cid: {"global_cell_idx": gi, "file": fn, "local_cell_idx": li,
                            "source_key": self._src_key(cid),
                            "lat": self.index["cells"][cid].get("lat"),
                            "lon": self.index["cells"][cid].get("lon")}
                      for cid, (fn, gi, li) in cell_map.items()},
            "source_inventory": {"cell_attrs_example": inv0, "non_position_datasets": {k: list(v) for k, v in extra0.items()}},
        }
        s3io.write_json(s3io.join(self.dst, MANIFEST_NAME), manifest)
        print(f"[ok] manifest -> {s3io.join(self.dst, MANIFEST_NAME)}  ({n} cells, {n_shards} files)", flush=True)
        return manifest

    def _write_shard(self, fpath, fname, si, lo, shard_cells, dims, bpc, ckpt, ckpt_path, cell_map, files_meta):
        import h5py
        P, L, H, W, D = dims
        nl = len(shard_cells)
        mode = "r+" if fpath.exists() else "w"
        f = h5py.File(fpath, mode)
        if DS_FEATURES not in f:
            f.create_dataset(DS_FEATURES, shape=(nl, P, L, H, W, D), dtype=FEATURES_DTYPE,
                             chunks=None, compression=None)          # contiguous
            f.create_dataset(DS_CELL_IDS, shape=(nl,), dtype=h5py.string_dtype())
            f.create_dataset(DS_CELL_LAT, shape=(nl,), dtype="f8")
            f.create_dataset(DS_CELL_LON, shape=(nl,), dtype="f8")
            f.create_dataset(DS_POS_LAT, shape=(nl, P), dtype="f8")
            f.create_dataset(DS_POS_LON, shape=(nl, P), dtype="f8")
            f.create_dataset(DS_LEVELS, data=np.asarray(self.levels_m, np.int32))
            f.attrs.update({"schema_version": SCHEMA_VERSION, "city": self.index["city"],
                            "axes": ",".join(AXES), "shard_index": si, "n_local_cells": nl})
        ds = f[DS_FEATURES]
        for j, cid in enumerate(shard_cells):
            gi = lo + j
            if cid in ckpt["done"]:
                cell_map[cid] = (fname, gi, j); continue
            local = self._download(cid)
            block, plat, plon, d2, inv, _ = read_source_cell(local, self.levels_m, expect_pos=P)
            if d2 != dims:
                raise ValueError(f"{cid}: dims {d2} != {dims} (inconsistent cell)")
            ds[j] = block
            f[DS_CELL_IDS][j] = cid
            f[DS_CELL_LAT][j] = float(self.index["cells"][cid].get("lat") or np.nan)
            f[DS_CELL_LON][j] = float(self.index["cells"][cid].get("lon") or np.nan)
            f[DS_POS_LAT][j] = plat; f[DS_POS_LON][j] = plon
            f.flush()
            if self.verify:
                back = ds[j]
                if not np.array_equal(back.view(np.uint16), block.view(np.uint16)):
                    raise RuntimeError(f"{cid}: read-back bitwise mismatch in {fname}[{j}]")
            Path(local).unlink(missing_ok=True)
            ckpt["done"][cid] = [fname, gi, j]
            cell_map[cid] = (fname, gi, j)
            s3io.write_json(str(ckpt_path), ckpt)
        # contiguous offset (must be defined for Range GET)
        off = ds.id.get_offset()
        if off is None:
            raise RuntimeError(f"{fname}: /features offset undefined (not contiguous?)")
        f.close()
        sha = _sha256(str(fpath))
        import sys
        meta = {"name": fname, "shape": [nl, P, L, H, W, D], "bytes": int(fpath.stat().st_size),
                "sha256": sha, "features_byte_offset": int(off), "bytes_per_cell": int(bpc),
                "itemsize": 2, "byte_order": "<" if sys.byteorder == "little" else ">",
                "n_local_cells": nl}
        files_meta[fname] = meta
        ckpt["files"][fname] = meta
        s3io.write_json(str(ckpt_path), ckpt)
        if s3io.is_s3(self.dst):
            s3io.upload(str(fpath), s3io.join(self.dst, fname))
            print(f"[shard {si}] uploaded {fname} ({meta['bytes']/2**30:.2f}GiB, sha {sha[:12]})", flush=True)
        else:
            s3io.upload(str(fpath), s3io.join(self.dst, fname))
            print(f"[shard {si}] wrote {fname} ({meta['bytes']/2**30:.2f}GiB)", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-prefix", required=True, help="dir holding _index.json + cells/ (s3:// or local)")
    ap.add_argument("--destination-prefix", required=True, help="target (s3:// or local)")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--cells-per-shard", type=int, default=0, help="0 = single features.h5")
    ap.add_argument("--prefetch", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-cells", type=int, default=0, help="SMOKE ONLY: first N cells → use a TEST destination")
    ap.add_argument("--no-verify", dest="verify", action="store_false", help="skip on-write read-back check")
    args = ap.parse_args(argv)

    conv = Converter(args.source_prefix, args.destination_prefix, args.work_dir,
                     cells_per_shard=args.cells_per_shard, prefetch=args.prefetch, verify=args.verify)
    order = conv.plan(max_cells=args.max_cells)
    if args.max_cells:
        print(f"[warn] --max-cells {args.max_cells}: PARTIAL set — point --destination-prefix at a TEST path", flush=True)
    conv.run(order, resume=args.resume, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
