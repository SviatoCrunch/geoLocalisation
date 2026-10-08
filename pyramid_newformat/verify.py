"""Streaming source-vs-result verification (bounded RAM): confirm the packed format is a bit-exact
repack of the source per-cell store, and that the manifest mapping / byte-offsets are correct.

For each cell (one at a time): read the new-format block via the reader AND the source cell block,
assert bitwise-equal float16 (uint16 view), assert coords, and (S3/local) assert the manifest
byte-offset Range read reproduces the same bytes. Reports pass/fail counts + first mismatch.
"""
from __future__ import annotations

import argparse

import numpy as np

from . import s3io
from .convert import Converter, read_source_cell, sorted_cell_ids
from .reader import NewFormatReader


def verify(source_prefix, dest_prefix, work_dir, max_cells=0, check_offset=True) -> dict:
    conv = Converter(source_prefix, dest_prefix, work_dir)       # for _resolve_src/_download + levels
    r = NewFormatReader(dest_prefix)
    order = [c for c in sorted_cell_ids(conv.index) if c in r.cells]
    if max_cells:
        order = order[:max_cells]
    n_ok = 0; mism = []
    for i, cid in enumerate(order):
        newblk = r.read_cell(cid)
        local = conv._download(cid)
        srcblk, plat, plon, dims, inv, _ = read_source_cell(local, conv.levels_m, expect_pos=r.P)
        __import__("pathlib").Path(local).unlink(missing_ok=True)
        if newblk.shape != srcblk.shape or not np.array_equal(newblk.view(np.uint16), srcblk.view(np.uint16)):
            mism.append((cid, "features")); continue
        # per-level slice consistency
        if not np.array_equal(r.read_cell_level(cid, conv.levels_m[0]).view(np.uint16), srcblk[:, 0].view(np.uint16)):
            mism.append((cid, "level-slice")); continue
        # per-position coords: reader (from manifest) must match the source cell's true coords
        rlat, rlon = r.read_cell_coords(cid)
        if not (np.allclose(rlat, plat, atol=1e-9, rtol=0) and np.allclose(rlon, plon, atol=1e-9, rtol=0)):
            mism.append((cid, "coords")); continue
        # offset / range read reproduces bytes
        if check_offset:
            fname, li = r._loc(cid); fm = r.files[fname]
            off = fm["features_byte_offset"] + li * fm["bytes_per_cell"]
            ver, etag = r._pin(fname)
            raw = s3io.range_get(r._file_uri(fname), off, fm["bytes_per_cell"], version=ver, etag=etag)
            if not np.array_equal(np.frombuffer(raw, "<f2").reshape(r.block_shape).view(np.uint16),
                                  srcblk.view(np.uint16)):
                mism.append((cid, "range-offset")); continue
        n_ok += 1
    res = {"n_checked": len(order), "n_ok": n_ok, "n_mismatch": len(mism), "first_mismatches": mism[:10]}
    print(f"[verify] ok {n_ok}/{len(order)}  mismatches={len(mism)}" + (f"  e.g. {mism[:5]}" if mism else ""),
          flush=True)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-prefix", required=True)
    ap.add_argument("--destination-prefix", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--max-cells", type=int, default=0)
    ap.add_argument("--no-offset", dest="check_offset", action="store_false")
    a = ap.parse_args(argv)
    res = verify(a.source_prefix, a.destination_prefix, a.work_dir, a.max_cells, a.check_offset)
    return 0 if res["n_mismatch"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
