"""Benchmark OLD per-cell store vs NEW packed format on the SAME cell list — measured, not claimed.

Reports, for each path: wall (open+read), bytes moved, structural request count (old = 1 object GET
per cell; new = coalesced Range GETs / contiguous slices), cold vs warm, peak RAM, and (if torch+cuda)
time to a ready GPU batch. Does NOT equate one HDF5 slice with one physical GET without measurement:
request counts are the access-pattern counts; wall is timed.
"""
from __future__ import annotations

import argparse
import time
import tracemalloc

import numpy as np

from . import s3io
from .convert import Converter, read_source_cell, sorted_cell_ids
from .reader import NewFormatReader


def _old_read_all(conv, cids):
    """Old pattern: per cell -> fetch the whole cell object + read every p{i}/l{L} grid."""
    gets = 0; bytes_ = 0; blocks = []
    for cid in cids:
        local = conv._download(cid); gets += 1
        bytes_ += int(conv.index["cells"][cid].get("bytes") or __import__("os").path.getsize(local))
        blk, *_ = read_source_cell(local, conv.levels_m)
        blocks.append(blk)
        __import__("pathlib").Path(local).unlink(missing_ok=True)
    return blocks, gets, bytes_


def bench(source_prefix, dest_prefix, work_dir, n_cells=10, device="cpu"):
    conv = Converter(source_prefix, dest_prefix, work_dir)
    r = NewFormatReader(dest_prefix)
    cids = [c for c in sorted_cell_ids(conv.index) if c in r.cells][:n_cells]
    bpc = r.files[next(iter(r.files))]["bytes_per_cell"]

    tracemalloc.start()
    t = time.perf_counter(); _old, old_gets, old_bytes = _old_read_all(conv, cids)
    old_wall = time.perf_counter() - t
    old_peak = tracemalloc.get_traced_memory()[1]; tracemalloc.reset_peak()

    # NEW cold (fresh reader) then warm
    r2 = NewFormatReader(dest_prefix)
    t = time.perf_counter(); new_blocks = r2.read_cells(cids); new_cold = time.perf_counter() - t
    t = time.perf_counter(); _ = r2.read_cells(cids); new_warm = time.perf_counter() - t
    new_peak = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    # new request count: coalesced runs across files
    runs = 0; bf = {}
    for c in cids:
        fn, li = r._loc(c); bf.setdefault(fn, []).append(li)
    for fn, lis in bf.items():
        lis.sort(); runs += 1 + sum(1 for a, b in zip(lis, lis[1:]) if b != a + 1)
    new_bytes = len(cids) * bpc

    print(f"cells={len(cids)}  bytes/cell={bpc/1e6:.1f}MB  store={'S3' if s3io.is_s3(dest_prefix) else 'local'}")
    print(f"OLD per-cell : wall={old_wall:.3f}s  reqs={old_gets}  bytes={old_bytes/1e6:.1f}MB  peakRAM={old_peak/1e6:.0f}MB")
    print(f"NEW packed   : cold={new_cold:.3f}s warm={new_warm:.3f}s  reqs(runs)={runs}  bytes={new_bytes/1e6:.1f}MB  peakRAM={new_peak/1e6:.0f}MB")
    if old_wall > 0:
        print(f"speedup (old_wall / new_cold) = {old_wall/max(new_cold,1e-9):.2f}x  (MEASURED on this run)")
    if device == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                batch = np.stack(new_blocks)
                torch.cuda.synchronize(); t = time.perf_counter()
                g = torch.from_numpy(batch).to("cuda"); torch.cuda.synchronize()
                print(f"GPU batch ready: {time.perf_counter()-t:.3f}s  shape={tuple(g.shape)} dtype={g.dtype}")
        except Exception as e:
            print(f"(GPU bench skipped: {e})")
    return {"old_wall": old_wall, "new_cold": new_cold, "new_warm": new_warm,
            "old_reqs": old_gets, "new_reqs": runs, "old_bytes": old_bytes, "new_bytes": new_bytes}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-prefix", required=True)
    ap.add_argument("--destination-prefix", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--n-cells", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    bench(a.source_prefix, a.destination_prefix, a.work_dir, a.n_cells, a.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
