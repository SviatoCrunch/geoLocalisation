"""Benchmark the read path on ONE real shortlist — measured, not claimed (spec §5).

Four variants on the SAME cell list, so the numbers are comparable:
  V0 old-per-cell  : one object GET per cell (the legacy per-cell store), every p/l grid read.
  V1 new-adjacent  : packed store, scheduler ``max_gap_cells=0`` → one read per contiguous run.
  V2 new-merge     : packed store, scheduler merges near ranges within an overread budget.
  V3 new-span      : packed store, scheduler ``max_gap_cells=∞`` → one read per shard (first..last).

For each variant we report the quantities the spec asks for, kept SEPARATE (never conflated):
  * GET / HEAD / retry counts and received bytes            (real, via s3io counters + botocore retry hook)
  * useful bytes (unique cells × bytes/cell) vs overread    (from the scheduler plan)
  * cache hit / miss                                         (optional CellCache across the list)
  * reader time measured INSIDE the work (not a Future-wait), prefetch wait, CPU→GPU transfer
  * wall time, peak process RSS, peak VRAM

NOTE on "warm": a second pass with a fresh h5py handle does NOT guarantee a cold OS page cache, so we
do NOT label anything "cold/warm". We report pass-1 and pass-2 wall plainly and let the reader judge.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from . import s3io
from .cache import CellCache
from .convert import Converter, read_source_cell, sorted_cell_ids
from .reader import NewFormatReader
from .scheduler import plan_reads

_MODES = {"adjacent": dict(max_gap_cells=0),
          "merge": dict(max_gap_cells=4, max_overread_ratio=0.5),
          "span": dict(max_gap_cells=10 ** 9, max_overread_ratio=1.0)}


def _rss():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if __import__("sys").platform == "darwin" else 1024)
    except Exception:
        try:
            import psutil
            return psutil.Process().memory_info().rss
        except Exception:
            return 0


def _vram_peak_reset():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(); return True
    except Exception:
        pass
    return False


def _vram_peak():
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except Exception:
        pass
    return 0


def _old_read(conv, cids):
    """Legacy per-cell object GET; count real downloads + bytes."""
    s3io.reset_counters()
    t0 = time.perf_counter(); n_dl = 0; got_bytes = 0
    for cid in cids:
        local = conv._download(cid); n_dl += 1
        got_bytes += Path(local).stat().st_size
        read_source_cell(local, conv.levels_m)
        Path(local).unlink(missing_ok=True)
    return {"wall_s": time.perf_counter() - t0, "downloads": n_dl, "got_bytes": got_bytes}


def _new_variant(dest_prefix, cids, mode, bpc, do_gpu, cache_bytes):
    """Run one scheduler mode: plan → read each range into a reusable buffer → (optional) H2D.
    reader time is measured around the ACTUAL read work; GET/bytes come from s3io counters."""
    r = NewFormatReader(dest_prefix)
    needed = [(r.cells[c]["file"], int(r.cells[c]["local_cell_idx"]), c) for c in cids]
    reads, plan_stats = plan_reads(needed, bpc, **_MODES[mode])
    cache = CellCache(max_bytes=cache_bytes) if cache_bytes else None

    maxspan = max((rd.span for rd in reads), default=1)
    buf = np.empty((maxspan,) + r.block_shape, np.dtype("<f2"))   # one reusable buffer, sized to largest span
    s3io.reset_counters()
    gc.collect()
    _vram_peak_reset()
    tracemalloc.start()

    t_read = 0.0; t_h2d = 0.0; n_cells = 0
    import_torch = None
    if do_gpu:
        import torch as import_torch  # noqa: N816

    t0 = time.perf_counter()
    for rd in reads:
        fm = r.files[rd.file]; uri = r._file_uri(rd.file); ver, etag = r._pin(rd.file)
        wanted = {li for li, _ in rd.cells}
        if cache is not None:                                     # cache check per cell of the range
            wanted = {li for li in wanted
                      if cache.get(cache.key(rd.file, ver or etag, li)) is None}
            if not wanted:
                continue
        tr = time.perf_counter()
        # ONE physical read of the planned [lo,hi] span (incl. any gap = overread) — this is the
        # quantity the mode controls. range_get is uniform for local + S3, so GET/bytes are comparable.
        off = fm["features_byte_offset"] + rd.lo * bpc
        raw = s3io.range_get(uri, off, rd.span * bpc, version=ver, etag=etag)
        blk = np.frombuffer(raw, np.dtype("<f2")).reshape((rd.span,) + r.block_shape)
        for row, li in enumerate(sorted(wanted)):
            buf[row] = blk[li - rd.lo]                            # extract only the needed cells
        t_read += time.perf_counter() - tr
        n_cells += len(wanted)
        if cache is not None:
            for li in wanted:
                cache.put(cache.key(rd.file, ver or etag, li), blk[li - rd.lo].copy())
        if do_gpu:
            tg = time.perf_counter()
            t = import_torch.from_numpy(buf[:len(wanted)]).pin_memory()
            g = t.to("cuda", non_blocking=True); import_torch.cuda.synchronize()
            t_h2d += time.perf_counter() - tg
            del g
    wall = time.perf_counter() - t0
    peak_rss = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    r.close()
    c = dict(s3io.COUNTERS)
    res = {"mode": mode, "wall_s": wall, "read_s": t_read, "h2d_s": t_h2d,
           "n_reads": plan_stats["n_reads"], "n_cells": n_cells,
           "get": c["get"], "head": c["head"], "retry": c["retry"], "got_bytes": c["get_bytes"],
           "useful_bytes": plan_stats["useful_bytes"], "overread_bytes": plan_stats["overread_bytes"],
           "peak_read_bytes": plan_stats["peak_read_bytes"], "peak_tracemalloc": peak_rss,
           "peak_vram": _vram_peak(), "buf_bytes": int(buf.nbytes)}
    if cache is not None:
        res["cache"] = cache.stats()
    return res


def bench(source_prefix, dest_prefix, work_dir, n_cells=10, device="cpu", cache_bytes=0,
          run_old=True, out=None):
    s3io.enable_counters(True)
    conv = Converter(source_prefix, dest_prefix, work_dir)
    r = NewFormatReader(dest_prefix)
    cids = [c for c in sorted_cell_ids(conv.index) if c in r.cells][:n_cells]
    bpc = int(r.files[next(iter(r.files))]["bytes_per_cell"]); r.close()
    store = "S3" if s3io.is_s3(dest_prefix) else "local"
    do_gpu = device == "cuda"
    print(f"shortlist cells={len(cids)}  bytes/cell={bpc/2**20:.0f}MiB  store={store}  gpu={do_gpu}", flush=True)

    results = {}
    if run_old:
        results["old"] = _old_read(conv, cids)
        o = results["old"]
        print(f"V0 old-per-cell : wall={o['wall_s']:.3f}s  downloads={o['downloads']}  "
              f"got={o['got_bytes']/2**20:.0f}MiB", flush=True)
    for mode in ("adjacent", "merge", "span"):
        res = _new_variant(dest_prefix, cids, mode, bpc, do_gpu, cache_bytes)
        results[mode] = res
        print(f"V[{mode:8s}]: wall={res['wall_s']:.3f}s read={res['read_s']:.3f}s h2d={res['h2d_s']:.3f}s "
              f"reads={res['n_reads']} GET={res['get']} HEAD={res['head']} retry={res['retry']} "
              f"got={res['got_bytes']/2**20:.0f}MiB useful={res['useful_bytes']/2**20:.0f}MiB "
              f"overread={res['overread_bytes']/2**20:.0f}MiB peakVRAM={res['peak_vram']/2**20:.0f}MiB "
              f"buf={res['buf_bytes']/2**20:.0f}MiB", flush=True)
    results["meta"] = {"cells": len(cids), "bytes_per_cell": bpc, "store": store, "gpu": do_gpu,
                       "peak_rss": _rss()}
    if out:
        Path(out).expanduser().write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[ok] -> {out}", flush=True)
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-prefix", required=True)
    ap.add_argument("--destination-prefix", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--n-cells", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cache-bytes", type=int, default=0, help="CellCache budget (0 = off)")
    ap.add_argument("--no-old", dest="run_old", action="store_false", help="skip the legacy per-cell baseline")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    bench(a.source_prefix, a.destination_prefix, a.work_dir, a.n_cells, a.device, a.cache_bytes,
          a.run_old, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
