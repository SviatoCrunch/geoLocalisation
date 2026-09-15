"""READ-ONLY I/O audit for the map_rerank store path - NO production behaviour is touched.

Reconstructs, per query, the exact set of (position, level) crop reads map_rerank would issue against
a `--store` (same geometry: top-k cells -> sliding window_centres -> levels -> dedup -> store.has), then
reports: HDF5 layout, reads/bytes per query, within-query vs cross-query reuse, whether pyramid levels
share a source grid (they don't, in a hybrid store), a small read-method benchmark, and the theoretical
read floor after a compact-descriptor prefilter. Purely measures; writes nothing but stdout.

Run::

    uv run --python 3.11 --with h5py --with numpy python -m patch_rerank.io_audit \
      --shortlist $OUT/kup/shortlist_kup_k70.json --dense-index $OUT/dict/tiles_index_dense.h5 \
      --store $OUT/kup/rerank_store_kup_hybrid.h5 --only-city kup \
      --k 70 --step-m 250 --levels-m 1000 900 800 700 600 500 400 300 \
      --max-queries 14 --bench-crops 600
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _fmt_gb(b):
    return f"{b / 1e9:.2f} GB" if b >= 1e9 else f"{b / 1e6:.1f} MB"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shortlist", required=True)
    ap.add_argument("--dense-index", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--levels-m", type=float, nargs="+", default=[1000, 900, 800, 700, 600, 500, 400, 300])
    ap.add_argument("--k", type=int, default=70)
    ap.add_argument("--step-m", type=float, default=250.0)
    ap.add_argument("--search-radius-m", type=float, default=0.0)
    ap.add_argument("--tile-size-m", type=float, default=1000.0)
    ap.add_argument("--only-city", default=None)
    ap.add_argument("--max-queries", type=int, default=14)
    ap.add_argument("--bench-crops", type=int, default=600, help="cap crops read in the method benchmark")
    ap.add_argument("--compact-pools", type=int, nargs="+", default=[1, 16, 64],
                    help="tokens kept per crop in a compact index (1=mean, 16=4x4, 64=8x8) for the floor estimate")
    ap.add_argument("--floor-topn", type=int, nargs="+", default=[128, 256, 512, 1024])
    args = ap.parse_args(argv)

    import h5py
    from .map_source import latlon_to_merc, true_m_to_crs   # torch-free (rasterio/cv2 are lazy inside)

    # snap / cell_window_centres inlined VERBATIM from map_rerank so the reconstructed read set is
    # identical, without importing map_rerank (which pulls torch + DINO). Keep in sync if those change.
    def snap(v, step):
        return round(v / step) * step

    def cell_window_centres(cx, cy, lat_, radius, step):
        step_crs = true_m_to_crs(step, lat_)
        rad_crs = true_m_to_crs(radius, lat_)
        offs = np.arange(-rad_crs, rad_crs + 1e-6, step_crs)
        return [(snap(cx + ox, step_crs), snap(cy + oy, step_crs)) for oy in offs for ox in offs]

    # minimal read-only store index (mirrors store.RerankStore, no torch)
    class _Store:
        def __init__(self, path):
            self.f = h5py.File(Path(path).expanduser(), "r")
            self.px = np.asarray(self.f["px"][:], float); self.py = np.asarray(self.f["py"][:], float)
            self.step_m = float(self.f.attrs["step_m"])
            self.levels_m = [float(x) for x in self.f.attrs["levels_m"]]
            self.output_px = int(self.f.attrs["output_px"])
            self.tile_size_m = float(self.f.attrs.get("tile_size_m", 1000.0))
            self._idx = {(round(float(x), 2), round(float(y), 2)): i
                         for i, (x, y) in enumerate(zip(self.px, self.py))}

        def _k(self, px, py):
            return (round(float(px), 2), round(float(py), 2))

        def has(self, px, py):
            return self._k(px, py) in self._idx

        def close(self):
            self.f.close()

    with h5py.File(Path(args.dense_index).expanduser(), "r") as f:
        ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f["tile_id"][:]]
        lat = np.asarray(f["lat"][:], float); lon = np.asarray(f["lon"][:], float)
    row_of = {t: i for i, t in enumerate(ids)}

    store = _Store(args.store)
    levels = [L for L in args.levels_m if L in store.levels_m] or store.levels_m
    radius_m = args.tile_size_m / 2.0
    if args.search_radius_m:
        radius_m = min(args.search_radius_m, args.tile_size_m / 2.0)

    # -- 1. HDF5 LAYOUT ----------------------------------------------------------------
    print("=" * 78)
    print("[1] STORE LAYOUT", args.store)  # noqa
    print(f"    n_positions(index)={len(store.px)} step_m={store.step_m} levels={store.levels_m} "
          f"output_px={store.output_px} tile_size_m={store.tile_size_m}")
    cache = store.f.id.get_access_plist().get_cache()  # (nslots, nbytes, w0)
    print(f"    file rdcc cache: nslots={cache[1]} nbytes={_fmt_gb(cache[2])} preempt_w0={cache[3]:.2f}")
    per_level_bytes = {}
    for L in store.levels_m:
        name = f"p0/l{int(L)}"
        if name not in store.f:
            continue
        d = store.f[name]
        nbytes = int(np.prod(d.shape)) * d.dtype.itemsize
        per_level_bytes[float(L)] = nbytes
        print(f"    {name}: shape={d.shape} dtype={d.dtype} chunks={d.chunks} "
              f"compression={d.compression} shuffle={getattr(d, 'shuffle', None)} bytes={_fmt_gb(nbytes)}")

    # -- 2. PER-QUERY READ SET (exact map_rerank geometry) ----------------------------
    sj = json.loads(Path(args.shortlist).expanduser().read_text())
    items = [(q, e) for q, e in sj["shortlist"].items()
             if not args.only_city or q.split(":", 1)[0] == args.only_city]
    items = items[:args.max_queries]

    all_positions = set()                                # cross-query reuse
    rows = []
    first_crop_keys = None
    for q, entry in items:
        cells = [c for c in entry["cells"][:args.k] if c in row_of]
        if not cells:
            continue
        step_crs0 = true_m_to_crs(args.step_m, lat[row_of[cells[0]]])
        seen_cell, uniq = set(), []
        for c in cells:                                  # dedup cells snapping to one grid point
            cx, cy = latlon_to_merc(lat[row_of[c]], lon[row_of[c]])
            key = (snap(cx, step_crs0), snap(cy, step_crs0))
            if key not in seen_cell:
                seen_cell.add(key); uniq.append(c)
        need = {}
        for c in uniq:
            clat = lat[row_of[c]]
            cx, cy = latlon_to_merc(clat, lon[row_of[c]])
            for (px, py) in cell_window_centres(cx, cy, clat, radius_m, args.step_m):
                need.setdefault((px, py), clat)
        present = [(px, py) for (px, py) in need if store.has(px, py)]
        crop_keys = [(store._idx[store._k(px, py)], L) for (px, py) in present for L in levels]
        bytes_q = sum(per_level_bytes.get(float(L), 0) for (_, L) in crop_keys)
        rows.append((q, len(uniq), len(need), len(present), len(need) - len(present),
                     len(crop_keys), bytes_q))
        for (px, py) in present:
            all_positions.add(store._k(px, py))
        if first_crop_keys is None and crop_keys:
            first_crop_keys = crop_keys

    print("=" * 78)
    print("[2] PER-QUERY READ SET (levels used:", [int(x) for x in levels], ")")
    print(f"    {'query':<42}{'cells':>6}{'pos':>6}{'present':>8}{'miss':>5}{'crops':>7}{'bytes':>10}")
    tot_crops, tot_bytes, tot_present = 0, 0, 0
    for (q, nc, npos, npres, nmiss, ncrops, nb) in rows:
        print(f"    {q:<42}{nc:>6}{npos:>6}{npres:>8}{nmiss:>5}{ncrops:>7}{_fmt_gb(nb):>10}")
        tot_crops += ncrops; tot_bytes += nb; tot_present += npres
    nq = len(rows)
    if nq:
        print(f"    -- mean/query: crops={tot_crops // nq} bytes={_fmt_gb(tot_bytes / nq)} "
              f"present_pos={tot_present // nq}")
        print(f"    CROSS-QUERY REUSE: {tot_present} present-pos across {nq} queries vs "
              f"{len(all_positions)} unique -> reuse x{tot_present / max(1, len(all_positions)):.2f} "
              f"(a persistent cross-query grid cache could save ~"
              f"{100 * (1 - len(all_positions) / max(1, tot_present)):.0f}% of reads over the whole run)")
    print("    WITHIN-QUERY level reuse: NONE - each (position, level) is a distinct dataset; in a HYBRID "
          "store the 8 levels are independent DINO extractions, NOT crops of one source grid.")

    # -- 3. READ-METHOD BENCHMARK (bounded sample) ------------------------------------
    print("=" * 78)
    print(f"[3] READ BENCHMARK (first query, first {args.bench_crops} crops)")
    if not first_crop_keys:
        print("    no crops to benchmark"); store.close(); return 0
    sample = first_crop_keys[:args.bench_crops]

    def _read(order):
        t0 = time.perf_counter(); nb = 0
        for (i, L) in order:
            a = np.asarray(store.f[f"p{i}/l{int(L)}"])   # raw fp16 disk read (no astype)
            nb += a.nbytes
        return time.perf_counter() - t0, nb

    natural = list(sample)
    by_row = sorted(sample, key=lambda t: (t[0], t[1]))  # sequential dataset order = written order
    for name, order in [("natural (shortlist) order", natural), ("sorted-by-dataset-row", by_row)]:
        dt, nb = _read(order)
        print(f"    {name:<26} {len(order)} reads {_fmt_gb(nb)} in {dt:.2f}s "
              f"-> {nb / 1e6 / dt:.0f} MB/s  ({1000 * dt / len(order):.1f} ms/read)")

    # -- 4. COMPACT-PREFILTER FLOOR ---------------------------------------------------
    print("=" * 78)
    print("[4] THEORETICAL READ FLOOR with a compact prefilter index (est. from mean bytes/query)")
    mean_crops = tot_crops / nq if nq else 0
    mean_full = tot_bytes / nq if nq else 0
    mean_grid_bytes = (mean_full / mean_crops) if mean_crops else 0
    D = store.f["p0/l1000"].shape[-1] if "p0/l1000" in store.f else 1024
    print(f"    baseline: ~{int(mean_crops)} crops, {_fmt_gb(mean_full)}/query (full grids)")
    for tok in args.compact_pools:
        comp_per_crop = tok * D * 2                        # fp16 compact descriptor per crop
        comp_total = comp_per_crop * mean_crops
        tag = {1: "mean-pool", 16: "4x4", 64: "8x8"}.get(tok, f"{tok}tok")
        line = f"    compact={tag} ({comp_per_crop} B/crop -> {_fmt_gb(comp_total)}/query for ALL crops)"
        floors = " | ".join(
            f"top{n}: {_fmt_gb(comp_total + n * mean_grid_bytes)}" for n in args.floor_topn)
        print(line)
        print(f"        + full grids for {tuple(args.floor_topn)} -> {floors}")
    print(f"    (vs {_fmt_gb(mean_full)} today; e.g. mean-pool + top256 ~ "
          f"{_fmt_gb((1 * D * 2) * mean_crops + 256 * mean_grid_bytes)})")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
