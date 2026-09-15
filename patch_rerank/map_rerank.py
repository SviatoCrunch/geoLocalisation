"""Stage-C: real-map sliding-window pyramid reranker.

For each of the top-K coarse cells, slide a window over the REAL GeoTIFF (configurable step) around
the cell; at every window centre cut a concentric pyramid of ground crops [1000,900,…,300] m; run
each crop through DINO (same space as the query H5); score it against the query (patch-RANSAC/MAGSAC
or pooled cosine). Per window position the level scores are summed; the cell score = MAX sum over its
positions; the winning position is the fine location. Window positions are snapped to a global grid
and de-duplicated, so overlapping top cells never recompute a shared crop. Writes a JSON of levels +
tops for downstream localization.

Run::

    uv run --python 3.11 --with "torch==2.5.1" --with opencv-python-headless --with rasterio \
           --with h5py --with numpy --with tqdm python -m patch_rerank.map_rerank \
      --shortlist /…/shortlist_k70.json --dense-index /…/dict/tiles_index_dense.h5 \
      --queries kramatorsc=/…/query_kramatorsc_d1024.h5 kup=/…/… liman_day=/…/… \
      --tif-dir /home/ubuntu/work/tif --k 8 --step-m 100 --search-radius-m 500 \
      --levels-m 1000 900 800 700 600 500 400 300 --output-px 840 \
      --scorer ransac --model homography --estimator magsac --device cuda \
      --out /…/map_rerank_k8.json
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import math
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # reduce fragmentation

import numpy as np

from .matcher import (_MIN_MATCHES, grid_keypoints, matched_coords, matched_coords_batch,
                      verify_inliers)
from .map_source import (latlon_to_merc, merc_to_latlon, open_src, read_pyramid_from_one,
                         resolve_maps, true_m_to_crs)
from .map_dino import build_matched_extractor, extract_grids
from .query_io import QueryGridStore
from .store import RerankStore


def pyramid_sizes(base: float, apex: float, step: float):
    n = int(round((base - apex) / step))
    return [base - i * step for i in range(n + 1)]


def snap(v: float, step: float) -> float:
    return round(v / step) * step


def _haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def cell_window_centres(cx, cy, lat, radius_m, step_m):
    """Snapped (to a global merc grid) window centres within ±radius around a cell centre."""
    step_crs = true_m_to_crs(step_m, lat)
    rad_crs = true_m_to_crs(radius_m, lat)
    offs = np.arange(-rad_crs, rad_crs + 1e-6, step_crs)
    out = []
    for oy in offs:
        for ox in offs:
            out.append((snap(cx + ox, step_crs), snap(cy + oy, step_crs)))
    return out


def _report(name, dists):
    d = np.array(dists) if dists else np.array([np.nan])
    thr = (250.0, 500.0, 1000.0)
    r = {f"distR@{int(t)}m": float((d <= t).mean()) for t in thr}
    print(f"[{name}] n={len(dists)} median_dist_m={float(np.median(d)):.1f} "
          + " ".join(f"{k}={v:.3f}" for k, v in r.items()), flush=True)
    return {"n": len(dists), "median_dist_m": float(np.median(d)), **r}


def _kv(a):
    c, p = a.split("=", 1)
    return c, p


def _dump_scores_row(score):
    """Serialize a per-query {(px,py,L): geom_score} dict -> [[px,py,level,score], ...] for the survival
    baseline (exact per-crop scores)."""
    return [[float(px), float(py), int(L), float(s)] for (px, py, L), s in score.items()]


def _aggregate_query(q, qlat, qlon, uniq_cells, cell_pos, score, row_of, lat, lon, levels_m, topk):
    """Per-query aggregation shared by query-major and crop-major, so both are bit-identical: per
    position Sum levels; cell = max sum over its positions; best position = fine location; top-k cells.
    Returns (per_query_record, kmz_entry, coarse_dist, fine_dist, fine_dist_topk)."""
    import numpy as np
    rec_cells, cell_max_sums = [], []
    best_overall = {"sum": -1e9, "lat": None, "lon": None}
    for c in uniq_cells:
        positions = []
        best_pos = {"sum": -1e9, "px": None, "py": None, "per_level": None}
        for (px, py) in cell_pos[c]:
            per_level = {int(L): score.get((px, py, L), 0.0) for L in levels_m}
            psum = float(sum(per_level.values()))
            plat, plon = merc_to_latlon(px, py)
            positions.append({"lat": plat, "lon": plon, "sum": psum})
            if psum > best_pos["sum"]:
                best_pos = {"sum": psum, "px": px, "py": py, "per_level": per_level}
        blat, blon = merc_to_latlon(best_pos["px"], best_pos["py"])
        best_level = max(best_pos["per_level"], key=best_pos["per_level"].get)
        cell_max_sums.append(best_pos["sum"])
        rec_cells.append({"cell_id": c, "cell_score": best_pos["sum"],
                          "best": {"lat": blat, "lon": blon, "level_m": best_level,
                                   "per_level": best_pos["per_level"]},
                          "positions": positions})
        if best_pos["sum"] > best_overall["sum"]:
            best_overall = {"sum": best_pos["sum"], "lat": blat, "lon": blon}

    ranked = sorted(rec_cells, key=lambda r: r["cell_score"], reverse=True)[:topk]
    topk_l = [{"rank": i + 1, "cell_id": r["cell_id"], "cell_score": r["cell_score"],
               "lat": r["best"]["lat"], "lon": r["best"]["lon"], "level_m": r["best"]["level_m"],
               "dist_m": _haversine_m(qlat, qlon, r["best"]["lat"], r["best"]["lon"])}
              for i, r in enumerate(ranked)]
    c0 = row_of[uniq_cells[0]]
    dc = _haversine_m(qlat, qlon, lat[c0], lon[c0])
    df = _haversine_m(qlat, qlon, best_overall["lat"], best_overall["lon"])
    dft = min((t["dist_m"] for t in topk_l), default=df)
    rec = {"gt": {"lat": qlat, "lon": qlon}, "coarse_dist_m": dc, "fine_dist_m": df,
           "fine_dist_topk_m": dft,
           "mean_cell_sum": float(np.mean(cell_max_sums)) if cell_max_sums else 0.0,
           "topk": topk_l, "cells": rec_cells}
    kmz_e = {"name": q, "gt": (qlat, qlon), "pred": (best_overall["lat"], best_overall["lon"]),
             "dist_m": df, "ranked": topk_l}
    return rec, kmz_e, dc, df, dft


def _plan_query(entry, k, row_of, lat, lon, step_m, radius_m):
    """Cells (top-k, deduped by snapped centre) -> per-cell sliding window centres -> unique `need`
    positions. Identical geometry for both execution orders."""
    cells = [c for c in entry["cells"][:k] if c in row_of]
    if not cells:
        return None
    step_crs0 = true_m_to_crs(step_m, lat[row_of[cells[0]]])
    seen_cell, uniq = set(), []
    for c in cells:
        cx, cy = latlon_to_merc(lat[row_of[c]], lon[row_of[c]])
        key = (snap(cx, step_crs0), snap(cy, step_crs0))
        if key not in seen_cell:
            seen_cell.add(key); uniq.append(c)
    cell_pos, need = {}, {}
    for c in uniq:
        clat = lat[row_of[c]]
        cx, cy = latlon_to_merc(clat, lon[row_of[c]])
        centres = cell_window_centres(cx, cy, clat, radius_m, step_m)
        cell_pos[c] = centres
        for (px, py) in centres:
            need.setdefault((px, py), clat)
    return uniq, cell_pos, need


def _run_crop_major(args, store, qstore, row_of, lat, lon, radius_m, backbone, proj, out_meta):
    """EXACT batch reranker: read every unique (position,level) grid ONCE and match it against all
    queries that requested it (inverted index), instead of re-reading it per query. Same candidate
    set / MNN / MAGSAC / aggregation as query-major -> identical ranking; only physical reads drop
    (measured ~8.95x fewer bytes for the 14-kup batch). Requires --store (random grid access)."""
    import time
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    dev = args.device
    levels = args.levels_m

    sj = json.loads(Path(args.shortlist).expanduser().read_text())
    plans = []                                            # one dict per kept query (descriptors on GPU)
    for q, entry in sj["shortlist"].items():
        if args.only_city and q.split(":", 1)[0] != args.only_city:
            continue
        if args.day_only and "_night" in q:
            continue
        if not qstore.has(q):
            continue
        qfeat, qxy, qlat, qlon = qstore.get(q)
        if not math.isfinite(qlat):
            continue
        plan = _plan_query(entry, args.k, row_of, lat, lon, args.step_m, radius_m)
        if plan is None:
            continue
        uniq, cell_pos, need = plan
        qf = qfeat.to(dev)
        plans.append({"q": q, "qfeat": qf, "qxy": qxy, "qlat": qlat, "qlon": qlon,
                      "qvec": F.normalize(qf.mean(0, keepdim=True), dim=1) if args.scorer == "cosine" else None,
                      "nqf": int(qf.shape[0]), "uniq": uniq, "cell_pos": cell_pos, "need": need, "score": {}})
        if args.max_queries and len(plans) >= args.max_queries:
            break

    inv = {}                                              # (dataset_row i, level) -> [(qidx, px, py)]
    for qi, p in enumerate(plans):
        for (px, py) in p["need"]:
            if not store.has(px, py):
                continue
            i = store._idx[store._k(px, py)]
            for L in levels:
                inv.setdefault((i, int(L)), []).append((qi, px, py))
    unique_keys = sorted(inv)                             # (i, L) sorted = written/dataset order
    relations = sum(len(v) for v in inv.values())

    prof = {"read": 0.0, "match": 0.0, "verify": 0.0, "bytes": 0}
    for (i, L) in tqdm(unique_keys, desc="crop-major", unit="crop"):
        t0 = time.perf_counter()
        arr = np.asarray(store.f[f"p{i}/l{int(L)}"])       # ONE read of this unique grid
        prof["bytes"] += arr.nbytes
        g = torch.from_numpy(arr.astype(np.float32))
        h, w, dd = g.shape
        gd = g.reshape(h * w, dd).to(dev)
        prof["read"] += time.perf_counter() - t0
        kp = grid_keypoints(h, w)
        for (qi, px, py) in inv[(i, L)]:                   # match this grid vs every query needing it
            p = plans[qi]
            if args.scorer == "cosine":
                cv = F.normalize(gd.mean(0, keepdim=True), dim=1)
                p["score"][(px, py, L)] = float((cv @ p["qvec"].t()).item())
                continue
            t1 = time.perf_counter()
            qm, rm, n_mut = matched_coords(p["qfeat"], gd, p["qxy"], kp)
            prof["match"] += time.perf_counter() - t1
            if n_mut < _MIN_MATCHES[args.model]:
                p["score"][(px, py, L)] = 0.0
                continue
            t2 = time.perf_counter()
            inl = verify_inliers(qm, rm, model=args.model, estimator=args.estimator,
                                 reproj_thresh=args.reproj_thresh)
            prof["verify"] += time.perf_counter() - t2
            p["score"][(px, py, L)] = inl.shape[0] / p["nqf"] if p["nqf"] else 0.0
        if str(dev).startswith("cuda"):
            del gd

    d_coarse, d_fine, d_fine_topk, kmz_entries = [], [], [], []
    out = {"meta": out_meta, "per_query": {}}
    for p in plans:
        rec, kmz_e, dc, df, dft = _aggregate_query(p["q"], p["qlat"], p["qlon"], p["uniq"],
                                                   p["cell_pos"], p["score"], row_of, lat, lon,
                                                   levels, args.topk)
        out["per_query"][p["q"]] = rec
        kmz_entries.append(kmz_e); d_coarse.append(dc); d_fine.append(df); d_fine_topk.append(dft)
    out["summary"] = {"coarse_top1_cell": _report("coarse", d_coarse),
                      "map_pyramid_fine": _report("fine", d_fine),
                      f"map_pyramid_fine_top{args.topk}": _report(f"fine@top{args.topk}", d_fine_topk)}
    out["rerank_diagnostics"] = {
        "execution_order": "crop-major", "queries": len(plans),
        "unique_datasets_read": len(unique_keys), "physical_bytes_requested": int(prof["bytes"]),
        "query_crop_relations": relations,
        "reuse_factor": relations / max(1, len(unique_keys)),
        "timings": {"read_s": prof["read"], "match_s": prof["match"], "verify_s": prof["verify"]}}
    print(f"[crop-major] queries={len(plans)} unique_crops={len(unique_keys)} relations={relations} "
          f"reuse=x{relations / max(1, len(unique_keys)):.2f} bytes={prof['bytes'] / 1e9:.2f}GB "
          f"read={prof['read']:.1f}s match={prof['match']:.1f}s verify={prof['verify']:.1f}s", flush=True)
    Path(args.out).expanduser().write_text(json.dumps(out), encoding="utf-8")
    print(f"[ok] -> {args.out}", flush=True)
    if args.dump_scores:
        Path(args.dump_scores).expanduser().write_text(
            json.dumps({p["q"]: _dump_scores_row(p["score"]) for p in plans}), encoding="utf-8")
        print(f"[ok] per-crop scores -> {args.dump_scores}", flush=True)
    if args.kmz:
        from .kmz import write_kmz
        write_kmz(args.kmz, kmz_entries)
        print(f"[ok] kmz ({len(kmz_entries)} queries) -> {args.kmz}", flush=True)
    store.close(); qstore.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shortlist", required=True)
    ap.add_argument("--dense-index", required=True)
    ap.add_argument("--queries", nargs="+", required=True)
    ap.add_argument("--tif-dir", default=None, help="dir of GeoTIFFs resolved by city name")
    ap.add_argument("--maps", nargs="+", default=None, help="city=path.tif (overrides --tif-dir)")
    ap.add_argument("--k", type=int, default=50, help="top-K coarse cells (from the best model) refined")
    ap.add_argument("--topk", type=int, default=5, help="report the top-N cells by rerank score per query")
    ap.add_argument("--levels-m", type=float, nargs="+", default=[1000, 900, 800, 700, 600, 500, 400, 300])
    ap.add_argument("--tile-size-m", type=float, default=1000.0,
                    help="checkerboard cell size; sliding is confined to tile/2 around each cell")
    ap.add_argument("--step-m", type=float, default=100.0, help="sliding step WITHIN each top cell")
    ap.add_argument("--search-radius-m", type=float, default=0.0,
                    help="0 = auto; sliding is hard-capped to tile/2 so window centres stay in the cell")
    ap.add_argument("--output-px", type=int, default=840)
    ap.add_argument("--scorer", choices=["ransac", "cosine"], default="ransac")
    ap.add_argument("--model", default="homography")
    ap.add_argument("--estimator", default="magsac")
    ap.add_argument("--reproj-thresh", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=8, help="crops per DINO forward batch (lower if OOM)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="ransac only: CPU threads for the cv2 geometric verify (cv2 releases the GIL, "
                         "so threads scale over cores). 1=serial (default). GPU mutual-NN is batched "
                         "separately; results are identical to serial.")
    ap.add_argument("--gpu-batch", type=int, default=64,
                    help="ransac only: crops per batched GPU mutual-NN matmul (one matmul + one host "
                         "transfer instead of per-crop .cpu() syncs — the real reranker bottleneck). "
                         "Lower if GPU OOM (mem ~ gpu_batch × Nq × Nr).")
    ap.add_argument("--profile", action="store_true",
                    help="print a per-query time breakdown (store read / host→device / mutual-NN / cv2 "
                         "verify) + crop/shape/position counts, to locate the bottleneck")
    ap.add_argument("--read-jobs", type=int, default=1,
                    help="--store only: threads that prefetch grids in parallel, each with its OWN h5py "
                         "handle (store read is the reranker bottleneck, ~80%%, latency-bound). 1=serial "
                         "(default). Try #cores. Bounded prefetch → memory ~ read_jobs×4 grids.")
    ap.add_argument("--execution-order", choices=["query-major", "crop-major"], default="query-major",
                    help="query-major (default, baseline): re-read each query's grids independently. "
                         "crop-major (--store only): read each UNIQUE (position,level) grid ONCE and "
                         "match vs every query needing it (inverted index) -> identical results, far "
                         "fewer physical reads for a multi-query batch.")
    ap.add_argument("--dump-scores", default=None,
                    help="also write per-(query,position,level) exact geometric scores to this JSON "
                         "(the baseline for compact_survival). {query: [[px,py,level,score],...]}")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--proj-seed", type=int, default=0)
    ap.add_argument("--store", default=None,
                    help="precomputed rerank token-grid store (precompute_store.py) → no fresh DINO")
    ap.add_argument("--max-queries", type=int, default=0, help="cap queries processed (0=all; smoke run)")
    ap.add_argument("--only-city", default=None, help="process only queries of this city (e.g. kup)")
    ap.add_argument("--day-only", action="store_true", help="skip *_night queries (map is day-only)")
    ap.add_argument("--kmz", default=None, help="also write a KMZ (GT + predicted point per query)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import h5py
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    qpaths = dict(_kv(a) for a in args.queries)
    with h5py.File(Path(args.dense_index).expanduser(), "r") as f:
        ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f["tile_id"][:]]
        lat = np.asarray(f["lat"][:], float); lon = np.asarray(f["lon"][:], float)
    row_of = {t: i for i, t in enumerate(ids)}

    cities = sorted(qpaths)
    store = RerankStore(args.store) if args.store else None
    if store is not None:                                 # align geometry to the store; no DINO/maps
        args.step_m = store.step_m
        args.output_px, args.tile_size_m = store.output_px, store.tile_size_m
        req = [L for L in args.levels_m if L in store.levels_m]   # honor a --levels-m SUBSET (speed)
        args.levels_m = req if req else store.levels_m
        ext = proj = patch = None
        backbone = str(store.f.attrs.get("backbone", "?"))
        srcs = {}
        print(f"[store] {args.store} step={store.step_m} levels(use)={args.levels_m} "
              f"of stored {store.levels_m} output_px={store.output_px} backbone={backbone}", flush=True)
    else:
        maps = dict(_kv(a) for a in args.maps) if args.maps else resolve_maps(args.tif_dir, cities)
        missing = [c for c in cities if c not in maps]
        if missing:
            raise SystemExit(f"no GeoTIFF resolved for cities {missing} (use --maps city=path.tif)")
        ext, proj, patch, backbone, D = build_matched_extractor(qpaths[cities[0]], args.device, args.proj_seed)
        print(f"[dino] backbone={backbone} D={D} projector={'yes' if proj else 'NO'} patch={patch}", flush=True)
        srcs = {c: open_src(maps[c]) for c in cities}
    qstore = QueryGridStore(qpaths)
    dev = args.device

    # sliding is confined to WITHIN each top cell: radius hard-capped to tile/2 (per user spec)
    radius_m = args.tile_size_m / 2.0
    if args.search_radius_m:
        radius_m = min(args.search_radius_m, args.tile_size_m / 2.0)

    sj = json.loads(Path(args.shortlist).expanduser().read_text())
    d_coarse, d_fine, d_fine_topk, kmz_entries = [], [], [], []
    out = {"meta": {"k": args.k, "topk": args.topk, "levels_m": list(args.levels_m), "step_m": args.step_m,
                    "search_radius_m": radius_m, "scorer": args.scorer, "jobs": args.jobs,
                    "gpu_batch": args.gpu_batch, "read_jobs": args.read_jobs,
                    "execution_order": args.execution_order,
                    "model": args.model, "estimator": args.estimator, "backbone": backbone,
                    "projector": bool(proj)}, "per_query": {}}

    if args.execution_order == "crop-major":
        if store is None:
            raise SystemExit("--execution-order crop-major requires --store (random grid access)")
        return _run_crop_major(args, store, qstore, row_of, lat, lon, radius_m, backbone, proj, out["meta"])

    dumped = {}                                          # query -> per-crop exact scores (--dump-scores)
    items = list(sj["shortlist"].items())
    for q, entry in tqdm(items, desc="map-rerank", unit="q"):
        if args.max_queries and len(d_fine) >= args.max_queries:
            break
        if args.only_city and q.split(":", 1)[0] != args.only_city:
            continue
        if args.day_only and "_night" in q:
            continue
        if not qstore.has(q):
            continue
        city = q.split(":", 1)[0]
        src = srcs.get(city)
        if store is None and src is None:                # with --store no GeoTIFF is needed
            continue
        qfeat, qxy, qlat, qlon = qstore.get(q)
        if not math.isfinite(qlat):
            continue
        qfeat = qfeat.to(dev)
        qvec = F.normalize(qfeat.mean(0, keepdim=True), dim=1) if args.scorer == "cosine" else None

        cells = [c for c in entry["cells"][:args.k] if c in row_of]
        # dedup cells whose centres snap to the same grid point
        step_crs0 = true_m_to_crs(args.step_m, lat[row_of[cells[0]]]) if cells else 1.0
        seen_cell, uniq_cells = set(), []
        for c in cells:
            cx, cy = latlon_to_merc(lat[row_of[c]], lon[row_of[c]])
            key = (snap(cx, step_crs0), snap(cy, step_crs0))
            if key not in seen_cell:
                seen_cell.add(key); uniq_cells.append(c)

        cell_pos = {}                                   # cell -> list of snapped merc centres
        need = {}                                       # (cx,cy) -> lat  (unique window centres)
        for c in uniq_cells:
            clat = lat[row_of[c]]
            cx, cy = latlon_to_merc(clat, lon[row_of[c]])
            centres = cell_window_centres(cx, cy, clat, radius_m, args.step_m)
            cell_pos[c] = centres
            for (px, py) in centres:
                need.setdefault((px, py), clat)

        # STREAMING: score each crop right after DINO, keep only SCALARS. No grid accumulation →
        # RAM is O(#scalars) + one batch (a big-city query has ~5000 crops; holding grids = ~14 GB
        # → host OOM). Positions are deduped per query (`need`), so each unique crop is scored once.
        prof = {"read": 0.0, "h2d": 0.0, "match": 0.0, "verify": 0.0,
                "crops": 0, "shapes": set(), "pos": 0, "miss": 0}

        def _grid_iter():
            """Yield (px, py, L, gd, h, w) for every crop, streaming from the store or fresh DINO."""
            if store is not None and args.read_jobs > 1:     # PARALLEL prefetch (handle per thread)
                keys = [(px, py, L) for (px, py) in need if store.has(px, py) for L in args.levels_m]
                prof["pos"] = sum(1 for (px, py) in need if store.has(px, py))
                prof["miss"] = len(need) - prof["pos"]
                tls = threading.local()
                def _read(key):
                    hnd = getattr(tls, "h", None)
                    if hnd is None:
                        hnd = tls.h = store.open_handle()    # one h5py handle per worker thread
                    return key, store.grid_from(hnd, *key)
                with cf.ThreadPoolExecutor(max_workers=args.read_jobs) as ex:
                    it = iter(keys)
                    inflight = collections.deque()           # bounded read-ahead window
                    for _ in range(args.read_jobs * 4):
                        k = next(it, None)
                        if k is None:
                            break
                        inflight.append(ex.submit(_read, k))
                    while inflight:
                        t0 = time.perf_counter()
                        (px, py, L), g = inflight.popleft().result()   # wait = read-bound stall
                        prof["read"] += time.perf_counter() - t0
                        k = next(it, None)
                        if k is not None:
                            inflight.append(ex.submit(_read, k))
                        h, w, dd = g.shape
                        t1 = time.perf_counter()
                        gd = g.reshape(h * w, dd).to(dev)
                        prof["h2d"] += time.perf_counter() - t1
                        prof["crops"] += 1; prof["shapes"].add((h, w))
                        yield px, py, L, gd, h, w
                return
            if store is not None:                            # serial precomputed grids
                for (px, py) in need:
                    if not store.has(px, py):
                        prof["miss"] += 1
                        continue
                    prof["pos"] += 1
                    for L in args.levels_m:
                        t0 = time.perf_counter()
                        g = store.grid(px, py, L); h, w, dd = g.shape
                        gf = g.reshape(h * w, dd)            # H5 read + decompress + numpy/torch convert
                        t1 = time.perf_counter()
                        gd = gf.to(dev)                      # host → device copy (async; timed coarsely)
                        prof["read"] += t1 - t0; prof["h2d"] += time.perf_counter() - t1
                        prof["crops"] += 1; prof["shapes"].add((h, w))
                        yield px, py, L, gd, h, w
                return
            buf_imgs, buf_keys = [], []                       # fresh DINO from the real map (batched)
            def _extract():
                t0 = time.perf_counter()
                grids = extract_grids(buf_imgs, ext, proj, patch, dev, amp=args.amp)
                prof["read"] += time.perf_counter() - t0
                for (bx, by, bL), g in zip(buf_keys, grids):
                    h, w, dd = g.shape
                    prof["crops"] += 1; prof["shapes"].add((h, w))
                    yield bx, by, bL, g.reshape(h * w, dd).to(dev), h, w
                buf_imgs.clear(); buf_keys.clear()
            for (px, py) in need:                            # one read/position; derive all levels
                prof["pos"] += 1
                pyr = read_pyramid_from_one(src, px, py, args.levels_m, need[(px, py)], args.output_px)
                for L in args.levels_m:
                    buf_imgs.append(pyr[L]); buf_keys.append((px, py, L))
                    if len(buf_imgs) >= args.batch:
                        yield from _extract()
            if buf_imgs:
                yield from _extract()

        score = {}
        if args.scorer == "cosine":
            for px, py, L, gd, h, w in _grid_iter():
                cv = F.normalize(gd.mean(0, keepdim=True), dim=1)
                score[(px, py, L)] = float((cv @ qvec.t()).item())
        else:
            # ransac: BATCHED GPU mutual-NN (one matmul + one host transfer per gpu_batch crops,
            # killing the per-crop .cpu() sync latency) → tiny point pairs → cv2 verify (serial, or
            # threaded with --jobs since cv2 releases the GIL). Grids held only within a batch.
            n_q = int(qfeat.shape[0])
            pool = cf.ThreadPoolExecutor(max_workers=args.jobs) if (args.jobs and args.jobs > 1) else None

            def _verify(item):
                key, qm, rm, n_mutual = item
                if n_mutual < _MIN_MATCHES[args.model]:
                    return key, 0.0
                inl = verify_inliers(qm, rm, model=args.model, estimator=args.estimator,
                                     reproj_thresh=args.reproj_thresh)
                return key, (inl.shape[0] / n_q if n_q else 0.0)

            bufs = {}                                        # (h,w) -> [keys, grids]; group by shape,
                                                             # since pyramid levels have DIFFERENT grids

            def _drain(hw):
                keys, grids = bufs[hw]
                if not grids:
                    return
                t0 = time.perf_counter()
                mc = matched_coords_batch(qfeat, torch.stack(grids), qxy, grid_keypoints(hw[0], hw[1]))
                t1 = time.perf_counter(); prof["match"] += t1 - t0     # incl .cpu() sync (real GPU time)
                items = [(keys[i], mc[i][0], mc[i][1], mc[i][2]) for i in range(len(keys))]
                res = pool.map(_verify, items) if pool else map(_verify, items)
                for key, s in res:
                    score[key] = s
                prof["verify"] += time.perf_counter() - t1
                keys.clear(); grids.clear()

            for px, py, L, gd, h, w in _grid_iter():
                keys, grids = bufs.setdefault((h, w), [[], []])
                keys.append((px, py, L)); grids.append(gd)
                if len(grids) >= args.gpu_batch:
                    _drain((h, w))
            for hw in list(bufs):
                _drain(hw)
            if pool is not None:
                pool.shutdown()
        crop_keys = [(px, py, L) for (px, py) in need for L in args.levels_m]

        if args.profile:
            tot = prof["read"] + prof["h2d"] + prof["match"] + prof["verify"]
            pct = lambda x: f"{100 * x / tot:.0f}%" if tot else "0%"
            tqdm.write(
                f"[prof {q}] read={prof['read']:.1f}s({pct(prof['read'])}) "
                f"h2d={prof['h2d']:.1f}s({pct(prof['h2d'])}) "
                f"match={prof['match']:.1f}s({pct(prof['match'])}) "
                f"verify={prof['verify']:.1f}s({pct(prof['verify'])}) sum={tot:.1f}s | "
                f"crops={prof['crops']} pos={prof['pos']} miss={prof['miss']} "
                f"shapes={sorted(prof['shapes'])} cells={len(uniq_cells)}")

        rec, kmz_e, dc, df, dft = _aggregate_query(q, qlat, qlon, uniq_cells, cell_pos, score,
                                                   row_of, lat, lon, args.levels_m, args.topk)
        out["per_query"][q] = rec
        kmz_entries.append(kmz_e); d_coarse.append(dc); d_fine.append(df); d_fine_topk.append(dft)
        if args.dump_scores:
            dumped[q] = _dump_scores_row(score)
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()

    out["summary"] = {"coarse_top1_cell": _report("coarse", d_coarse),
                      "map_pyramid_fine": _report("fine", d_fine),
                      f"map_pyramid_fine_top{args.topk}": _report(f"fine@top{args.topk}", d_fine_topk)}
    Path(args.out).expanduser().write_text(json.dumps(out), encoding="utf-8")
    print(f"[ok] -> {args.out}", flush=True)
    if args.dump_scores:
        Path(args.dump_scores).expanduser().write_text(json.dumps(dumped), encoding="utf-8")
        print(f"[ok] per-crop scores -> {args.dump_scores}", flush=True)
    if args.kmz:
        from .kmz import write_kmz
        write_kmz(args.kmz, kmz_entries)
        print(f"[ok] kmz ({len(kmz_entries)} queries) -> {args.kmz}", flush=True)
    for s in srcs.values():
        s.close()
    if store is not None:
        store.close()
    qstore.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
