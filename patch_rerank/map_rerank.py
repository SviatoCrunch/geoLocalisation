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
import json
import math
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # reduce fragmentation

import numpy as np

from .matcher import grid_keypoints, ransac_match
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
    ap.add_argument("--prefilter-topn", type=int, default=0,
                    help="ransac only: cheap pooled-cosine GPU gate over ALL window positions, then run "
                         "MAGSAC on just the top-N positions (0=off; big speedup, positions ranked by "
                         "Σ-levels cosine)")
    ap.add_argument("--model", default="homography")
    ap.add_argument("--estimator", default="magsac")
    ap.add_argument("--reproj-thresh", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=8, help="crops per DINO forward batch (lower if OOM)")
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
                    "search_radius_m": radius_m, "scorer": args.scorer,
                    "prefilter_topn": args.prefilter_topn,
                    "model": args.model, "estimator": args.estimator, "backbone": backbone,
                    "projector": bool(proj)}, "per_query": {}}

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
        qvec = F.normalize(qfeat.mean(0, keepdim=True), dim=1)   # pooled query (cosine scorer + prefilter gate)

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
        def _score_grid(gd, h, w, mode):
            if mode == "ransac":
                return ransac_match(qfeat, gd, qxy, grid_keypoints(h, w), model=args.model,
                                    estimator=args.estimator, reproj_thresh=args.reproj_thresh).score
            cv = F.normalize(gd.mean(0, keepdim=True), dim=1)
            return float((cv @ qvec.t()).item())

        def _run(positions, mode, dst):
            """Score (position × level) for `positions` into dst[(px,py,L)] with `mode`; streams grids."""
            if store is not None:                            # precomputed grids (no DINO/map)
                for (px, py) in positions:
                    if not store.has(px, py):
                        continue
                    for L in args.levels_m:
                        g = store.grid(px, py, L); h, w, dd = g.shape
                        dst[(px, py, L)] = _score_grid(g.reshape(h * w, dd).to(dev), h, w, mode)
                return
            buf_imgs, buf_keys = [], []                      # fresh DINO from the real map
            def _flush():
                if not buf_imgs:
                    return
                for (px, py, L), g in zip(buf_keys, extract_grids(buf_imgs, ext, proj, patch, dev, amp=args.amp)):
                    h, w, dd = g.shape
                    dst[(px, py, L)] = _score_grid(g.reshape(h * w, dd).to(dev), h, w, mode)
                buf_imgs.clear(); buf_keys.clear()
            for (px, py) in positions:                       # one read/position; derive all levels
                pyr = read_pyramid_from_one(src, px, py, args.levels_m, need[(px, py)], args.output_px)
                for L in args.levels_m:
                    buf_imgs.append(pyr[L]); buf_keys.append((px, py, L))
                    if len(buf_imgs) >= args.batch:
                        _flush()
            _flush()

        score = {}
        positions = list(need.keys())
        if args.scorer == "ransac" and args.prefilter_topn > 0 and len(positions) > args.prefilter_topn:
            cos = {}
            _run(positions, "cosine", cos)                   # cheap GPU gate over ALL positions
            psum = {pp: sum(cos.get((pp[0], pp[1], L), -1e9) for L in args.levels_m) for pp in positions}
            kept = [pp for pp, _ in sorted(psum.items(), key=lambda kv: kv[1], reverse=True)[:args.prefilter_topn]]
            _run(kept, "ransac", score)                      # expensive MAGSAC only on the survivors
        else:
            _run(positions, args.scorer, score)
        crop_keys = [(px, py, L) for (px, py) in need for L in args.levels_m]

        # aggregate: per position Σ levels; cell = max; best position = fine location
        rec_cells, cell_max_sums = [], []
        best_overall = {"sum": -1e9, "lat": None, "lon": None}
        for c in uniq_cells:
            positions = []
            best_pos = {"sum": -1e9, "px": None, "py": None, "per_level": None}
            for (px, py) in cell_pos[c]:
                per_level = {int(L): score.get((px, py, L), 0.0) for L in args.levels_m}
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

        # top-N cells by rerank score (rec_cells is in coarse order) -> ranked fine predictions
        ranked = sorted(rec_cells, key=lambda r: r["cell_score"], reverse=True)[:args.topk]
        topk = [{"rank": i + 1, "cell_id": r["cell_id"], "cell_score": r["cell_score"],
                 "lat": r["best"]["lat"], "lon": r["best"]["lon"], "level_m": r["best"]["level_m"],
                 "dist_m": _haversine_m(qlat, qlon, r["best"]["lat"], r["best"]["lon"])}
                for i, r in enumerate(ranked)]

        c0 = row_of[uniq_cells[0]]
        d_coarse.append(_haversine_m(qlat, qlon, lat[c0], lon[c0]))
        d_fine.append(_haversine_m(qlat, qlon, best_overall["lat"], best_overall["lon"]))
        d_fine_topk.append(min((t["dist_m"] for t in topk), default=d_fine[-1]))  # best-of-top-N
        kmz_entries.append({"name": q, "gt": (qlat, qlon),
                            "pred": (best_overall["lat"], best_overall["lon"]), "dist_m": d_fine[-1],
                            "ranked": topk})
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()
        out["per_query"][q] = {"gt": {"lat": qlat, "lon": qlon},
                               "coarse_dist_m": d_coarse[-1], "fine_dist_m": d_fine[-1],
                               "fine_dist_topk_m": d_fine_topk[-1],
                               "mean_cell_sum": float(np.mean(cell_max_sums)) if cell_max_sums else 0.0,
                               "topk": topk, "cells": rec_cells}

    out["summary"] = {"coarse_top1_cell": _report("coarse", d_coarse),
                      "map_pyramid_fine": _report("fine", d_fine),
                      f"map_pyramid_fine_top{args.topk}": _report(f"fine@top{args.topk}", d_fine_topk)}
    Path(args.out).expanduser().write_text(json.dumps(out), encoding="utf-8")
    print(f"[ok] -> {args.out}", flush=True)
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
