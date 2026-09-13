"""Stage-A: training-free patch-RANSAC rerank of the coarse shortlist → does it localize to metres?

For each query it takes the top ``--pool-cells`` coarse cells (from ``shortlist_k70.json``), expands
them to their dense stride-250 tiles, patch-RANSAC-matches the query token grid against each dense
tile, and picks the top-1 by inlier ratio. It reports distance-based recall (distR@250/500/1000 m)
of the reranked top-1 vs the coarse top-1 (the best cell centre) — the test of whether geometric
reranking delivers metric localization without training a fine model.

Run::

    uv run --python 3.11 --with "torch==2.5.1" --with opencv-python-headless --with h5py \
           --with numpy --with tqdm python -m patch_rerank.rerank_eval \
      --shortlist /…/shortlist_k70.json \
      --queries kramatorsc=/…/query_kramatorsc_d1024.h5 kup=/…/… liman_day=/…/… \
      --dense-galleries kramatorsc=/…/map_dinov2_kramatorsc_s250m_d1024.h5 kup=/…/… liman_day=/…/… \
      --dense-index /…/dict/tiles_index_dense.h5 \
      --pool-cells 5 --max-pool 200 --device cuda --out /…/rerank_k70.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .matcher import (ESTIMATORS, GEOM_MODELS, central_mask, grid_keypoints, grid_to_latlon,
                      matched_coords, verify_inliers)
from .query_io import QueryGridStore

_R = 6378137.0                                   # EPSG:3857 sphere radius
_COS = 0.657                                     # grid→true-metre factor at ~48.9°
_THR = (250.0, 500.0, 1000.0)


def _kv(a):
    c, p = a.split("=", 1)
    return c, p


def _latlon_to_xy(lat, lon):
    x = _R * np.radians(lon)
    y = _R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return x, y


def _haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _report(name, dists):
    d = np.array(dists) if dists else np.array([np.nan])
    r = {f"distR@{int(t)}m": float((d <= t).mean()) for t in _THR}
    print(f"[{name}] n={len(dists)} median_dist_m={float(np.median(d)):.1f} "
          + " ".join(f"{k}={v:.3f}" for k, v in r.items()), flush=True)
    return {"n": len(dists), "median_dist_m": float(np.median(d)), **r}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shortlist", required=True)
    ap.add_argument("--queries", nargs="+", required=True)
    ap.add_argument("--dense-galleries", nargs="+", required=True, help="city=dense d1024 H5 (token grids)")
    ap.add_argument("--dense-index", required=True)
    ap.add_argument("--pool-cells", type=int, default=5, help="rerank dense tiles of the top-N coarse cells")
    ap.add_argument("--max-pool", type=int, default=200, help="cap dense tiles reranked per query")
    ap.add_argument("--border-m", type=float, default=250.0)
    ap.add_argument("--tile-size-m", type=float, default=1000.0)
    ap.add_argument("--levels-m", type=float, nargs="+",
                    default=[1000, 800, 600, 500, 400, 300],
                    help="concentric central-crop levels matched per tile; best (max score) wins")
    ap.add_argument("--reproj-thresh", type=float, default=2.0)
    ap.add_argument("--models", nargs="+", default=list(GEOM_MODELS),
                    help=f"geometric models to sweep (any of {GEOM_MODELS})")
    ap.add_argument("--estimators", nargs="+", default=list(ESTIMATORS),
                    help=f"RANSAC estimators to sweep (any of {ESTIMATORS})")
    ap.add_argument("--grid-size", type=int, default=0, help="resample tile token grids to g×g (0=native)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import h5py
    import torch
    from tqdm import tqdm
    from geo_e2c_train.data import TileGridLoader

    with h5py.File(Path(args.dense_index).expanduser(), "r") as f:
        ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f["tile_id"][:]]
        lat = np.asarray(f["lat"][:], float)
        lon = np.asarray(f["lon"][:], float)
        city = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f["city"][:]])
    row_of = {t: i for i, t in enumerate(ids)}
    dx, dy = _latlon_to_xy(lat, lon)
    dense_xy = np.stack([dx, dy], 1)
    reach = (args.tile_size_m / 2.0 + args.border_m) / _COS

    sj = json.loads(Path(args.shortlist).expanduser().read_text())
    qstore = QueryGridStore(dict(_kv(a) for a in args.queries))
    tiles = TileGridLoader(dict(_kv(a) for a in args.dense_galleries),
                           grid_size=(args.grid_size or None), cache=False)
    dev = args.device

    combos = [(m, e) for m in args.models for e in args.estimators]   # geom-model × estimator sweep
    d_coarse = []
    d_by = {c: [] for c in combos}
    lh_by = {c: {} for c in combos}
    out = {"meta": {"shortlist": args.shortlist, "pool_cells": args.pool_cells,
                    "max_pool": args.max_pool, "levels_m": list(args.levels_m),
                    "combos": [f"{m}/{e}" for m, e in combos]}, "per_query": {}}
    for q, entry in tqdm(sj["shortlist"].items(), desc="rerank", unit="q"):
        if not qstore.has(q):
            continue
        qf, qxy, qlat, qlon = qstore.get(q)
        if not math.isfinite(qlat):
            continue
        qf = qf.to(dev)
        n_q = int(qf.shape[0])
        pool, seen = [], set()                                  # dense tiles of top cells, cell-ranked
        for cid in entry["cells"][:args.pool_cells]:
            r = row_of.get(cid)
            if r is None:
                continue
            m = ((np.abs(dense_xy[:, 0] - dense_xy[r, 0]) <= reach)
                 & (np.abs(dense_xy[:, 1] - dense_xy[r, 1]) <= reach) & (city == city[r]))
            for i in np.nonzero(m)[0]:
                if i not in seen:
                    seen.add(i); pool.append(int(i))
            if args.max_pool and len(pool) >= args.max_pool:
                pool = pool[:args.max_pool]; break
        c0 = row_of.get(entry["cells"][0])
        if not pool or c0 is None:
            continue
        # best per combo defaults to the coarse cell centre (= no-improvement fallback)
        best = {c: {"score": -1.0, "level": None, "lat": float(lat[c0]), "lon": float(lon[c0])}
                for c in combos}
        for i in pool:
            g = tiles.grid(ids[i])                              # (H,W,D)
            H, W, D = g.shape
            tf = g.reshape(H * W, D).to(dev)
            txy = grid_keypoints(H, W)
            for L in args.levels_m:                            # concentric central-crop levels
                m = central_mask(H, W, L, args.tile_size_m)
                if int(m.sum()) < 8:
                    continue
                qm, rm, n_mut = matched_coords(qf, tf[torch.from_numpy(m)], qxy, txy[m])
                if n_mut < 4:
                    continue
                for c in combos:                               # cheap: reuse the same matched pairs
                    inl = verify_inliers(qm, rm, model=c[0], estimator=c[1],
                                         reproj_thresh=args.reproj_thresh)
                    sc = inl.shape[0] / n_q if n_q else 0.0
                    if sc > best[c]["score"]:
                        if inl.shape[0] > 0:
                            cx, cy = inl.mean(0)
                            plat, plon = grid_to_latlon(cx, cy, H, W, lat[i], lon[i], args.tile_size_m)
                        else:
                            plat, plon = float(lat[i]), float(lon[i])
                        best[c] = {"score": sc, "level": float(L), "lat": plat, "lon": plon}
        d_coarse.append(_haversine_m(qlat, qlon, lat[c0], lon[c0]))
        rec = {"coarse_dist_m": d_coarse[-1], "pool": len(pool)}
        for c in combos:
            b = best[c]
            d = _haversine_m(qlat, qlon, b["lat"], b["lon"])
            d_by[c].append(d)
            if b["level"] is not None:
                lh_by[c][b["level"]] = lh_by[c].get(b["level"], 0) + 1
            rec[f"{c[0]}/{c[1]}"] = {"dist_m": d, "score": b["score"], "level_m": b["level"]}
        out["per_query"][q] = rec

    res = {"coarse_top1_cell": _report("coarse", d_coarse)}
    best_combo, best_metric = None, -1.0
    for c in combos:
        name = f"{c[0]}/{c[1]}"
        r = _report(name, d_by[c])
        lh = {int(k): lh_by[c][k] for k in sorted(lh_by[c])}
        print(f"[levels {name}] best-level hist (m→count): {lh}", flush=True)
        r["best_level_hist_m"] = lh
        res[name] = r
        if r["distR@250m"] > best_metric:
            best_metric, best_combo = r["distR@250m"], name
    print(f"[best] geom/estimator by distR@250m = {best_combo} ({best_metric:.3f})", flush=True)
    out["summary"] = res
    out["summary"]["best_by_distR250"] = best_combo
    if args.out:
        Path(args.out).expanduser().write_text(json.dumps(out), encoding="utf-8")
        print(f"[ok] -> {args.out}", flush=True)
    tiles.close(); qstore.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
