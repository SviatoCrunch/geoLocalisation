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

from .matcher import grid_keypoints, ransac_match
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
    ap.add_argument("--reproj-thresh", type=float, default=2.0)
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

    d_coarse, d_rerank = [], []
    out = {"meta": {"shortlist": args.shortlist, "pool_cells": args.pool_cells,
                    "max_pool": args.max_pool}, "per_query": {}}
    for q, entry in tqdm(sj["shortlist"].items(), desc="rerank", unit="q"):
        if not qstore.has(q):
            continue
        qf, qxy, qlat, qlon = qstore.get(q)
        if not math.isfinite(qlat):
            continue
        qf = qf.to(dev)
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
        if not pool:
            continue
        best_s, best_i = -1.0, None
        for i in pool:
            g = tiles.grid(ids[i])                              # (H,W,D)
            H, W, D = g.shape
            s = ransac_match(qf, g.reshape(H * W, D).to(dev), qxy, grid_keypoints(H, W),
                             reproj_thresh=args.reproj_thresh).score
            if s > best_s:
                best_s, best_i = s, i
        c0 = row_of.get(entry["cells"][0])
        dc = _haversine_m(qlat, qlon, lat[c0], lon[c0]) if c0 is not None else float("nan")
        dr = _haversine_m(qlat, qlon, lat[best_i], lon[best_i])
        d_coarse.append(dc); d_rerank.append(dr)
        out["per_query"][q] = {"coarse_dist_m": dc, "rerank_dist_m": dr,
                               "rerank_tile": ids[best_i], "rerank_score": best_s,
                               "pool": len(pool)}

    res = {"coarse_top1_cell": _report("coarse", d_coarse),
           "patch_rerank_top1": _report("rerank", d_rerank)}
    out["summary"] = res
    if args.out:
        Path(args.out).expanduser().write_text(json.dumps(out), encoding="utf-8")
        print(f"[ok] -> {args.out}", flush=True)
    tiles.close(); qstore.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
