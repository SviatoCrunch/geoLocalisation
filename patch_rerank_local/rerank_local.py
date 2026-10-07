"""Isolated local fine-reranker: local cell pyramids × query grids → ranked cells, on GPU or CPU.

The no-S3 sibling of ``patch_rerank.search_pyramid_s3``. It reads cell pyramids from local disk
(:class:`LocalCellStore`) — the production case where the map is downloaded once — and reranks each
query's coarse shortlist by the same training-free patch-RANSAC score (``inliers / n_query_patches``)
with the same level→position→cell aggregation.

Only the data IO is reimplemented (local, no boto3, no diagnostics archive). The scoring
(:func:`_load_cell`, :func:`_score_loaded`, :func:`_aggregate`) is copied verbatim from
``search_pyramid_s3`` so numbers match the canonical run bit-for-bit, and the geometry primitives are
the shared, pure (no-IO) ``patch_rerank.matcher``.

``--verify-backend`` is ``cpu_magsac`` only (cv2 USAC MAGSAC, the canonical path). The valid GPU
MAGSAC++ path lives in :mod:`magsacpp_torch.search` (its ``--store-dir`` reads the same local store);
the gpu_batch / gpu_kornia verifiers were removed. ``--device`` selects cuda/cpu for the DINO
mutual-NN matmul.

Run::

    uv run --python 3.11 --with "torch==2.5.1" --with h5py --with numpy --with tqdm \
           --with opencv-python-headless python -m patch_rerank_local.rerank_local \
      --queries kup=/…/query_kup_d1024.h5 \
      --shortlist /…/shortlist_kup_prod_k100.json \
      --cells-dir /…/pyramid_dinov2vitg14_p448/kup   --index /…/pyramid_…/kup/_index.json \
      --k-coarse 100 --topk 5 --level-agg sum --cell-agg mean \
      --execution-order cell-major --verify-backend cpu_magsac --device cuda \
      --dump-scores /…/kup_scores_sum.json --out /…/kup_search_sum.json

Then the top-K curve::  python -m patch_rerank_local.curve --search …/kup_search_sum.json \
                                                             --dump …/kup_scores_sum.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np


def _haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _load_cell(cd, device):
    """Read every (position, level) grid of a cell to ``device`` ONCE → (grids (Ncrops,N,D), keys, kp).
    Reused across all queries in cell-major so a cell's tokens are loaded a single time.
    (Verbatim from ``patch_rerank.search_pyramid_s3`` — keep in sync.)"""
    import torch

    from patch_rerank.matcher import grid_keypoints
    keys, grids, kp = [], [], None
    for i in range(cd.n_pos):
        for L in cd.levels:
            g = cd.grid(i, L)
            h, w, D = g.shape
            if kp is None:
                kp = grid_keypoints(h, w)
            grids.append(g.reshape(h * w, D).to(device)); keys.append((i, L))
    return torch.stack(grids), keys, kp


def _score_loaded(grids, keys, kp, cd, qfeat, qxy, n_q, args, device):
    """Score a cell's pre-loaded grids against one query → (pyramid_score[n_pos], best_level[n_pos]).

    ``--verify-backend`` picks the verifier; ``--level-agg`` combines a position's per-level scores by
    max (best level) or sum (multi-scale). Grids matched/verified in --grid-chunk sub-batches.
    (Verbatim from ``search_pyramid_s3`` minus the diagnostics-archive branch — keep in sync.)"""
    from patch_rerank.matcher import _MIN_MATCHES, matched_coords_batch, verify_inliers
    _MIN = _MIN_MATCHES["homography"]
    if args.verify_backend != "cpu_magsac":                          # GPU MAGSAC++ = magsacpp_torch.search
        raise ValueError("patch_rerank_local supports only cpu_magsac; the valid GPU MAGSAC++ "
                         "path is magsacpp_torch.search --store-dir")
    sc = {}
    ch = max(1, args.grid_chunk)
    for s in range(0, len(keys), ch):
        gsub, ksub = grids[s:s + ch], keys[s:s + ch]
        mc = matched_coords_batch(qfeat, gsub, qxy, kp)              # cv2 MAGSAC on host (canonical)
        for j, (i, L) in enumerate(ksub):
            qm, rm, n_mut = mc[j]
            if n_q and n_mut >= _MIN:
                inl = verify_inliers(qm, rm, model="homography", estimator="magsac",
                                     reproj_thresh=args.reproj_thresh)
                sc[(i, L)] = inl.shape[0] / n_q
            else:
                sc[(i, L)] = 0.0
    pyr, blv = [], []
    for i in range(cd.n_pos):
        per = {L: sc[(i, L)] for L in cd.levels}
        bL = max(per, key=per.get)                                   # reported level = best-scoring level
        pyr.append(sum(per.values()) if args.level_agg == "sum" else per[bL])
        blv.append(int(bL))
    return pyr, blv


def _aggregate(pyr, blv, cd, cell_agg):
    """pyramid scores → cell score (mean|min|max) + best pyramid (point) & its level. (Verbatim.)"""
    arr = np.asarray(pyr, float)
    cs = {"mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max())}[cell_agg]
    bi = int(arr.argmax())                                    # point = best pyramid in the cell
    return cs, bi, float(cd.lat[bi]), float(cd.lon[bi]), int(blv[bi])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", nargs="+", required=True, help="city=query_grids.h5 (DINO token grids)")
    ap.add_argument("--shortlist", required=True, help="coarse top-N cells per query (matching cell ids)")
    ap.add_argument("--cells-dir", required=True, help="local dir of cell H5s (root with _index.json / cells/)")
    ap.add_argument("--index", default=None, help="explicit _index.json (default: <cells-dir>/_index.json)")
    ap.add_argument("--open-cap", type=int, default=16, help="max open cell file handles (fd bound)")
    ap.add_argument("--k-coarse", type=int, default=100)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--cell-agg", choices=["mean", "min", "max"], default="mean")
    ap.add_argument("--level-agg", choices=["max", "sum"], default="sum")
    ap.add_argument("--execution-order", choices=["cell-major", "query-major"], default="cell-major",
                    help="cell-major: read each cell ONCE, score all its queries (default). "
                         "query-major: per query read its cells.")
    ap.add_argument("--verify-backend", choices=["cpu_magsac"], default="cpu_magsac",
                    help="cpu_magsac cv2 USAC MAGSAC (only backend here; GPU MAGSAC++ = magsacpp_torch.search)")
    ap.add_argument("--grid-chunk", type=int, default=64)
    ap.add_argument("--reproj-thresh", type=float, default=2.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only-city", default=None)
    ap.add_argument("--day-only", action="store_true")
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--dump-scores", default=None, help="JSON {q:{cell:{mean,best,lat,lon,level_m}}} (ALL cands)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import time

    import torch
    from tqdm import tqdm

    from patch_rerank.query_io import QueryGridStore

    from .local_cell_store import LocalCellStore

    qstore = QueryGridStore(dict(a.split("=", 1) for a in args.queries))
    store = LocalCellStore(args.cells_dir, index_path=args.index, open_cap=args.open_cap)
    sj = json.loads(Path(args.shortlist).expanduser().read_text(encoding="utf-8"))
    sh = sj.get("shortlist", sj)
    dev = args.device

    out = {"meta": {"k_coarse": args.k_coarse, "topk": args.topk, "cell_agg": args.cell_agg,
                    "level_agg": args.level_agg, "verify_backend": args.verify_backend,
                    "execution_order": args.execution_order, "device": dev,
                    "cells_dir": str(args.cells_dir)}, "per_query": {}}
    per_query, d_fine, d_ftop = {}, [], []
    t_fetch_all = t_score_all = 0.0
    dumped = {} if args.dump_scores else None

    def _rec(cid, pyr, blv, cd):
        cs, bi, blat, blon, bL = _aggregate(pyr, blv, cd, args.cell_agg)
        return {"cell_id": cid, "cell_score": cs, "lat": blat, "lon": blon, "level_m": bL,
                "best_score": float(max(pyr)), "mean_score": float(np.mean(pyr))}

    plans = {}
    for q, entry in sh.items():
        if args.only_city and q.split(":", 1)[0] != args.only_city:
            continue
        if args.day_only and "_night" in q:
            continue
        if not qstore.has(q):
            continue
        if args.max_queries and len(plans) >= args.max_queries:
            break
        qfeat, qxy, qlat, qlon = qstore.get(q)
        top = entry["cells"][:args.k_coarse]
        cands = [c for c in top if store.has(c)]
        plans[q] = {"qfeat": qfeat.to(dev), "qxy": qxy, "qlat": qlat, "qlon": qlon,
                    "n_q": int(qfeat.shape[0]), "cands": cands, "coarse_top": top}
    qcell = {q: {} for q in plans}
    tscore = {q: 0.0 for q in plans}

    if args.execution_order == "cell-major":
        inv = collections.defaultdict(list)
        for q, p in plans.items():
            for cid in p["cands"]:
                inv[cid].append(q)
        for cid in tqdm(sorted(inv), desc="cells", unit="cell"):
            t0 = time.perf_counter(); cd = store.cell(cid); t_fetch_all += time.perf_counter() - t0
            grids, keys, kp = _load_cell(cd, dev)
            for q in inv[cid]:
                p = plans[q]
                t1 = time.perf_counter()
                pyr, blv = _score_loaded(grids, keys, kp, cd, p["qfeat"], p["qxy"], p["n_q"], args, dev)
                dt = time.perf_counter() - t1; t_score_all += dt; tscore[q] += dt
                qcell[q][cid] = _rec(cid, pyr, blv, cd)
            del grids
            if str(dev).startswith("cuda"):
                torch.cuda.empty_cache()
    else:                                                     # query-major
        for q, p in tqdm(plans.items(), desc="search", unit="q"):
            for cid in p["cands"]:
                t0 = time.perf_counter(); cd = store.cell(cid); t_fetch_all += time.perf_counter() - t0
                grids, keys, kp = _load_cell(cd, dev)
                t1 = time.perf_counter()
                pyr, blv = _score_loaded(grids, keys, kp, cd, p["qfeat"], p["qxy"], p["n_q"], args, dev)
                dt = time.perf_counter() - t1; t_score_all += dt; tscore[q] += dt
                qcell[q][cid] = _rec(cid, pyr, blv, cd)
                del grids
            if str(dev).startswith("cuda"):
                torch.cuda.empty_cache()

    for q, p in plans.items():
        recs = list(qcell[q].values())
        ranked = sorted(recs, key=lambda r: r["cell_score"], reverse=True)[:args.topk]
        has_gt = math.isfinite(p["qlat"]) and math.isfinite(p["qlon"])
        for i, r in enumerate(ranked):
            r["rank"] = i + 1
            r["dist_m"] = _haversine_m(p["qlat"], p["qlon"], r["lat"], r["lon"]) if has_gt else None
        df = ranked[0]["dist_m"] if (ranked and has_gt) else None
        dft = min((r["dist_m"] for r in ranked), default=None) if has_gt else None
        rec = {"gt": [p["qlat"], p["qlon"]] if has_gt else None, "fine_dist_m": df,
               "fine_dist_topk_m": dft, "n_candidates": len(p["cands"]), "topk": ranked,
               "timings": {"score_s": tscore[q]}}
        per_query[q] = rec; out["per_query"][q] = rec
        if dumped is not None:
            dumped[q] = {r["cell_id"]: {"mean": r["mean_score"], "best": r["best_score"],
                                        "lat": r["lat"], "lon": r["lon"], "level_m": r["level_m"]}
                         for r in recs}
        if has_gt:
            d_fine.append(df); d_ftop.append(dft)

    out["speed"] = {"queries": len(per_query), "fetch_s_total": t_fetch_all, "score_s_total": t_score_all,
                    "s_per_query": (t_fetch_all + t_score_all) / max(1, len(per_query))}
    print(f"[speed] queries={len(per_query)} fetch={t_fetch_all:.1f}s score={t_score_all:.1f}s "
          f"| {(t_fetch_all + t_score_all) / max(1, len(per_query)):.1f}s/q", flush=True)
    Path(args.out).expanduser().write_text(json.dumps(out), encoding="utf-8")
    print(f"[ok] -> {args.out}", flush=True)
    if dumped is not None:
        Path(args.dump_scores).expanduser().write_text(json.dumps(dumped), encoding="utf-8")
        print(f"[ok] cell scores -> {args.dump_scores}", flush=True)
    store.close(); qstore.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
