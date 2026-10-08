"""Sequential, batched GPU rerank over the packed new format — tuned for throughput-capped disks.

Why this shape (measured on the EBS volume): scattered per-cell reads ≈ 140 MB/s, large SEQUENTIAL
reads ≈ 200 MB/s, and PARALLEL reads are SLOWER (contention). So:
  * order the needed cells by (file, local_cell_idx) and read each chunk as ONE contiguous
    ``ds[lo:hi]`` span  -> minimal disk requests, max sequential throughput;
  * a single 1-chunk-ahead PREFETCH thread reads the next span while the GPU scores the current one
    (read is ~65% of wall, so overlap hides the GPU/compute part);
  * per chunk: ONE batched mutual-NN + ONE batched MAGSAC++ over ALL the chunk's crops per query
    (reuses magsacpp_torch primitives — identical math), then aggregate level->pos->cell and record.

Reads are capped by the volume (~200 MB/s sequential); this reaches that floor and removes the
per-cell Python loop + scattered reads. For lower wall: lighter embeddings / faster storage.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def _haversine_m(a, b, c, d):
    r = 6371000.0
    p1, p2 = math.radians(a), math.radians(c)
    dp, dl = math.radians(c - a), math.radians(d - b)
    x = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(x)))


def _candidates(ind, qid):
    if isinstance(ind, dict) and "shortlist" in ind:
        return (ind["shortlist"].get(qid) or {}).get("cells", [])
    v = ind.get(qid, [])
    return v.get("cells", []) if isinstance(v, dict) else v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", nargs="+", required=True, help="city=query_grids.h5 (1..N frames)")
    ap.add_argument("--indices", required=True, help="JSON {qid:[idx...]} or shortlist; idx=cell_id|global_cell_idx")
    ap.add_argument("--dataset-dir", required=True, help="new-format store dir (local)")
    ap.add_argument("--read-chunk-cells", type=int, default=16, dest="rck",
                    help="cells per sequential read span (also the GPU sub-batch); RAM ~= 2*rck*419MB")
    ap.add_argument("--k-coarse", type=int, default=1000)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--level-agg", choices=["sum", "max"], default="sum")
    ap.add_argument("--cell-agg", choices=["mean", "min", "max"], default="mean")
    ap.add_argument("--reproj-thresh", type=float, default=2.0)
    ap.add_argument("--mpp-sigma-max", type=float, default=2.0, dest="mpp_sigma_max")
    ap.add_argument("--mpp-hyps", type=int, default=256, dest="mpp_hyps")
    ap.add_argument("--mpp-irls", type=int, default=1, dest="mpp_irls")
    ap.add_argument("--mpp-dtype", choices=["float32", "float64"], default="float32", dest="mpp_dtype")
    ap.add_argument("--mpp-solver", choices=["svd", "closed_form"], default="closed_form", dest="mpp_solver")
    ap.add_argument("--mpp-seed", type=int, default=0, dest="mpp_seed")
    ap.add_argument("--match-budget", type=int, default=50_000_000, dest="match_budget")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    for p in (str(repo), str(repo / "experiments")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch

    from magsacpp_torch import MagsacppConfig
    from magsacpp_torch.batch import estimate_homography_magsacpp_batch
    from magsacpp_torch.search import _mutual_nn_padded
    from patch_rerank.matcher import grid_keypoints
    from patch_rerank.query_io import QueryGridStore

    from .reader import NewFormatReader

    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    cfg = MagsacppConfig(sigma_max=args.mpp_sigma_max, inlier_threshold=args.reproj_thresh,
                         max_hypotheses=args.mpp_hyps, irls_iters=args.mpp_irls,
                         minimal_solver=args.mpp_solver,
                         dtype=torch.float64 if args.mpp_dtype == "float64" else torch.float32)
    gen = torch.Generator(device=dev).manual_seed(args.mpp_seed)

    r = NewFormatReader(args.dataset_dir)
    P, L, H, W, D = r.P, r.L, r.H, r.W, r.D
    levels = r.levels_m
    kp = grid_keypoints(H, W)
    by_gidx = {int(c["global_cell_idx"]): cid for cid, c in r.cells.items()}

    def resolve(x):
        return by_gidx[int(x)] if (isinstance(x, (int, np.integer)) or str(x).lstrip("-").isdigit()) else x

    qs = QueryGridStore(dict(a.split("=", 1) for a in args.queries))
    ind = json.loads(Path(args.indices).expanduser().read_text())
    qids_all = list(ind.get("shortlist", ind).keys())

    Q, cand = {}, {}
    for qid in qids_all:
        if not qs.has(qid):
            continue
        if args.max_queries and len(Q) >= args.max_queries:
            break
        feat, xy, lat, lon = qs.get(qid)
        cells = [resolve(x) for x in _candidates(ind, qid)][:args.k_coarse]
        cells = [c for c in cells if c in r.cells]
        if not cells:
            continue
        Q[qid] = {"feat": feat.to(dev), "xy": xy, "nq": int(feat.shape[0]), "lat": lat, "lon": lon}
        cand[qid] = set(cells)
    if not Q:
        print("[search_seq] no queries matched"); return 2

    union = sorted(set().union(*cand.values()),
                   key=lambda c: (r.cells[c]["file"], int(r.cells[c]["local_cell_idx"])))
    need = {c: [q for q in Q if c in cand[q]] for c in union}
    qcell = {q: {} for q in Q}

    chunks = [union[i:i + args.rck] for i in range(0, len(union), args.rck)]

    def read_chunk(sub):                                       # ONE sequential ds[lo:hi] span per run;
        arr = np.stack(r.read_cells(sub))                      # features + per-pos coords read in THIS
        co = [r.read_cell_coords(c) for c in sub]              # thread (h5py not thread-safe -> no sharing)
        return sub, arr, co

    t_read = t_gpu = 0.0
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(read_chunk, chunks[0]) if chunks else None
    t0 = time.perf_counter()
    for ci in range(len(chunks)):
        tr = time.perf_counter(); sub, arr, co = fut.result(); t_read += time.perf_counter() - tr
        fut = ex.submit(read_chunk, chunks[ci + 1]) if ci + 1 < len(chunks) else None   # prefetch next
        tg = time.perf_counter()
        C = len(sub)
        grids = torch.from_numpy(arr).reshape(C * P * L, H * W, D).to(dev)   # one H2D, fp16 (matcher casts)
        qset = {q for c in sub for q in need[c]}
        for q in qset:
            p = Q[q]
            qm, rm, valid = _mutual_nn_padded(p["feat"], grids, p["xy"], kp, dev, args.match_budget)
            counts, _ = estimate_homography_magsacpp_batch(rm, qm, valid, config=cfg, generator=gen,
                                                           return_counts=True)
            s = (counts.double() / p["nq"] if p["nq"] else counts.double()).view(C, P, L)
            pyr = s.sum(2) if args.level_agg == "sum" else s.amax(2)         # (C, P)
            for k, cid in enumerate(sub):
                if cid not in cand[q]:
                    continue
                prow = pyr[k]
                bi = int(prow.argmax())
                cs = {"mean": float(prow.mean()), "min": float(prow.min()), "max": float(prow.max())}[args.cell_agg]
                lvl_i = int(s[k, bi].argmax())
                plat, plon = co[k]
                qcell[q][cid] = {"cell_id": cid, "cell_score": cs, "lat": float(plat[bi]),
                                 "lon": float(plon[bi]), "level_m": int(levels[lvl_i])}
        del grids
        if str(dev).startswith("cuda"):
            torch.cuda.synchronize(); torch.cuda.empty_cache()
        t_gpu += time.perf_counter() - tg
    wall = time.perf_counter() - t0
    ex.shutdown()

    thr = (250.0, 500.0, 1000.0)
    per_query, d1, dk = {}, [], []
    for q in Q:
        p = Q[q]; has_gt = math.isfinite(p["lat"]) and math.isfinite(p["lon"])
        ranked = sorted(qcell[q].values(), key=lambda v: v["cell_score"], reverse=True)[:args.topk]
        for i, v in enumerate(ranked):
            v["rank"] = i + 1
            v["dist_m"] = _haversine_m(p["lat"], p["lon"], v["lat"], v["lon"]) if has_gt else None
        per_query[q] = {"gt": [p["lat"], p["lon"]] if has_gt else None, "topk": ranked,
                        "n_candidates": len(cand[q])}
        if has_gt and ranked:
            d1.append(ranked[0]["dist_m"]); dk.append(min(v["dist_m"] for v in ranked))
    summary = {"n_images": len(Q), "n_with_gt": len(d1)}
    if d1:
        summary["top1"] = {f"distR@{int(t)}m": float((np.asarray(d1) <= t).mean()) for t in thr}
        summary[f"top{args.topk}"] = {f"distR@{int(t)}m": float((np.asarray(dk) <= t).mean()) for t in thr}
    out = {"meta": {"dataset_dir": args.dataset_dir, "read_chunk_cells": args.rck, "topk": args.topk,
                    "level_agg": args.level_agg, "cell_agg": args.cell_agg, "device": str(dev)},
           "summary": summary, "per_query": per_query,
           "speed": {"wall_s": wall, "read_s": t_read, "gpu_s": t_gpu, "n_cells": len(union),
                     "n_chunks": len(chunks)}}
    Path(args.out).expanduser().write_text(json.dumps(out), encoding="utf-8")
    print(f"[search_seq] cells={len(union)} chunks={len(chunks)} rck={args.rck} | "
          f"wall={wall:.1f}s read={t_read:.1f}s gpu={t_gpu:.1f}s"
          + (f" | top{args.topk} distR@250m={summary[f'top{args.topk}']['distR@250m']:.3f}" if d1 else ""), flush=True)
    print(f"[ok] -> {args.out}", flush=True)
    r.close(); qs.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
