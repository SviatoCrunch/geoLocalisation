"""Minimum-reads GPU rerank over the packed new format: read a shard's NEEDED cells as coalesced
sequential runs, slice in RAM, score PER LEVEL on GPU. Shard-by-shard with a real 1-shard-ahead
prefetch (overlaps the sequential read with GPU work) and per-shard progress logging.

Read pattern (h5py/HDF-forum consensus): few large sequential reads, slice in memory; no chunking
(data is contiguous). Per shard we read only its needed cells via ``reader.read_cells`` which coalesces
consecutive local indices into one ``ds[lo:hi]`` span — so reads ≈ #contiguous runs (≤ #shards when
dense), not per-cell (143) or per-level (1144). GPU batches PER LEVEL (n_positions) across
``--gpu-cells`` cells so VRAM stays tiny regardless of read size. Math = faithful MAGSAC++
(magsacpp_torch primitives), identical scoring/aggregation.

RAM: ~ needed-cells-of-one-shard (×2 with prefetch). Use ``--no-prefetch`` if RAM-limited.
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
    ap.add_argument("--queries", nargs="+", required=True)
    ap.add_argument("--indices", required=True)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--gpu-cells", type=int, default=4, dest="gpu_cells",
                    help="cells per per-level GPU batch (crops = gpu_cells*n_positions)")
    ap.add_argument("--no-prefetch", dest="prefetch", action="store_false", help="disable 1-shard-ahead read (low RAM)")
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

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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
                         max_hypotheses=args.mpp_hyps, irls_iters=args.mpp_irls, minimal_solver=args.mpp_solver,
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
    Q, cand = {}, {}
    for qid in list(ind.get("shortlist", ind).keys()):
        if not qs.has(qid):
            continue
        if args.max_queries and len(Q) >= args.max_queries:
            break
        feat, xy, lat, lon = qs.get(qid)
        cells = [c for c in (resolve(x) for x in _candidates(ind, qid)) if c in r.cells][:args.k_coarse]
        if not cells:
            continue
        Q[qid] = {"feat": feat.to(dev), "xy": xy, "nq": int(feat.shape[0]), "lat": lat, "lon": lon}
        cand[qid] = set(cells)
    if not Q:
        print("[search_shard] no queries matched"); return 2

    union = set().union(*cand.values())
    byfile = {}                                               # file -> sorted [(local_idx, cell_id)]
    for cid in union:
        c = r.cells[cid]; byfile.setdefault(c["file"], []).append((int(c["local_cell_idx"]), cid))
    for f in byfile:
        byfile[f].sort()
    qneed = {cid: [q for q in Q if cid in cand[q]] for cid in union}
    qcell = {q: {} for q in Q}
    files = sorted(byfile)
    print(f"[search_shard] queries={len(Q)} cells={len(union)} shards={len(files)} "
          f"(cells/shard={[len(byfile[f]) for f in files]}) gpu_cells={args.gpu_cells} prefetch={args.prefetch}", flush=True)

    def load_shard(fname):                                    # read this shard's needed cells (coalesced runs)
        cids = [c for _, c in byfile[fname]]
        arr = np.stack(r.read_cells(cids))                    # (nc, P, L, H, W, D) fp16, few sequential reads
        co = [r.read_cell_coords(c) for c in cids]
        return cids, arr, co

    t_read = t_gpu = 0.0
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(load_shard, files[0]) if files else None
    t0 = time.perf_counter()
    for fi, fname in enumerate(files):
        tr = time.perf_counter()
        cids, arr, co = fut.result() if args.prefetch else load_shard(fname)
        sr = time.perf_counter() - tr; t_read += sr
        if args.prefetch:
            fut = ex.submit(load_shard, files[fi + 1]) if fi + 1 < len(files) else None
        tg = time.perf_counter()
        n = len(cids)
        for cs in range(0, n, args.gpu_cells):
            ce = min(cs + args.gpu_cells, n); nc = ce - cs
            sub_cids = cids[cs:ce]
            qset = {q for c in sub_cids for q in qneed[c]}
            accq = {q: np.zeros((nc, P, L), np.float64) for q in qset}
            for li_ in range(L):
                crops = np.ascontiguousarray(arr[cs:ce, :, li_]).reshape(nc * P, H * W, D)
                grids = torch.from_numpy(crops).to(dev)
                for q in qset:
                    p = Q[q]
                    qm, rm, valid = _mutual_nn_padded(p["feat"], grids, p["xy"], kp, dev, args.match_budget)
                    counts, _ = estimate_homography_magsacpp_batch(rm, qm, valid, config=cfg, generator=gen,
                                                                   return_counts=True)
                    accq[q][:, :, li_] = (counts.double() / p["nq"] if p["nq"] else counts.double()).view(nc, P).cpu().numpy()
                del grids
                if str(dev).startswith("cuda"):
                    torch.cuda.empty_cache()
            for k in range(nc):
                cid = sub_cids[k]; plat, plon = co[cs + k]
                for q in qneed[cid]:
                    s = accq[q][k]
                    pyr = s.sum(1) if args.level_agg == "sum" else s.max(1)
                    bi = int(pyr.argmax())
                    cs_ = {"mean": float(pyr.mean()), "min": float(pyr.min()), "max": float(pyr.max())}[args.cell_agg]
                    lvl_i = int(s[bi].argmax())
                    qcell[q][cid] = {"cell_id": cid, "cell_score": cs_, "lat": float(plat[bi]),
                                     "lon": float(plon[bi]), "level_m": int(levels[lvl_i])}
        del arr
        if str(dev).startswith("cuda"):
            torch.cuda.synchronize()
        sg = time.perf_counter() - tg; t_gpu += sg
        print(f"  shard {fi + 1}/{len(files)} {fname}: cells={n} read={sr:.1f}s gpu={sg:.1f}s", flush=True)
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
        per_query[q] = {"gt": [p["lat"], p["lon"]] if has_gt else None, "topk": ranked, "n_candidates": len(cand[q])}
        if has_gt and ranked:
            d1.append(ranked[0]["dist_m"]); dk.append(min(v["dist_m"] for v in ranked))
    summary = {"n_images": len(Q), "n_with_gt": len(d1)}
    if d1:
        summary["top1"] = {f"distR@{int(t)}m": float((np.asarray(d1) <= t).mean()) for t in thr}
        summary[f"top{args.topk}"] = {f"distR@{int(t)}m": float((np.asarray(dk) <= t).mean()) for t in thr}
    out = {"meta": {"dataset_dir": args.dataset_dir, "gpu_cells": args.gpu_cells, "prefetch": args.prefetch,
                    "topk": args.topk, "level_agg": args.level_agg, "cell_agg": args.cell_agg, "device": str(dev)},
           "summary": summary, "per_query": per_query,
           "speed": {"wall_s": wall, "read_s": t_read, "gpu_s": t_gpu, "n_shards": len(files), "n_cells": len(union)}}
    Path(args.out).expanduser().write_text(json.dumps(out), encoding="utf-8")
    print(f"[search_shard] DONE cells={len(union)} shards={len(files)} | wall={wall:.1f}s read={t_read:.1f}s gpu={t_gpu:.1f}s"
          + (f" | top{args.topk} distR@250m={summary[f'top{args.topk}']['distR@250m']:.3f}" if d1 else ""), flush=True)
    print(f"[ok] -> {args.out}", flush=True)
    r.close(); qs.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
