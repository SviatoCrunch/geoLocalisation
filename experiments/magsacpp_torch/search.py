"""Batched multi-image geo-search on the S3 pyramid store with the fast torch MAGSAC++ reranker.

Input = a BATCH of query embeddings (DINO token grids) that arrive TOGETHER. For each image it runs
coarse-shortlist -> fast rerank and returns the top-K candidate cells + a predicted lat/lon. If the
queries carry GT lat/lon it also reports distR@250/500/1000 + median/p90/p95. torch-only (our
closed_form minimal solver + eigh IRLS refit, batched); S3 cells are fetched one-at-a-time and
evicted (cell-major: each cell is loaded ONCE and scored against every image that shortlisted it).

Core API is batch-native (:func:`search` takes the embeddings directly, not a file), so a live feed
of "embeddings that arrive together" can call it without touching disk; the CLI just loads a query
H5 + shortlist for convenience.

This is the SEARCH sibling of ``evaluate`` (which is the cv2-vs-torch A/B harness): no cv2, no GT
required. Throughput levers to come (batch mutual-NN across the images sharing a cell, coarse stage).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch.nn.functional as F                           # noqa: E402

from .batch import estimate_homography_magsacpp_batch      # noqa: E402
from .config import MagsacppConfig                        # noqa: E402
from .evaluate import _torch_counts                       # batched torch inlier counts (reused)  # noqa: E402


def _pct(d, p):
    return float(np.percentile(np.asarray(d, float), p)) if len(d) else float("nan")


def _mutual_nn_padded(q_feat, r_feats, q_xy, r_xy, device, match_budget: int = 50_000_000):
    """Vectorized mutual-NN for ALL crops at once → padded matched coords, fully on GPU.

    ``q_feat`` (Nq,D), ``r_feats`` (C,Nr,D). Returns (qm, rm, valid) with qm/rm (C,M_max,2) and
    valid (C,M_max) bool — the matched query/ref coordinates per crop, padded to the per-cell max.
    The cosine-sims tensor (chunk,Nq,Nr) is the memory hog, so crops are processed in chunks of
    ``match_budget // (Nq*Nr)`` ("memory at a time"); only the small mutual-match triples are kept.
    Same match SET as patch_rerank.matcher.matched_coords_batch, but no per-crop Python/numpy and no
    host round-trip."""
    q = F.normalize(q_feat.float(), dim=1)
    R = F.normalize(r_feats.float(), dim=2)
    C, Nr, _ = R.shape
    Nq = q.shape[0]
    qar = torch.arange(Nq, device=device)
    cs, qis, ris = [], [], []
    step = max(1, match_budget // max(1, Nq * Nr))
    for lo in range(0, C, step):
        hi = min(C, lo + step)
        sims = torch.matmul(q, R[lo:hi].transpose(1, 2))      # (c, Nq, Nr)
        q2r = sims.argmax(2)                                  # (c, Nq) best ref per query token
        r2q = sims.argmax(1)                                  # (c, Nr) best query per ref token
        mutual = torch.gather(r2q, 1, q2r) == qar             # (c, Nq)
        nz = mutual.nonzero(as_tuple=False)                   # (K,2): [c_local, i] ascending
        if nz.numel():
            cl, ii = nz[:, 0], nz[:, 1]
            cs.append(cl + lo); qis.append(ii); ris.append(q2r[cl, ii])
        del sims
    if not cs:
        z = torch.zeros(C, 0, 2, dtype=torch.float64, device=device)
        return z, z, torch.zeros(C, 0, dtype=torch.bool, device=device)
    all_c = torch.cat(cs); all_qi = torch.cat(qis); all_ri = torch.cat(ris)
    counts = torch.bincount(all_c, minlength=C)               # matches per crop
    M_max = int(counts.max())
    offs = torch.cumsum(counts, 0) - counts                   # crop start offset (all_c is non-decreasing)
    slot = torch.arange(all_c.numel(), device=device) - offs[all_c]   # within-crop rank
    qxy_t = torch.as_tensor(np.asarray(q_xy, np.float64), device=device)
    rxy_t = torch.as_tensor(np.asarray(r_xy, np.float64), device=device)
    qm = torch.zeros(C, M_max, 2, dtype=torch.float64, device=device)
    rm = torch.zeros(C, M_max, 2, dtype=torch.float64, device=device)
    valid = torch.zeros(C, M_max, dtype=torch.bool, device=device)
    qm[all_c, slot] = qxy_t[all_qi]
    rm[all_c, slot] = rxy_t[all_ri]
    valid[all_c, slot] = True
    return qm, rm, valid


def search(queries: dict, shortlist: dict, store, *, config: MagsacppConfig, k_coarse: int = 30,
           topk: int = 5, level_agg: str = "sum", cell_agg: str = "mean", device: str = "cuda",
           generator=None, flat: bool = True, match_budget: int = 50_000_000):
    """Localize a batch of query embeddings.

    ``queries``: ``{query_id: {"feat": (N,D) tensor, "xy": (N,2), "lat": float|nan, "lon": float|nan}}``
    (feat already on ``device``). ``shortlist``: ``{"shortlist": {query_id: {"cells": [...]}}}``.
    ``store``: a ``patch_rerank.cell_store_s3.CellStoreS3``. Returns ``(per_query, summary, speed)``.
    """
    from patch_rerank.matcher import _MIN_MATCHES, matched_coords_batch
    from patch_rerank.search_pyramid_s3 import _aggregate, _haversine_m, _load_cell
    from tqdm import tqdm
    _MIN = _MIN_MATCHES["homography"]
    gen = generator

    # plan: query -> candidate cells present in the store
    plans = {}
    for q, qd in queries.items():
        entry = shortlist.get("shortlist", {}).get(q)
        if entry is None:
            continue
        cands = [c for c in entry["cells"][:k_coarse] if store.has(c)]
        plans[q] = {**qd, "n_q": int(qd["feat"].shape[0]), "cands": cands}
    qcell = {q: {} for q in plans}
    inv = collections.defaultdict(list)
    for q, p in plans.items():
        for cid in p["cands"]:
            inv[cid].append(q)

    t_fetch = t_match = t_verify = 0.0
    for cid in tqdm(sorted(inv), desc="cells", unit="cell"):
        t0 = time.perf_counter(); cd = store.cell(cid); t_fetch += time.perf_counter() - t0
        grids, keys, kp = _load_cell(cd, device)
        nlev = len(cd.levels)
        for q in inv[cid]:
            p = plans[q]
            if flat:
                # one flat GPU pass over all C = n_pos*n_levels crops: match -> MAGSAC++ counts ->
                # score -> reshape (n_pos, n_levels) aggregation. No per-crop Python / host round-trip.
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                tm = time.perf_counter()
                qm, rm, valid = _mutual_nn_padded(p["feat"], grids, p["xy"], kp, device, match_budget)
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                t_match += time.perf_counter() - tm
                tv = time.perf_counter()
                counts, _ = estimate_homography_magsacpp_batch(rm, qm, valid, config=config,
                                                               generator=gen, return_counts=True)
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                t_verify += time.perf_counter() - tv
                score = counts.double() / p["n_q"] if p["n_q"] else counts.double()
                s2d = score.view(cd.n_pos, nlev)                       # crops ordered pos-major
                pyr = s2d.sum(1) if level_agg == "sum" else s2d.amax(1)   # (n_pos,)
                bi = int(pyr.argmax())
                lvl_i = int(s2d[bi].argmax())
                cs = {"mean": float(pyr.mean()), "min": float(pyr.min()),
                      "max": float(pyr.max())}[cell_agg]
                qcell[q][cid] = {"cell_id": cid, "cell_score": cs, "lat": float(cd.lat[bi]),
                                 "lon": float(cd.lon[bi]), "level_m": int(cd.levels[lvl_i])}
                continue
            tm = time.perf_counter()
            mc = matched_coords_batch(p["feat"], grids, p["xy"], kp)
            t_match += time.perf_counter() - tm
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            tv = time.perf_counter()
            cnts = _torch_counts(mc, p["n_q"], config, device, gen, _MIN, batched=True)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            t_verify += time.perf_counter() - tv
            sc = {(i, L): (cnts[j] / p["n_q"] if p["n_q"] else 0.0) for j, (i, L) in enumerate(keys)}
            pyr, blv = [], []
            for i in range(cd.n_pos):
                per = {L: sc[(i, L)] for L in cd.levels}
                bL = max(per, key=per.get)
                pyr.append(sum(per.values()) if level_agg == "sum" else per[bL])
                blv.append(int(bL))
            cs, bi, blat, blon, bL = _aggregate(pyr, blv, cd, cell_agg)
            qcell[q][cid] = {"cell_id": cid, "cell_score": cs, "lat": blat, "lon": blon, "level_m": bL}
        del grids
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    per_query, d1, dk = {}, [], []
    for q, p in plans.items():
        has_gt = np.isfinite(p.get("lat", np.nan)) and np.isfinite(p.get("lon", np.nan))
        ranked = sorted(qcell[q].values(), key=lambda r: r["cell_score"], reverse=True)[:topk]
        for i, r in enumerate(ranked):
            r["rank"] = i + 1
            r["dist_m"] = (_haversine_m(p["lat"], p["lon"], r["lat"], r["lon"]) if has_gt else None)
        pred = ranked[0] if ranked else None
        rec = {"pred": ({"lat": pred["lat"], "lon": pred["lon"], "cell_id": pred["cell_id"],
                         "level_m": pred["level_m"]} if pred else None),
               "topk": ranked, "n_candidates": len(p["cands"]),
               "gt": ([p["lat"], p["lon"]] if has_gt else None)}
        per_query[q] = rec
        if has_gt and ranked:
            d1.append(ranked[0]["dist_m"]); dk.append(min(r["dist_m"] for r in ranked))

    thr = (250.0, 500.0, 1000.0)
    summary = {"n_images": len(plans), "n_with_gt": len(d1)}
    if d1:
        summary["top1"] = {"median_m": _pct(d1, 50), "p90_m": _pct(d1, 90), "p95_m": _pct(d1, 95),
                           **{f"distR@{int(t)}m": float((np.asarray(d1) <= t).mean()) for t in thr}}
        summary["topk"] = {"median_m": _pct(dk, 50), "p90_m": _pct(dk, 90), "p95_m": _pct(dk, 95),
                           **{f"distR@{int(t)}m": float((np.asarray(dk) <= t).mean()) for t in thr}}
    speed = {"fetch_s": t_fetch, "match_s": t_match, "verify_s": t_verify,
             "cells_downloaded": store.n_downloads}
    return per_query, summary, speed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", nargs="+", required=True, help="city=query_d1024.h5 (token grids)")
    ap.add_argument("--shortlist", required=True)
    ap.add_argument("--index-uri", required=True, help="s3://…/<city>/_index.json")
    ap.add_argument("--cache-dir", default="/tmp/cellcache")
    ap.add_argument("--cache-cap", type=int, default=2)
    ap.add_argument("--k-coarse", type=int, default=30)
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
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--legacy-match", action="store_true", dest="legacy_match",
                    help="use the per-crop match+count path instead of the flat GPU pass")
    ap.add_argument("--match-budget", type=int, default=50_000_000, dest="match_budget",
                    help="max elements of the (chunk,Nq,Nr) cosine-sims tensor per match chunk")
    ap.add_argument("--only-city", default=None)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from patch_rerank.cell_store_s3 import CellStoreS3
    from patch_rerank.query_io import QueryGridStore

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    cfg = MagsacppConfig(sigma_max=args.mpp_sigma_max, inlier_threshold=args.reproj_thresh,
                         max_hypotheses=args.mpp_hyps, irls_iters=args.mpp_irls,
                         minimal_solver=args.mpp_solver,
                         dtype=torch.float64 if args.mpp_dtype == "float64" else torch.float32)
    gen = torch.Generator(device=device).manual_seed(args.mpp_seed)

    qstore = QueryGridStore(dict(a.split("=", 1) for a in args.queries))
    store = CellStoreS3(args.index_uri, cache_dir=args.cache_dir, cache_cap=args.cache_cap)
    sj = json.loads(Path(args.shortlist).expanduser().read_text())

    # build the batch of embeddings that "arrive together"
    queries = {}
    for q in sj.get("shortlist", {}):
        if args.only_city and q.split(":", 1)[0] != args.only_city:
            continue
        if not qstore.has(q):
            continue
        if args.max_queries and len(queries) >= args.max_queries:
            break
        feat, xy, lat, lon = qstore.get(q)
        queries[q] = {"feat": feat.to(device), "xy": xy, "lat": lat, "lon": lon}
    if not queries:
        print("[search] no queries matched shortlist/store/city filters"); return 2

    per_query, summary, speed = search(queries, sj, store, config=cfg, k_coarse=args.k_coarse,
                                       topk=args.topk, level_agg=args.level_agg,
                                       cell_agg=args.cell_agg, device=device, generator=gen,
                                       flat=not args.legacy_match, match_budget=args.match_budget)
    out = {"meta": {"k_coarse": args.k_coarse, "topk": args.topk, "level_agg": args.level_agg,
                    "cell_agg": args.cell_agg, "index_uri": args.index_uri, "device": str(device),
                    "mpp": {"sigma_max": args.mpp_sigma_max, "hyps": args.mpp_hyps,
                            "irls": args.mpp_irls, "dtype": args.mpp_dtype, "solver": args.mpp_solver}},
           "summary": summary, "per_query": per_query, "speed": speed}
    if summary.get("top1"):
        print(f"[search] images={summary['n_images']} with_gt={summary['n_with_gt']} "
              f"top1 distR@250m={summary['top1']['distR@250m']:.3f} "
              f"top{args.topk} distR@250m={summary['topk']['distR@250m']:.3f} "
              f"median_top1={summary['top1']['median_m']:.0f}m", flush=True)
    else:
        print(f"[search] images={summary['n_images']} (no GT -> search-only, no metrics)", flush=True)
    print(f"[speed] fetch={speed['fetch_s']:.1f}s match={speed['match_s']:.1f}s "
          f"verify={speed['verify_s']:.1f}s downloads={speed['cells_downloaded']}", flush=True)
    Path(args.out).expanduser().write_text(json.dumps(out), encoding="utf-8")
    print(f"[ok] -> {args.out}", flush=True)
    store.close(); qstore.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
