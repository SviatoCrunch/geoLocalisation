"""Real-data A/B: cv2 USAC_MAGSAC vs Torch MAGSAC++ on IDENTICAL cached correspondences.

Runs the production patch_rerank two-stage ranker (coarse shortlist -> per-cell pyramid rerank ->
Sigma-levels + mean-cell -> distR@250m) UNCHANGED except the geometric verifier, and compares the
two verifiers on the SAME mutual-NN pairs and the SAME aggregation, so the only difference is the
inlier counter. This is the honest A/B the brief asks for; it reuses patch_rerank verbatim (no edits
to it) and only ADDS the torch_magsacpp backend here.

Data lives on S3; cells are fetched one-at-a-time and EVICTED (CellStoreS3 LRU with a small
--cache-cap => download->score->delete, peak disk ~= cache_cap cells). Cell-major: each unique
shortlisted cell is fetched ONCE and scored against every query that needs it.

Kramatorsk recipe (memory): vitg14 32x32 store + Sigma-levels + mean-cell; the known-good CPU verifier
is cv2 USAC_MAGSAC. Example (run from geoLocalisation/experiments, see README):

    uv run --python 3.11 --with "torch==2.5.1" --with numpy --with h5py --with boto3 --with tqdm \
        --with opencv-python-headless python -m magsacpp_torch.evaluate \
        --queries kram=/home/ubuntu/work/out/gallery_h5/geo_iso_noverlap/queries/query_kramatorsc_d1024.h5 \
        --shortlist /home/ubuntu/work/out/shortlist_kram_k100.json \
        --index-uri s3://geo-reference/embeddings/kram/pyramid_dinov2vitg14_p448/kram/_index.json \
        --k-coarse 30 --topk 5 --level-agg sum --cell-agg mean --cache-cap 2 \
        --backends cpu_magsac torch_magsacpp --device cuda --out /home/ubuntu/work/ab_kram.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# make patch_rerank importable regardless of CWD (repo root = experiments/magsacpp_torch -> ../..)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .batch import estimate_homography_magsacpp_batch     # noqa: E402
from .config import MagsacppConfig                       # noqa: E402
from .estimator import estimate_homography_magsacpp       # noqa: E402


def _pct(d, p):
    return float(np.percentile(np.asarray(d, float), p)) if len(d) else float("nan")


def _recall(d, t):
    d = np.asarray(d, float)
    return float((d <= t).mean()) if d.size else float("nan")


def _torch_inliers(qm, rm, cfg, device, gen):
    """Inlier count from the Torch MAGSAC++ core. H maps rm(map)->qm(query), same as cv2 path."""
    rm_t = torch.as_tensor(np.asarray(rm, np.float64), dtype=cfg.dtype, device=device)
    qm_t = torch.as_tensor(np.asarray(qm, np.float64), dtype=cfg.dtype, device=device)
    res = estimate_homography_magsacpp(rm_t, qm_t, config=cfg, generator=gen)
    return res.inlier_count if res.success else 0


def _torch_counts(mc, n_q, cfg, device, gen, min_sample, batched):
    """Inlier counts for ALL crops of a (cell,query). ``batched`` => one cross-pair call over all
    crops (the GPU throughput path); else per-crop. Crops with <min_sample matches -> 0."""
    counts = [0] * len(mc)
    idxs = [j for j, (qm, rm, nm) in enumerate(mc) if n_q and nm >= min_sample]
    if not idxs:
        return counts
    if not batched:
        for j in idxs:
            qm, rm, _ = mc[j]
            counts[j] = _torch_inliers(qm, rm, cfg, device, gen)
        return counts
    Nmax = max(mc[j][1].shape[0] for j in idxs)
    B = len(idxs)
    p1 = torch.zeros(B, Nmax, 2, dtype=cfg.dtype, device=device)   # source = rm (map)
    p2 = torch.zeros(B, Nmax, 2, dtype=cfg.dtype, device=device)   # dest   = qm (query)
    valid = torch.zeros(B, Nmax, dtype=torch.bool, device=device)
    for b, j in enumerate(idxs):
        qm, rm, _ = mc[j]
        m = rm.shape[0]
        p1[b, :m] = torch.as_tensor(np.asarray(rm, np.float64), dtype=cfg.dtype, device=device)
        p2[b, :m] = torch.as_tensor(np.asarray(qm, np.float64), dtype=cfg.dtype, device=device)
        valid[b, :m] = True
    res = estimate_homography_magsacpp_batch(p1, p2, valid, config=cfg, generator=gen)
    for b, j in enumerate(idxs):
        counts[j] = res[b].inlier_count
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", nargs="+", required=True, help="city=query_d1024.h5")
    ap.add_argument("--shortlist", required=True, help="coarse top-N cells per query (store ids)")
    ap.add_argument("--index-uri", default=None, help="s3://…/<city>/_index.json (S3 store)")
    ap.add_argument("--local-store", default=None, dest="local_store",
                    help="local dir (aws s3 sync of the store prefix) — offline, no S3 at runtime")
    ap.add_argument("--cache-dir", default="/tmp/cellcache")
    ap.add_argument("--cache-cap", type=int, default=2, help="LRU cells kept on disk (download->evict)")
    ap.add_argument("--k-coarse", type=int, default=30)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--level-agg", choices=["sum", "max"], default="sum")
    ap.add_argument("--cell-agg", choices=["mean", "min", "max"], default="mean")
    ap.add_argument("--reproj-thresh", type=float, default=2.0, help="inlier threshold (query px); cv2 + torch")
    ap.add_argument("--backends", nargs="+", default=["cpu_magsac", "torch_magsacpp"],
                    choices=["cpu_magsac", "torch_magsacpp"])
    # torch MAGSAC++ knobs (sigma_max is the key quality knob — tie to the noise/threshold scale)
    ap.add_argument("--mpp-sigma-max", type=float, default=2.0, dest="mpp_sigma_max")
    ap.add_argument("--mpp-hyps", type=int, default=1000, dest="mpp_hyps")
    ap.add_argument("--mpp-irls", type=int, default=3, dest="mpp_irls")
    ap.add_argument("--mpp-dtype", choices=["float32", "float64"], default="float32", dest="mpp_dtype")
    ap.add_argument("--mpp-seed", type=int, default=0, dest="mpp_seed")
    ap.add_argument("--mpp-per-pair", action="store_true", dest="mpp_per_pair",
                    help="force the slow per-crop torch path (default: cross-pair batched)")
    ap.add_argument("--mpp-solver", choices=["svd", "closed_form"], default="svd", dest="mpp_solver",
                    help="minimal 4-pt solver: svd or closed_form (adjugate, no SVD -- GPU-fast)")
    ap.add_argument("--mpp-lut", action="store_true", dest="mpp_lut",
                    help="use the gamma LUT instead of torch.special (faster loss/weights on GPU)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only-city", default=None)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from tqdm import tqdm
    from patch_rerank.cell_store_s3 import CellStoreS3
    from patch_rerank.matcher import _MIN_MATCHES, matched_coords_batch, verify_inliers
    from patch_rerank.query_io import QueryGridStore
    from patch_rerank.search_pyramid_s3 import _aggregate, _haversine_m, _load_cell
    _MIN = _MIN_MATCHES["homography"]

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    if device != args.device:
        print(f"[evaluate] CUDA unavailable -> device={device}", flush=True)
    cfg = MagsacppConfig(sigma_max=args.mpp_sigma_max, inlier_threshold=args.reproj_thresh,
                         max_hypotheses=args.mpp_hyps, irls_iters=args.mpp_irls,
                         minimal_solver=args.mpp_solver, use_lut=args.mpp_lut,
                         dtype=torch.float64 if args.mpp_dtype == "float64" else torch.float32)
    gen = torch.Generator(device=device).manual_seed(args.mpp_seed)

    qstore = QueryGridStore(dict(a.split("=", 1) for a in args.queries))
    if args.local_store:
        from .local_store import LocalCellStore
        store = LocalCellStore(args.local_store)
    elif args.index_uri:
        store = CellStoreS3(args.index_uri, cache_dir=args.cache_dir, cache_cap=args.cache_cap)
    else:
        print("[evaluate] need --local-store <dir> or --index-uri s3://…"); return 2
    sj = json.loads(Path(args.shortlist).expanduser().read_text())

    # --- per-query plan: query features on device + its candidate cells ---
    plans = {}
    for q, entry in sj["shortlist"].items():
        if args.only_city and q.split(":", 1)[0] != args.only_city:
            continue
        if not qstore.has(q):
            continue
        if args.max_queries and len(plans) >= args.max_queries:
            break
        qfeat, qxy, qlat, qlon = qstore.get(q)
        cands = [c for c in entry["cells"][:args.k_coarse] if store.has(c)]
        plans[q] = {"qfeat": qfeat.to(device), "qxy": qxy, "qlat": qlat, "qlon": qlon,
                    "n_q": int(qfeat.shape[0]), "cands": cands}
    if not plans:
        print("[evaluate] no queries matched shortlist/store/city filters", flush=True)
        return 2

    backends = list(args.backends)
    qcell = {b: {q: {} for q in plans} for b in backends}   # backend -> q -> {cid: rec}
    t_verify = {b: 0.0 for b in backends}
    t_fetch = t_match = t_load = 0.0

    inv = collections.defaultdict(list)                     # cell -> queries needing it
    for q, p in plans.items():
        for cid in p["cands"]:
            inv[cid].append(q)

    for cid in tqdm(sorted(inv), desc="cells", unit="cell"):
        t0 = time.perf_counter(); cd = store.cell(cid); t_fetch += time.perf_counter() - t0
        tl = time.perf_counter(); grids, keys, kp = _load_cell(cd, device)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        t_load += time.perf_counter() - tl
        for q in inv[cid]:
            p = plans[q]
            tm = time.perf_counter()
            mc = matched_coords_batch(p["qfeat"], grids, p["qxy"], kp)   # identical pairs for both
            t_match += time.perf_counter() - tm
            nq = p["n_q"]
            sc = {}
            if "cpu_magsac" in backends:
                tv = time.perf_counter()
                sc["cpu_magsac"] = {}
                for j, (i, L) in enumerate(keys):
                    qm, rm, n_mut = mc[j]
                    n_in = (verify_inliers(qm, rm, model="homography", estimator="magsac",
                                           reproj_thresh=args.reproj_thresh).shape[0]
                            if (nq and n_mut >= _MIN) else 0)
                    sc["cpu_magsac"][(i, L)] = n_in / nq if nq else 0.0
                t_verify["cpu_magsac"] += time.perf_counter() - tv
            if "torch_magsacpp" in backends:
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                tv = time.perf_counter()
                cnts = _torch_counts(mc, nq, cfg, device, gen, _MIN, not args.mpp_per_pair)
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                t_verify["torch_magsacpp"] += time.perf_counter() - tv
                sc["torch_magsacpp"] = {(i, L): (cnts[j] / nq if nq else 0.0)
                                        for j, (i, L) in enumerate(keys)}
            for b in backends:
                pyr, blv = [], []
                for i in range(cd.n_pos):
                    per = {L: sc[b][(i, L)] for L in cd.levels}
                    bL = max(per, key=per.get)
                    pyr.append(sum(per.values()) if args.level_agg == "sum" else per[bL])
                    blv.append(int(bL))
                cs, bi, blat, blon, bL = _aggregate(pyr, blv, cd, args.cell_agg)
                qcell[b][q][cid] = {"cell_id": cid, "cell_score": cs, "lat": blat, "lon": blon,
                                    "level_m": bL}
        del grids
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    # --- aggregate per backend -> ranking -> distances ---
    out = {"meta": {"backends": backends, "k_coarse": args.k_coarse, "topk": args.topk,
                    "level_agg": args.level_agg, "cell_agg": args.cell_agg,
                    "reproj_thresh": args.reproj_thresh,
                    "store": args.local_store or args.index_uri,
                    "store_config": store.config, "device": str(device),
                    "mpp": {"sigma_max": args.mpp_sigma_max, "hyps": args.mpp_hyps,
                            "irls": args.mpp_irls, "dtype": args.mpp_dtype,
                            "solver": args.mpp_solver, "batched": not args.mpp_per_pair,
                            "lut": args.mpp_lut},
                    "fingerprint": hashlib.sha1(
                        (args.shortlist + "|" + str(args.local_store or args.index_uri) + "|"
                         + json.dumps(store.config, sort_keys=True)).encode()).hexdigest()[:12]},
           "per_query": {}, "summary": {}, "speed": {}}
    top1 = {b: [] for b in backends}
    topk = {b: [] for b in backends}
    for q, p in plans.items():
        has_gt = np.isfinite(p["qlat"]) and np.isfinite(p["qlon"])
        rec = {"gt": [p["qlat"], p["qlon"]] if has_gt else None, "n_candidates": len(p["cands"])}
        for b in backends:
            recs = list(qcell[b][q].values())
            ranked = sorted(recs, key=lambda r: r["cell_score"], reverse=True)[:args.topk]
            d1 = _haversine_m(p["qlat"], p["qlon"], ranked[0]["lat"], ranked[0]["lon"]) \
                if (ranked and has_gt) else None
            dk = min((_haversine_m(p["qlat"], p["qlon"], r["lat"], r["lon"]) for r in ranked),
                     default=None) if has_gt else None
            rec[b] = {"top1_dist_m": d1, "topk_dist_m": dk,
                      "top1": {k: ranked[0][k] for k in ("cell_id", "lat", "lon", "level_m", "cell_score")}
                      if ranked else None}
            if has_gt:
                top1[b].append(d1); topk[b].append(dk)
        out["per_query"][q] = rec

    thr = (250.0, 500.0, 1000.0)
    for b in backends:
        d1, dk = top1[b], topk[b]
        out["summary"][b] = {
            "n": len(d1),
            "top1": {"median_m": _pct(d1, 50), "p90_m": _pct(d1, 90), "p95_m": _pct(d1, 95),
                     **{f"distR@{int(t)}m": _recall(d1, t) for t in thr}},
            "topk": {"median_m": _pct(dk, 50), "p90_m": _pct(dk, 90), "p95_m": _pct(dk, 95),
                     **{f"distR@{int(t)}m": _recall(dk, t) for t in thr}},
        }
        print(f"[{b}] n={len(d1)} top1 distR@250m={_recall(d1,250):.3f} top{args.topk} "
              f"distR@250m={_recall(dk,250):.3f} median_top1={_pct(d1,50):.0f}m", flush=True)

    # paired analysis (brief §10): queries CPU localizes <=250 but torch loses, and vice-versa
    if {"cpu_magsac", "torch_magsacpp"} <= set(backends):
        c, t = np.array(top1["cpu_magsac"], float), np.array(top1["torch_magsacpp"], float)
        lost = int(((c <= 250) & (t > 250)).sum())
        gained = int(((c > 250) & (t <= 250)).sum())
        out["summary"]["paired_top1@250m"] = {"cpu_ok": int((c <= 250).sum()),
                                              "torch_ok": int((t <= 250).sum()),
                                              "torch_lost": lost, "torch_gained": gained,
                                              "n": int(len(c))}
        print(f"[paired top1@250m] cpu_ok={int((c<=250).sum())} torch_ok={int((t<=250).sum())} "
              f"torch_lost={lost} torch_gained={gained} (brief acceptance: lost==0)", flush=True)

    out["speed"] = {"queries": len(plans), "fetch_s": t_fetch, "load_s": t_load, "match_s": t_match,
                    "verify_s": t_verify, "cells_downloaded": store.n_downloads}
    print(f"[speed] q={len(plans)} fetch={t_fetch:.1f}s load={t_load:.1f}s match={t_match:.1f}s "
          + " ".join(f"verify[{b}]={t_verify[b]:.1f}s" for b in backends)
          + f" downloads={store.n_downloads}", flush=True)
    Path(args.out).expanduser().write_text(json.dumps(out), encoding="utf-8")
    print(f"[ok] -> {args.out}", flush=True)
    store.close(); qstore.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
