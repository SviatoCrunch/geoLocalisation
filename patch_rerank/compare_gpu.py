"""A/B a GPU homography verifier against the frozen cv2.MAGSAC++ diagnostics archive.

The archive (``search_pyramid_s3 --diag-dir``) stored, for every query×cell×position×level, the exact
``qm``/``rm`` fed to cv2 and cv2's inlier count + level score. This tool replays those SAME
correspondences through a GPU verifier (``gpu_verify``) and compares — WITHOUT re-running DINO /
matching / S3 — on two levels:

  * per-call geometry: Δ(inliers) = gpu − cv2, agreement, correlation, bucketed by pyramid level and by
    #mutual (where do they diverge?);
  * end-to-end ranking: recompute gpu level→position(Σ)→cell(mean) scores, re-rank the 100 candidates,
    and compare top-1/top-5 geo error vs cv2 (from summary.jsonl) and vs GT — which frames the GPU
    backend LOSES or GAINS at distR@250m top-5.

MAGSAC is randomised and the GPU MSAC objective differs, so this is a STATISTICAL comparison, not a
bit-exact one (see gpu_verify). The cv2 archive is the reference; this says where the GPU path fails.

Run::  python -m patch_rerank.compare_gpu --diag-dir /…/diag_kram_test --backend gpu_batch --device cuda
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .diag_stats import _haversine_m

_THR = (250.0, 500.0, 1000.0)


def _arrays_path(uri: str, diag_dir: Path, tmp: str) -> str:
    if uri.startswith("s3://"):
        import boto3
        b, key = uri[5:].split("/", 1)
        loc = os.path.join(tmp, os.path.basename(key))
        if not os.path.exists(loc):
            boto3.client("s3").download_file(b, key, loc)
        return loc
    return str(diag_dir / uri)


def _aggregate_gpu(cells: dict, level_agg: str, cell_agg: str):
    """cells[cid] = {pos_id: {"levels": {L: score}, "lat":, "lon":}} → ranked list of
    {cell_id, cell_score, lat, lon} using the SERVICE rule (position=Σlevels, cell=mean positions,
    point=best position)."""
    ranked = []
    for cid, poss in cells.items():
        pscore, plat, plon = [], [], []
        for pid, pv in sorted(poss.items()):
            lv = list(pv["levels"].values())
            pscore.append(sum(lv) if level_agg == "sum" else (max(lv) if lv else 0.0))
            plat.append(pv["lat"]); plon.append(pv["lon"])
        arr = np.asarray(pscore, float)
        cs = {"mean": arr.mean(), "min": arr.min(), "max": arr.max()}[cell_agg]
        bi = int(arr.argmax())
        ranked.append({"cell_id": cid, "cell_score": float(cs), "lat": plat[bi], "lon": plon[bi]})
    ranked.sort(key=lambda r: r["cell_score"], reverse=True)
    return ranked


def compute(diag_dir, gpu_counts_fn, *, level_agg="sum", cell_agg="mean", only_done=True,
            max_queries=0, verbose=False) -> dict:
    """Core A/B. ``gpu_counts_fn(pairs)`` takes a list of ``(qm, rm)`` arrays and returns a list of gpu
    inlier COUNTS (injectable → testable without a GPU). ``max_queries`` limits the frames processed (0
    = all); ``verbose`` prints a per-frame progress line (the GPU pass over all pairs is slow)."""
    import time

    import h5py
    d = Path(diag_dir).expanduser()
    done = set(json.loads((d / "completed.json").read_text())["queries"]) if (d / "completed.json").exists() else None
    summ = {}
    for line in (d / "summary.jsonl").read_text(encoding="utf-8").splitlines():
        s = json.loads(line); summ[s["query_id"]] = s

    # per-query structure from records.jsonl (ALL cell/pos/level, incl. skips)
    recs = collections.defaultdict(list)
    nq = {}
    for line in (d / "records.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        recs[r["query_id"]].append(r)
        nq[r["query_id"]] = int(r["n_query_patches"])

    per_call = []                                           # (level_m, n_mutual, cv2_inl, gpu_inl)
    cv2_t1, cv2_t5, gpu_t1, gpu_t5 = [], [], [], []
    lost, gained = [], []
    tmp = tempfile.mkdtemp()
    todo = [q for q in recs if not (only_done and done is not None and q not in done) and q in summ]
    if max_queries:
        todo = todo[:max_queries]
    for qi, q in enumerate(todo):
        rows = recs[q]
        s = summ[q]
        gt = s.get("gt")
        has_gt = bool(gt) and None not in gt
        n_q = nq[q] or 1
        # collect pairs for the cv2-called groups of this query
        uri = s.get("arrays") or f"arrays/{q.replace(':', '_').replace('/', '_')}.h5"
        t0 = time.perf_counter()
        pairs, keys = [], []
        with h5py.File(_arrays_path(uri, d, tmp), "r") as f:
            for r in rows:
                if r["cv2_called"] and r["h5_group"] in f:
                    g = f[r["h5_group"]]
                    pairs.append((g["qm"][:], g["rm"][:])); keys.append(r)
        if verbose:
            print(f"[compare] {qi + 1}/{len(todo)} {q}: {len(pairs)} pairs "
                  f"(read {time.perf_counter() - t0:.1f}s) → gpu…", flush=True)
        gpu_counts = gpu_counts_fn(pairs) if pairs else []
        if verbose:
            print(f"[compare]   gpu done in {time.perf_counter() - t0:.1f}s", flush=True)

        # build gpu cell structure (start from ALL rows at score 0, fill cv2-called with gpu score)
        cells: dict = collections.defaultdict(dict)
        for r in rows:
            cells[r["cell_id"]].setdefault(
                r["position_id"], {"levels": {}, "lat": r.get("pos_lat"), "lon": r.get("pos_lon")})
            cells[r["cell_id"]][r["position_id"]]["levels"][int(r["level_id"])] = 0.0
        for r, gc in zip(keys, gpu_counts):
            cells[r["cell_id"]][r["position_id"]]["levels"][int(r["level_id"])] = gc / n_q
            per_call.append((r["level_m"], r["n_mutual"], r["n_inliers"], gc))

        ranked = _aggregate_gpu(cells, level_agg, cell_agg)
        if has_gt:
            for r in ranked[:5]:
                r["dist_m"] = _haversine_m(gt[0], gt[1], r["lat"], r["lon"])
            g1 = ranked[0]["dist_m"]; g5 = min(r["dist_m"] for r in ranked[:5])
            c1 = (s.get("metrics") or {}).get("fine_dist_m")
            c5 = (s.get("metrics") or {}).get("fine_dist_topk_m")
            gpu_t1.append(g1); gpu_t5.append(g5); cv2_t1.append(c1); cv2_t5.append(c5)
            if c5 is not None:
                if c5 <= 250 and g5 > 250:
                    lost.append(q)
                elif c5 > 250 and g5 <= 250:
                    gained.append(q)

    def _distR(xs, t):
        v = [x for x in xs if x is not None]
        return (sum(x <= t for x in v) / len(v)) if v else None

    dlt = np.array([g - c for (_, _, c, g) in per_call], float) if per_call else np.array([])
    cv2c = np.array([c for (_, _, c, _) in per_call], float)
    gpuc = np.array([g for (_, _, _, g) in per_call], float)
    by_level = {}
    for lm, _, c, g in per_call:
        by_level.setdefault(int(lm), []).append(g - c)
    corr = float(np.corrcoef(cv2c, gpuc)[0, 1]) if len(cv2c) > 1 and cv2c.std() and gpuc.std() else None

    return {
        "n_queries": len(gpu_t1), "n_calls": len(per_call),
        "per_call": {
            "mean_delta_inliers": float(dlt.mean()) if dlt.size else None,
            "median_delta_inliers": float(np.median(dlt)) if dlt.size else None,
            "mean_abs_delta": float(np.abs(dlt).mean()) if dlt.size else None,
            "exact_match_frac": float((dlt == 0).mean()) if dlt.size else None,
            "corr_inlier_counts": corr,
            "mean_delta_by_level_m": {k: float(np.mean(v)) for k, v in sorted(by_level.items())}},
        "ranking": {
            "cv2": {f"top1_distR@{int(t)}m": _distR(cv2_t1, t) for t in _THR}
            | {f"top5_distR@{int(t)}m": _distR(cv2_t5, t) for t in _THR}
            | {"top1_median_m": float(np.median([x for x in cv2_t1 if x is not None])) if cv2_t1 else None},
            "gpu": {f"top1_distR@{int(t)}m": _distR(gpu_t1, t) for t in _THR}
            | {f"top5_distR@{int(t)}m": _distR(gpu_t5, t) for t in _THR}
            | {"top1_median_m": float(np.median(gpu_t1)) if gpu_t1 else None},
            "paired_top5@250m": {"gpu_lost": len(lost), "gpu_gained": len(gained),
                                 "lost_queries": lost, "gained_queries": gained}},
    }


def _magsacpp_fn(device: str, n_hyp: int, sigma_max: float, reproj_thresh: float):
    """Backend = the faithful Torch MAGSAC++ port (experiments/magsacpp_torch), the 'latest GPU
    variant'. Config mirrors its optimized GPU path (closed_form, float32, sigma_max/inlier_threshold
    set to match the cv2 run). Feeds points1=rm, points2=qm (same rm→qm direction as cv2) and returns
    inlier counts at inlier_threshold=reproj_thresh — directly comparable to cv2's n_inliers."""
    import sys
    exp = str(Path(__file__).resolve().parents[1] / "experiments")
    if exp not in sys.path:
        sys.path.insert(0, exp)
    import torch
    from magsacpp_torch import MagsacppConfig
    from magsacpp_torch.batch import estimate_homography_magsacpp_batch
    cfg = MagsacppConfig(dtype=torch.float32, minimal_solver="closed_form", max_hypotheses=int(n_hyp),
                         sigma_max=float(sigma_max), inlier_threshold=float(reproj_thresh))

    def fn(pairs):
        if not pairs:
            return []
        P = len(pairs)
        nmax = max(p[1].shape[0] for p in pairs)                 # rm point count
        p1 = torch.zeros(P, nmax, 2, dtype=torch.float32, device=device)   # rm (source)
        p2 = torch.zeros(P, nmax, 2, dtype=torch.float32, device=device)   # qm (dest)
        valid = torch.zeros(P, nmax, dtype=torch.bool, device=device)
        for i, (qm, rm) in enumerate(pairs):
            m = rm.shape[0]
            p1[i, :m] = torch.as_tensor(np.asarray(rm, np.float32), device=device)
            p2[i, :m] = torch.as_tensor(np.asarray(qm, np.float32), device=device)
            valid[i, :m] = True
        counts, _ = estimate_homography_magsacpp_batch(p1, p2, valid, config=cfg, return_counts=True)
        return [int(x) for x in counts.tolist()]
    return fn


def _gpu_counts_fn(backend: str, device: str, n_hyp: int, seed: int, sigma_max: float = 2.0,
                   reproj_thresh: float = 2.0):
    if backend == "magsacpp_torch":
        return _magsacpp_fn(device, n_hyp, sigma_max, reproj_thresh)

    def fn(pairs):
        if backend == "gpu_batch":
            from .gpu_verify import ransac_homography_batch
            inls = ransac_homography_batch([(q, r) for (q, r) in pairs], n_hyp=n_hyp, device=device,
                                           seed=seed)
            return [int(x.shape[0]) for x in inls]
        from .gpu_verify import verify_inliers_kornia
        return [int(verify_inliers_kornia(q, r, device=device, seed=seed).shape[0]) for (q, r) in pairs]
    return fn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag-dir", required=True)
    ap.add_argument("--backend", choices=["gpu_batch", "gpu_kornia", "magsacpp_torch"],
                    default="gpu_batch", help="magsacpp_torch = the faithful Torch MAGSAC++ port")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-hyp", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-max", type=float, default=2.0, help="magsacpp_torch sigma_max")
    ap.add_argument("--reproj-thresh", type=float, default=2.0, help="inlier threshold (match cv2 run)")
    ap.add_argument("--level-agg", choices=["sum", "max"], default="sum")
    ap.add_argument("--cell-agg", choices=["mean", "min", "max"], default="mean")
    ap.add_argument("--max-queries", type=int, default=0, help="limit frames (0=all) for a quick A/B")
    ap.add_argument("--out", default=None, help="default: <diag-dir>/compare_<backend>.json")
    args = ap.parse_args(argv)
    res = compute(args.diag_dir, _gpu_counts_fn(args.backend, args.device, args.n_hyp, args.seed,
                                                args.sigma_max, args.reproj_thresh),
                  level_agg=args.level_agg, cell_agg=args.cell_agg, max_queries=args.max_queries,
                  verbose=True)
    out = Path(args.out).expanduser() if args.out else Path(args.diag_dir).expanduser() / f"compare_{args.backend}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    pc, rk = res["per_call"], res["ranking"]
    print(f"[per-call] n={res['n_calls']} meanΔinl={pc['mean_delta_inliers']} "
          f"|Δ|={pc['mean_abs_delta']} corr={pc['corr_inlier_counts']} "
          f"exact={pc['exact_match_frac']}", flush=True)
    print(f"[per-call] meanΔ by level: {pc['mean_delta_by_level_m']}", flush=True)
    print(f"[ranking] cv2 top5@250m={rk['cv2']['top5_distR@250m']}  gpu top5@250m={rk['gpu']['top5_distR@250m']}  "
          f"gpu_lost={rk['paired_top5@250m']['gpu_lost']} gpu_gained={rk['paired_top5@250m']['gpu_gained']}",
          flush=True)
    if rk["paired_top5@250m"]["lost_queries"]:
        print(f"[ranking] LOST frames (cv2 hit, gpu miss @250m top5): {rk['paired_top5@250m']['lost_queries']}",
              flush=True)
    print(f"[ok] -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
