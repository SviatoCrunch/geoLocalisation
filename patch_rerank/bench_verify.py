"""Benchmark the geometric-verification step in isolation, on FIXED mutual-NN pairs (no DINO).

Feed it the ``--dump-mnn`` .npz written by ``map_rerank`` (the exact (qm, rm) matched coords of every
scored crop of a real query). It times each verifier backend — CPU cv2 MAGSAC, per-pair kornia GPU
RANSAC, and the batched multi-pair GPU MSAC — with warm-up + repeats + CUDA synchronisation, and
separates host→device transfer from the fit itself. It also reports the inlier-count / score
distribution per backend and the per-pair rank agreement of GPU vs CPU (whether the reranker would see
different numbers), WITHOUT touching the score formula or aggregation.

Run (on the T4)::

    uv run --python 3.11 --with "torch==2.5.1" --with kornia --with opencv-python-headless \
           --with numpy python -m patch_rerank.bench_verify \
      --pairs /…/pairs_kup.npz --device cuda --backends cpu_magsac gpu_kornia gpu_batch \
      --warmup 2 --repeats 5 --jobs 8 --gpu-n-hyp 256 --gpu-score-type msac \
      --out /…/bench_kup.json

``--jobs`` threads the cv2 backend (cv2 releases the GIL). Speed is judged by ``fit_s`` (fit only) and
``pairs_per_s``; ``full_s`` adds the one-off host→device transfer. Quality is judged by the inlier /
score distributions and the CPU↔GPU agreement — plus an end-to-end ``map_rerank`` top-K diff (below).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def _load_pairs(path):
    """→ (pairs list of (qm (M,2) f32, rm (M,2) f32), nq (P,) int, reproj_thresh float)."""
    z = np.load(Path(path).expanduser())
    qm, rm, lengths, nq = z["qm"], z["rm"], z["lengths"], z["nq"]
    thr = float(z["reproj_thresh"])
    pairs, off = [], 0
    for m in lengths:
        m = int(m)
        pairs.append((qm[off:off + m], rm[off:off + m]))
        off += m
    return pairs, nq.astype(np.int64), thr


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _peak_mem_mb(device):
    if str(device).startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 1e6
    return float("nan")


def _reset_mem(device):
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def _spearman(a, b):
    """Spearman rank correlation of two 1-D arrays (no scipy dep)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


# --- backends: each returns a list of inlier COUNTS aligned to `pairs` (score = count / nq) ---

def _run_cpu(pairs, thr, jobs):
    from .matcher import verify_inliers
    def _n(p):
        return verify_inliers(p[0], p[1], model="homography", estimator="magsac",
                              reproj_thresh=thr).shape[0]
    if jobs and jobs > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            return list(ex.map(_n, pairs))
    return [_n(p) for p in pairs]


def _run_kornia(pairs_dev, thr, max_iter, seed, device):
    from .gpu_verify import verify_inliers_kornia
    return [verify_inliers_kornia(qm, rm, reproj_thresh=thr, max_iter=max_iter,
                                  seed=seed, device=device).shape[0]
            for (qm, rm) in pairs_dev]


def _run_batch(pairs_dev, thr, n_hyp, score_type, refine_iter, seed, device, budget):
    from .gpu_verify import ransac_homography_batch
    inls = ransac_homography_batch(pairs_dev, reproj_thresh=thr, n_hyp=n_hyp, score_type=score_type,
                                   refine=refine_iter > 0, refine_iter=refine_iter, device=device,
                                   seed=seed, points_budget=budget)
    return [x.shape[0] for x in inls]


def _time(fn, warmup, repeats, device):
    for _ in range(warmup):
        fn()
    _sync(device)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        _sync(device)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(np.min(ts)), out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True, help="--dump-mnn .npz from map_rerank")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backends", nargs="+", default=["cpu_magsac", "gpu_kornia", "gpu_batch"],
                    choices=["cpu_magsac", "gpu_kornia", "gpu_batch"])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=1, help="cpu_magsac cv2 verify threads")
    ap.add_argument("--gpu-n-hyp", type=int, default=256)
    ap.add_argument("--gpu-max-iter", type=int, default=10)
    ap.add_argument("--gpu-score-type", choices=["msac", "ransac"], default="msac")
    ap.add_argument("--gpu-refine-iter", type=int, default=5)
    ap.add_argument("--gpu-seed", type=int, default=0)
    ap.add_argument("--gpu-points-budget", type=int, default=24_000_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    pairs, nq, thr = _load_pairs(args.pairs)
    P = len(pairs)
    Ms = np.array([p[0].shape[0] for p in pairs])
    print(f"[pairs] {P} pairs | n_mutual: min={Ms.min()} median={int(np.median(Ms))} "
          f"mean={Ms.mean():.1f} p90={int(np.percentile(Ms, 90))} max={Ms.max()} "
          f"| <4: {(Ms < 4).sum()} | n_query median={int(np.median(nq))} | thr={thr}", flush=True)

    dev = args.device
    results, counts_by = {}, {}

    for backend in args.backends:
        _reset_mem(dev)
        if backend == "cpu_magsac":
            # transfer N/A (host arrays). fit = the cv2 loop.
            fit_med, fit_min, counts = _time(lambda: _run_cpu(pairs, thr, args.jobs),
                                             args.warmup, args.repeats, "cpu")
            transfer_s, peak = 0.0, float("nan")
        else:
            # host→device ONCE (measured), then time the fit on resident GPU tensors
            _sync(dev); t0 = time.perf_counter()
            pairs_dev = [(torch.as_tensor(qm, device=dev, dtype=torch.float64),
                          torch.as_tensor(rm, device=dev, dtype=torch.float64)) for (qm, rm) in pairs]
            _sync(dev); transfer_s = time.perf_counter() - t0
            if backend == "gpu_kornia":
                fn = lambda: _run_kornia(pairs_dev, thr, args.gpu_max_iter, args.gpu_seed, dev)
            else:
                fn = lambda: _run_batch(pairs_dev, thr, args.gpu_n_hyp, args.gpu_score_type,
                                        args.gpu_refine_iter, args.gpu_seed, dev, args.gpu_points_budget)
            fit_med, fit_min, counts = _time(fn, args.warmup, args.repeats, dev)
            peak = _peak_mem_mb(dev)

        counts = np.asarray(counts, float)
        counts_by[backend] = counts
        scores = counts / np.maximum(nq, 1)
        rec = {"fit_s_median": fit_med, "fit_s_min": fit_min, "transfer_s": transfer_s,
               "full_s_median": fit_med + transfer_s, "pairs_per_s": P / fit_med if fit_med else 0.0,
               "peak_mem_mb": peak,
               "inliers": {"mean": float(counts.mean()), "median": float(np.median(counts)),
                           "p90": float(np.percentile(counts, 90)), "max": float(counts.max()),
                           "frac_zero": float((counts == 0).mean())},
               "score": {"mean": float(scores.mean()), "median": float(np.median(scores)),
                         "max": float(scores.max())}}
        results[backend] = rec
        print(f"[{backend}] fit={fit_med * 1e3:.1f}ms (min {fit_min * 1e3:.1f}) "
              f"transfer={transfer_s * 1e3:.1f}ms full={rec['full_s_median'] * 1e3:.1f}ms "
              f"| {rec['pairs_per_s']:.0f} pairs/s peak={peak:.0f}MB "
              f"| inliers mean={counts.mean():.1f} med={np.median(counts):.0f} "
              f"zero={100 * (counts == 0).mean():.0f}%", flush=True)

    # agreement of each GPU backend vs cpu_magsac (does the reranker see different per-pair numbers?)
    if "cpu_magsac" in counts_by:
        base = counts_by["cpu_magsac"]
        for backend in args.backends:
            if backend == "cpu_magsac":
                continue
            c = counts_by[backend]
            both_pos = ((base > 0) & (c > 0))
            agree_zero = ((base == 0) == (c == 0)).mean()
            rho = _spearman(base, c)
            mae = float(np.abs(base - c).mean())
            results[backend]["vs_cpu"] = {"spearman_inliers": rho, "mae_inliers": mae,
                                          "frac_zero_agree": float(agree_zero),
                                          "both_positive": int(both_pos.sum())}
            print(f"[{backend} vs cpu] spearman(inliers)={rho:.3f} MAE={mae:.1f} "
                  f"zero-agree={100 * agree_zero:.0f}% both>0={int(both_pos.sum())}/{P}", flush=True)

    payload = {"meta": {"pairs": P, "n_mutual_median": int(np.median(Ms)),
                        "n_query_median": int(np.median(nq)), "reproj_thresh": thr,
                        "device": dev, "warmup": args.warmup, "repeats": args.repeats,
                        "jobs": args.jobs, "gpu_n_hyp": args.gpu_n_hyp,
                        "gpu_score_type": args.gpu_score_type, "gpu_refine_iter": args.gpu_refine_iter},
               "backends": results}
    if args.out:
        Path(args.out).expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[ok] -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
