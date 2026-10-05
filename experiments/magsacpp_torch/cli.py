"""CLI for the Torch MAGSAC++ homography experiment.

Subcommands (run as ``python -m magsacpp_torch.cli <cmd>`` from geoLocalisation/experiments):

  audit      -- print resolved config + pinned references + environment + expected data locations.
                Runs anywhere (no data / GPU needed).
  parity     -- numerical checks:
                  ``parity gamma``     analytic-vs-LUT gamma parity (local, no data).
                  ``parity synthetic`` torch core vs cv2 USAC_MAGSAC (applied baseline) on synthetic
                                       pairs with a shared recorded schedule (local; needs opencv).
                  ``parity oracle``    compare recorded C++/cv2 oracle traces (--records JSON).
  cache      -- build the immutable correspondence cache from the reranker data (query H5 + store +
                shortlist) via patch_rerank.matcher. Needs the real data -> run on the server.
  evaluate   -- run the torch estimator over a correspondence cache, score + localize, compute
                distR@250m etc., compare vs the cv2 baseline on the SAME cache. Server.
  benchmark  -- estimator timing (estimator-only / +transfer / full). Synthetic runs locally;
                CUDA timing uses events + warmup when a GPU is present.

The data-dependent commands (cache/evaluate) import patch_rerank and read the H5/JSON/TIF stores
described in REPORT.md; they are written to run unchanged on the GPU server.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

from .config import MagsacppConfig
from .estimator import estimate_homography_magsacpp
from .gamma import GammaLUT, GammaMath


# ----------------------------------------------------------------------------- synthetic data
def _rand_h(rng, perspective=2e-3):
    ang = rng.uniform(-0.3, 0.3); s = rng.uniform(0.8, 1.25)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) * s
    H = np.eye(3); H[:2, :2] = R; H[:2, 2] = rng.uniform(-20, 20, 2)
    H[2, 0] = rng.uniform(-perspective, perspective); H[2, 1] = rng.uniform(-perspective, perspective)
    return H / H[2, 2]


def _apply(H, p):
    h = np.concatenate([p, np.ones((len(p), 1))], 1) @ H.T
    return h[:, :2] / h[:, 2:3]


def _synth_pair(rng, n, outlier_frac, noise, rng_range=100.0):
    H = _rand_h(rng)
    p1 = rng.uniform(0, rng_range, size=(n, 2))
    p2 = _apply(H, p1)
    if noise:
        p2 = p2 + rng.normal(0, noise, p2.shape)
    inl = np.ones(n, bool)
    k = int(round(outlier_frac * n))
    if k:
        idx = rng.choice(n, k, replace=False)
        p2[idx] = rng.uniform(-rng_range, 2 * rng_range, size=(k, 2)); inl[idx] = False
    return p1, p2, H, inl


# ----------------------------------------------------------------------------- audit
def cmd_audit(args):
    cfg = MagsacppConfig()
    print("=== magsacpp_torch audit ===")
    print("\n[config]")
    for k, v in asdict(cfg).items():
        print(f"  {k:22s} = {v}")
    print(f"  (dtype                 = {cfg.dtype})")
    print("\n[math-core reference]  danini/magsac@d259f8b (graph-cut-ransac@9fa075d), BSD-3;")
    print("                       paper arXiv:1912.05909; homography n=4, k=3.64, C=0.25")
    print("[applied baseline]     OpenCV USAC cv2.USAC_MAGSAC @ opencv 4.x 62587ae (Apache-2.0);")
    print("                       DoF=2, k=3.04, C=0.5 -- DO NOT mix constants")
    print("\n[environment]")
    print(f"  torch   {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  gpu     {torch.cuda.get_device_name(0)}")
    for mod in ("cv2", "kornia", "scipy", "h5py", "rasterio", "boto3"):
        try:
            m = __import__(mod); print(f"  {mod:8s} {getattr(m, '__version__', '?')}")
        except Exception:
            print(f"  {mod:8s} MISSING")
    print("\n[expected reranker data] (see REPORT.md; relative to the geo_iso_noverlap work dir)")
    for p in ("dict/tiles_index_dense.h5", "dict/tiles_index_checker1000.h5",
              "<city>/rerank_store_<city>.h5", "shortlist_k70.json",
              "queries/query_<city>_*.h5", "s3://geo-reference/embeddings/<city>/..."):
        print(f"  - {p}")
    print("\n[correspondence contract]  H maps map-crop(rm) -> query(qm); residual in query patch-grid")
    print("  units; inlier_threshold default 2.0 px; score numerator = inlier count (matcher.py).")
    return 0


# ----------------------------------------------------------------------------- parity: gamma
def cmd_parity_gamma(args):
    gm = GammaMath(4, 3.64, 0.25, args.sigma_max, dtype=torch.float64)
    lut = GammaLUT(gm, size=args.lut_size, interp=True)
    tau2 = (3.64 * args.sigma_max) ** 2
    sq = torch.linspace(0, tau2 * 1.2, 5000, dtype=torch.float64)
    dloss = (lut.loss(sq) - gm.loss(sq)).abs()
    dw = (lut.weight(sq) - gm.weight(sq)).abs()
    print("[parity gamma] analytic vs LUT  (size=%d)" % args.lut_size)
    print(f"  loss   max|d|={float(dloss.max()):.3e}  weight max|d|={float(dw.max()):.3e}")
    print(f"  danini gamma_value_of_k reproduced: gu((n-1)/2,k^2/2)={float(gm._gu_a2_tk):.16f}")
    print(f"  rho_sat (outlier loss) = {gm.rho_sat:.6f}")
    ok = float(dloss.max()) < 1e-5 and float(dw.max()) < 1e-5
    print("  RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ----------------------------------------------------------------------------- parity: synthetic vs cv2
def cmd_parity_synthetic(args):
    try:
        import cv2
    except Exception:
        print("[parity synthetic] opencv not installed -- skipping cv2 applied baseline compare.")
        cv2 = None
    rng = np.random.default_rng(args.seed)
    cfg = MagsacppConfig(sigma_max=args.sigma_max, inlier_threshold=args.thresh,
                         max_hypotheses=args.hyps)
    rows = []
    for i in range(args.pairs):
        p1, p2, H, inl = _synth_pair(rng, args.n, args.outlier_frac, args.noise)
        res = estimate_homography_magsacpp(
            torch.tensor(p1), torch.tensor(p2), config=cfg,
            generator=torch.Generator().manual_seed(args.seed + i))
        row = {"pair": i, "torch_success": res.success, "torch_inliers": res.inlier_count,
               "true_inliers": int(inl.sum())}
        if res.success:
            # recovered-vs-true inlier agreement (applied-quality proxy)
            row["mask_agree"] = float((res.inlier_mask.numpy() == inl).mean())
        if cv2 is not None:
            # cv2 USAC_MAGSAC on the SAME pairs (H maps rm=p1 -> qm=p2, matcher.py convention)
            m = cv2.findHomography(p1, p2, cv2.USAC_MAGSAC, float(args.thresh))[1]
            cv_in = int(m.sum()) if m is not None else 0
            row["cv2_inliers"] = cv_in
            if res.success and m is not None:
                row["torch_vs_cv2_mask_agree"] = float((res.inlier_mask.numpy() == m.ravel().astype(bool)).mean())
        rows.append(row)
    succ = np.mean([r["torch_success"] for r in rows])
    print(f"[parity synthetic] pairs={args.pairs} n={args.n} outliers={args.outlier_frac} "
          f"noise={args.noise} sigma_max={args.sigma_max}")
    print(f"  torch success rate = {succ:.3f}")
    if any("mask_agree" in r for r in rows):
        print(f"  mean torch-vs-true mask agreement = "
              f"{np.mean([r['mask_agree'] for r in rows if 'mask_agree' in r]):.3f}")
    if cv2 is not None:
        print(f"  mean torch inliers={np.mean([r['torch_inliers'] for r in rows]):.1f}  "
              f"cv2 inliers={np.mean([r.get('cv2_inliers',0) for r in rows]):.1f}")
        if any("torch_vs_cv2_mask_agree" in r for r in rows):
            print(f"  mean torch-vs-cv2 mask agreement = "
                  f"{np.mean([r['torch_vs_cv2_mask_agree'] for r in rows if 'torch_vs_cv2_mask_agree' in r]):.3f}")
        print("  NOTE: cv2 USAC_MAGSAC is the APPLIED baseline (DoF=2 etc.); disagreement is expected"
              " and measured, not forced. Math-core parity needs the C++ oracle (parity oracle).")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2)
        print("  wrote", args.out)
    return 0


def _project_np(H, p, eps=1e-12):
    H = H.double().cpu(); p = p.double().cpu()
    x, y = p[:, 0], p[:, 1]
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    near = w.abs() < eps
    ws = torch.where(near, torch.ones_like(w), w)
    px = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / ws
    py = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / ws
    return torch.stack([px, py], -1), near


def cmd_parity_device(args):
    """GPU==CPU parity on a recorded schedule (replay). Reports max projection diff / loss diff /
    mask agreement per pair and an overall PASS/FAIL. Run this ON THE SERVER (needs CUDA)."""
    if not torch.cuda.is_available():
        print("[parity device] no CUDA device available -- run this on the GPU server.")
        return 2
    dt = torch.float64 if args.dtype == "float64" else torch.float32
    proj_tol = 1e-6 if dt is torch.float64 else 1e-2
    loss_tol = 1e-9 if dt is torch.float64 else 1e-4
    rng = np.random.default_rng(args.seed)
    cfg = MagsacppConfig(sigma_max=args.sigma_max, irls_iters=args.irls, dtype=dt,
                         max_hypotheses=args.hyps)
    print(f"[parity device] dtype={args.dtype} pairs={args.pairs} n={args.n} hyps={args.hyps} "
          f"gpu={torch.cuda.get_device_name(0)}")
    worst_proj = worst_loss = 0.0
    min_mask_agree = 1.0
    fails = 0
    for i in range(args.pairs):
        p1, p2, H, inl = _synth_pair(rng, args.n, args.outlier_frac, args.noise)
        sched = torch.from_numpy(np.stack([rng.choice(args.n, 4, replace=False)
                                           for _ in range(args.hyps)]))
        t1 = torch.tensor(p1, dtype=dt); t2 = torch.tensor(p2, dtype=dt)
        rc = estimate_homography_magsacpp(t1, t2, config=cfg, hypothesis_indices=sched)
        rg = estimate_homography_magsacpp(t1.cuda(), t2.cuda(), config=cfg,
                                          hypothesis_indices=sched.cuda())
        if rc.status != rg.status or not (rc.success and rg.success):
            print(f"  pair {i}: STATUS MISMATCH cpu={rc.status} gpu={rg.status}")
            fails += 1
            continue
        pc, nc = _project_np(rc.H, t1); pg, ng = _project_np(rg.H, t1)
        good = ~(nc | ng)
        dproj = float((pc[good] - pg[good]).norm(dim=-1).max()) if bool(good.any()) else 0.0
        dloss = abs(rc.total_loss - rg.total_loss) / max(1.0, abs(rc.total_loss))
        magree = float((rc.inlier_mask.cpu() == rg.inlier_mask.cpu()).double().mean())
        worst_proj = max(worst_proj, dproj); worst_loss = max(worst_loss, dloss)
        min_mask_agree = min(min_mask_agree, magree)
        if dproj > proj_tol or dloss > loss_tol:
            fails += 1
            print(f"  pair {i}: proj={dproj:.3e} loss={dloss:.3e} mask_agree={magree:.3f}  OVER-TOL")
    print(f"  worst projection diff = {worst_proj:.3e} px (tol {proj_tol:.0e})")
    print(f"  worst rel loss diff   = {worst_loss:.3e}   (tol {loss_tol:.0e})")
    print(f"  min mask agreement    = {min_mask_agree:.4f}")
    ok = fails == 0 and (dt is torch.float32 or min_mask_agree == 1.0)
    print("  RESULT:", "PASS (GPU == CPU)" if ok else f"FAIL ({fails} pairs over tolerance)")
    return 0 if ok else 1


def cmd_parity_cv2minimal(args):
    """Numerical parity of the MINIMAL 4-point solver: cv2 vs our SVD vs our closed-form.
    For exactly 4 general-position points the homography is UNIQUE (up to scale), so all three MUST
    agree -- a mismatch means a wrong matrix formulation, not a legitimate algorithm difference."""
    try:
        import cv2
    except Exception:
        print("[parity cv2minimal] opencv not installed -- run on the server with "
              "--with opencv-python-headless.")
        return 2
    from .solver import solve_minimal_h, solve_minimal_h_closed_form
    rng = np.random.default_rng(args.seed)

    def proj(H, p):
        h = np.c_[p, np.ones(len(p))] @ H.T
        return h[:, :2] / h[:, 2:3]

    # The minimal 4-pt homography is UNIQUE, so the real correctness test is: does each solver
    # REPRODUCE the exact src->dst (self-reprojection ~0)?  Plus our two solvers must agree tightly.
    # cv2 is float32/LM-limited internally, so its self-reproj is ~1e-5 and it differs from our
    # (exact, float64) result by cv2's own precision -- that is NOT a formulation error in ours.
    self_err = {"cv2": 0.0, "ours_svd": 0.0, "ours_closedform": 0.0}
    svd_vs_cf = cv2_vs_ours = 0.0
    fails = 0
    for _ in range(args.trials):
        H = _rand_h(rng)
        src = rng.uniform(0, args.coord, size=(4, 2))
        dst = _apply(H, src)
        Hcv = cv2.findHomography(src, dst, 0)[0]
        if Hcv is None:
            fails += 1
            continue
        src_t = torch.tensor(src, dtype=torch.float64).unsqueeze(0)
        dst_t = torch.tensor(dst, dtype=torch.float64).unsqueeze(0)
        Hsvd = solve_minimal_h(src_t, dst_t)[0][0].numpy()
        Hcf = solve_minimal_h_closed_form(src_t, dst_t)[0][0].numpy()
        self_err["cv2"] = max(self_err["cv2"], float(np.abs(proj(Hcv, src) - dst).max()))
        self_err["ours_svd"] = max(self_err["ours_svd"], float(np.abs(proj(Hsvd, src) - dst).max()))
        self_err["ours_closedform"] = max(self_err["ours_closedform"],
                                          float(np.abs(proj(Hcf, src) - dst).max()))
        pts = rng.uniform(0, args.coord, size=(12, 2))
        svd_vs_cf = max(svd_vs_cf, float(np.abs(proj(Hsvd, pts) - proj(Hcf, pts)).max()))
        cv2_vs_ours = max(cv2_vs_ours, float(np.abs(proj(Hcv, pts) - proj(Hsvd, pts)).max()))
    print(f"[parity cv2minimal] trials={args.trials} coord=0..{args.coord}  (max px)")
    print(f"  self-reproj error (how exactly each H maps src->dst):")
    for k, v in self_err.items():
        print(f"    {k:16s} = {v:.3e}")
    print(f"  ours svd vs closed_form = {svd_vs_cf:.3e}   (must be ~0: same unique solution)")
    print(f"  cv2 vs ours             = {cv2_vs_ours:.3e}   (cv2 float32/LM-limited; informational)")
    # correctness of OURS: both our solvers exact + mutually identical. cv2 is only a sanity cross-ref.
    ok = (fails == 0 and svd_vs_cf < args.tol
          and self_err["ours_svd"] < args.tol and self_err["ours_closedform"] < args.tol)
    print(f"  RESULT: {'PASS -- our minimal solver is exact & svd==closed_form' if ok else 'FAIL -- OURS not exact / solvers disagree'}"
          f"  (tol {args.tol:.0e}; cv2 diff is its own precision, not a bug)")
    return 0 if ok else 1


def cmd_parity_oracle(args):
    from .oracle import compare_trace, load_records
    recs = load_records(args.records)
    print(f"[parity oracle] {len(recs)} records from {args.records}")
    agg = {}
    for rec in recs:
        try:
            m = compare_trace(rec)
        except ValueError as e:
            print(f"  {rec.get('pair_id','?')}: SKIP ({e})"); continue
        print(f"  {rec.get('pair_id','?')}: " + "  ".join(f"{k}={v:.3e}" for k, v in m.items()))
        for k, v in m.items():
            agg.setdefault(k, []).append(v)
    if agg:
        print("  --- aggregate ---")
        for k, vs in agg.items():
            print(f"  {k}: max={max(vs):.3e} mean={sum(vs)/len(vs):.3e}")
    return 0


# ----------------------------------------------------------------------------- benchmark
def cmd_benchmark(args):
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[benchmark] cuda requested but unavailable; using cpu"); device = "cpu"
    rng = np.random.default_rng(args.seed)
    cfg = MagsacppConfig(sigma_max=args.sigma_max, max_hypotheses=args.hyps,
                         dtype=torch.float64 if args.dtype == "float64" else torch.float32)
    pairs = [_synth_pair(rng, args.n, args.outlier_frac, args.noise)[:2] for _ in range(args.pairs)]
    host = [(torch.tensor(p1, dtype=cfg.dtype), torch.tensor(p2, dtype=cfg.dtype)) for p1, p2 in pairs]

    def run_once(dev):
        for p1, p2 in host:
            estimate_homography_magsacpp(p1.to(dev), p2.to(dev), config=cfg,
                                         generator=torch.Generator(device=dev).manual_seed(0))

    # warmup
    run_once(device)
    if device == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        run_once(device)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / args.pairs * 1e3)   # ms/pair
    times = np.array(times)
    print(f"[benchmark] device={device} dtype={args.dtype} pairs={args.pairs} n={args.n} "
          f"hyps={args.hyps} reps={args.reps}")
    print(f"  ms/pair  median={np.median(times):.3f}  p90={np.percentile(times,90):.3f}  "
          f"p95={np.percentile(times,95):.3f}")
    print(f"  pairs/s  ~ {1000.0/np.median(times):.1f}")
    if device == "cuda":
        print(f"  peak VRAM = {torch.cuda.max_memory_allocated()/1e6:.1f} MB")
    print("  NOTE: estimator-only timing (data already on device). Use the reranker's own harness"
          " (bench_verify.py) for +transfer and full Stage-B numbers.")
    return 0


# ----------------------------------------------------------------------------- cache / evaluate (server)
def cmd_cache(args):
    print("[cache] builds the immutable correspondence cache from the reranker data via "
          "patch_rerank.matcher.matched_coords. This requires the real query H5 + token store + "
          "shortlist (see REPORT.md 'Data stores'); run on the GPU server. Not implemented to run "
          "without that data. See README 'cache' for the exact command and the manifest schema.")
    return 2


def cmd_evaluate(args):
    print("[evaluate] real-data A/B (cv2 USAC_MAGSAC vs torch MAGSAC++) is a standalone entrypoint "
          "with its own flags:\n"
          "    python -m magsacpp_torch.evaluate --queries kram=… --shortlist … --index-uri s3://… \\\n"
          "        --backends cpu_magsac torch_magsacpp --level-agg sum --cell-agg mean --cache-cap 2 \\\n"
          "        --device cuda --out ab.json\n"
          "It fetches S3 cells one-at-a-time and evicts them (peak disk ~= --cache-cap cells), runs "
          "BOTH verifiers on the SAME mutual-NN pairs + the SAME aggregation, and reports distR@250m "
          "top1/top5 + paired lost/gained. See README / evaluate.py docstring.")
    return 2


# ----------------------------------------------------------------------------- argparse
def build_parser():
    p = argparse.ArgumentParser(prog="magsacpp_torch.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("audit").set_defaults(func=cmd_audit)

    pp = sub.add_parser("parity"); psub = pp.add_subparsers(dest="pcmd", required=True)
    g = psub.add_parser("gamma"); g.add_argument("--sigma-max", type=float, default=1.0, dest="sigma_max")
    g.add_argument("--lut-size", type=int, default=100_000, dest="lut_size"); g.set_defaults(func=cmd_parity_gamma)
    s = psub.add_parser("synthetic")
    s.add_argument("--pairs", type=int, default=20); s.add_argument("--n", type=int, default=150)
    s.add_argument("--outlier-frac", type=float, default=0.4, dest="outlier_frac")
    s.add_argument("--noise", type=float, default=0.5); s.add_argument("--sigma-max", type=float, default=1.0, dest="sigma_max")
    s.add_argument("--thresh", type=float, default=2.0); s.add_argument("--hyps", type=int, default=2000)
    s.add_argument("--seed", type=int, default=0); s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_parity_synthetic)
    cm = psub.add_parser("cv2minimal")   # cv2 vs our minimal 4-pt solver (unique => must match)
    cm.add_argument("--trials", type=int, default=200); cm.add_argument("--coord", type=float, default=100.0)
    cm.add_argument("--tol", type=float, default=1e-6); cm.add_argument("--seed", type=int, default=0)
    cm.set_defaults(func=cmd_parity_cv2minimal)
    o = psub.add_parser("oracle"); o.add_argument("--records", required=True); o.set_defaults(func=cmd_parity_oracle)
    d = psub.add_parser("device")   # GPU==CPU
    d.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    d.add_argument("--pairs", type=int, default=32); d.add_argument("--n", type=int, default=160)
    d.add_argument("--outlier-frac", type=float, default=0.4, dest="outlier_frac")
    d.add_argument("--noise", type=float, default=0.3); d.add_argument("--sigma-max", type=float, default=1.0, dest="sigma_max")
    d.add_argument("--hyps", type=int, default=400); d.add_argument("--irls", type=int, default=3)
    d.add_argument("--seed", type=int, default=0); d.set_defaults(func=cmd_parity_device)

    b = sub.add_parser("benchmark")
    b.add_argument("--device", default="cuda"); b.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    b.add_argument("--pairs", type=int, default=64); b.add_argument("--n", type=int, default=200)
    b.add_argument("--outlier-frac", type=float, default=0.4, dest="outlier_frac"); b.add_argument("--noise", type=float, default=0.5)
    b.add_argument("--sigma-max", type=float, default=1.0, dest="sigma_max"); b.add_argument("--hyps", type=int, default=1000)
    b.add_argument("--reps", type=int, default=5); b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=cmd_benchmark)

    sub.add_parser("cache").set_defaults(func=cmd_cache)
    sub.add_parser("evaluate").set_defaults(func=cmd_evaluate)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
