"""Field-by-field cv2-vs-GPU comparison on the frozen diagnostics archive.

``compare_gpu`` compares only inlier COUNTS (and the re-ranked result). This tool compares the FULL
per-call output that ``cv2.findHomography(USAC_MAGSAC)`` gives — **H** and the **inlier mask** —
against the Torch MAGSAC++ port, by replaying the archived ``rm``/``qm`` (no DINO/matching/S3):

  * **mask IoU** — |cv2∩gpu| / |cv2∪gpu| over the correspondences (do they call the SAME points inliers?);
  * **H reprojection diff** — ‖π(H_cv2·rm) − π(H_gpu·rm)‖ over the points (H is up-to-scale, so compare
    how differently the two homographies MAP the actual points, in query patch-grid px);
  * **Δinliers** — gpu − cv2 count.

Aggregated overall and per pyramid level, so you see where cv2 and the GPU port agree/diverge field-wise.
This is the GPU counterpart of the cv2 archive: the same outputs (H, mask), directly comparable.

Run::  python -m patch_rerank.compare_fields --diag-dir /…/diag_kram_test --device cuda [--max-queries N]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np


def _arrays_path(uri: str, diag_dir: Path, tmp: str) -> str:
    """Resolve a diag-archive arrays H5 location: download from s3:// into ``tmp`` once, else the
    local ``diag_dir/<uri>``. (Inlined from the removed ``compare_gpu``.)"""
    if uri.startswith("s3://"):
        import boto3
        b, key = uri[5:].split("/", 1)
        loc = os.path.join(tmp, os.path.basename(key))
        if not os.path.exists(loc):
            boto3.client("s3").download_file(b, key, loc)
        return loc
    return str(diag_dir / uri)


def _transform(H, pts):
    """H (P,3,3), pts (P,N,2) → (P,N,2) projected (homogeneous divide)."""
    import torch
    ones = torch.ones(*pts.shape[:-1], 1, dtype=pts.dtype, device=pts.device)
    pr = torch.einsum("pij,pnj->pni", H, torch.cat([pts, ones], -1))
    return pr[..., :2] / pr[..., 2:3]


def compute(diag_dir, *, device="cuda", n_hyp=256, sigma_max=2.0, reproj_thresh=2.0,
            max_queries=0, only_done=True, verbose=False) -> dict:
    import sys

    import h5py
    import torch
    exp = str(Path(__file__).resolve().parents[1] / "experiments")
    if exp not in sys.path:
        sys.path.insert(0, exp)
    from magsacpp_torch import MagsacppConfig
    from magsacpp_torch.batch import estimate_homography_magsacpp_batch
    cfg = MagsacppConfig(dtype=torch.float32, minimal_solver="closed_form", max_hypotheses=int(n_hyp),
                         sigma_max=float(sigma_max), inlier_threshold=float(reproj_thresh))

    d = Path(diag_dir).expanduser()
    done = set(json.loads((d / "completed.json").read_text())["queries"]) if (d / "completed.json").exists() else None
    arrays_of = {}
    for line in (d / "summary.jsonl").read_text(encoding="utf-8").splitlines():
        s = json.loads(line)
        arrays_of[s["query_id"]] = s.get("arrays") or f"arrays/{s['query_id'].replace(':', '_').replace('/', '_')}.h5"
    recs = collections.defaultdict(list)
    for line in (d / "records.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["cv2_called"] and r["h5_group"]:
            recs[r["query_id"]].append((r["h5_group"], r["level_m"]))

    todo = [q for q in recs if (not only_done or done is None or q in done) and q in arrays_of]
    if max_queries:
        todo = todo[:max_queries]

    iou_sum = collections.defaultdict(float); iou_n = collections.defaultdict(int)
    hdiff_vals = collections.defaultdict(list)
    dcount = collections.defaultdict(float)
    n_total = 0; both_same = 0
    tmp = tempfile.mkdtemp()
    for qi, q in enumerate(todo):
        t0 = time.perf_counter()
        groups = recs[q]
        qm_l, rm_l, Hcv_l, mcv_l, lvl_l = [], [], [], [], []
        with h5py.File(_arrays_path(arrays_of[q], d, tmp), "r") as f:
            for grp, lvl in groups:
                g = f[grp]
                qm_l.append(np.asarray(g["qm"])); rm_l.append(np.asarray(g["rm"]))
                Hcv_l.append(np.asarray(g["H"])); mcv_l.append(np.asarray(g["mask"]).astype(bool))
                lvl_l.append(int(lvl))
        P = len(qm_l)
        if not P:
            continue
        nmax = max(a.shape[0] for a in rm_l)
        p1 = torch.zeros(P, nmax, 2, dtype=torch.float32, device=device)   # rm
        p2 = torch.zeros(P, nmax, 2, dtype=torch.float32, device=device)   # qm
        valid = torch.zeros(P, nmax, dtype=torch.bool, device=device)
        mcv = torch.zeros(P, nmax, dtype=torch.bool, device=device)
        Hcv = torch.zeros(P, 3, 3, dtype=torch.float32, device=device)
        for i in range(P):
            m = rm_l[i].shape[0]
            p1[i, :m] = torch.as_tensor(rm_l[i], dtype=torch.float32, device=device)
            p2[i, :m] = torch.as_tensor(qm_l[i], dtype=torch.float32, device=device)
            valid[i, :m] = True
            mcv[i, :m] = torch.as_tensor(mcv_l[i], device=device)
            Hcv[i] = torch.as_tensor(Hcv_l[i], dtype=torch.float32, device=device)

        results = estimate_homography_magsacpp_batch(p1, p2, valid, config=cfg)   # list of P results
        Hg = torch.stack([r.H.to(torch.float32) for r in results])                # (P,3,3)
        mg = torch.zeros(P, nmax, dtype=torch.bool, device=device)
        for i, r in enumerate(results):
            mm = r.inlier_mask.to(device)
            mg[i, :mm.shape[0]] = mm[:nmax] if mm.shape[0] >= nmax else torch.nn.functional.pad(
                mm, (0, nmax - mm.shape[0]))

        inter = (mcv & mg & valid).sum(1).double()
        union = ((mcv | mg) & valid).sum(1).double().clamp(min=1)
        iou = (inter / union).cpu().numpy()
        gpu_c = (mg & valid).sum(1).cpu().numpy()
        cv2_c = (mcv & valid).sum(1).cpu().numpy()
        # H reprojection difference over valid points (skip pairs with non-finite gpu H)
        ok = torch.isfinite(Hg.reshape(P, -1)).all(1)
        pg = _transform(Hg, p1); pc = _transform(Hcv, p1)
        dd = torch.linalg.norm(pg - pc, dim=-1)
        dd = torch.where(valid & torch.isfinite(dd), dd, torch.full_like(dd, float("nan")))
        hdiff = torch.nanmean(dd, dim=1).cpu().numpy()

        for i in range(P):
            lv = lvl_l[i]
            iou_sum[lv] += float(iou[i]); iou_n[lv] += 1
            dcount[lv] += float(gpu_c[i] - cv2_c[i])
            if bool(ok[i]) and np.isfinite(hdiff[i]):
                hdiff_vals[lv].append(float(hdiff[i]))
            n_total += 1
            if int(gpu_c[i]) == int(cv2_c[i]) and float(iou[i]) == 1.0:
                both_same += 1
        if verbose:
            print(f"[fields] {qi + 1}/{len(todo)} {q}: {P} calls ({time.perf_counter() - t0:.1f}s)",
                  flush=True)

    levels = sorted(iou_n)
    allv = [v for lv in levels for v in hdiff_vals[lv]]

    def _med(xs):
        return float(np.median(xs)) if xs else None
    return {
        "n_calls": n_total,
        "identical_frac": (both_same / n_total if n_total else None),   # same count AND same inlier set
        "mask_iou": {"mean": (sum(iou_sum.values()) / max(1, sum(iou_n.values()))),
                     "by_level_m": {lv: round(iou_sum[lv] / iou_n[lv], 4) for lv in levels}},
        "H_reproj_diff_px": {"median": _med(allv),
                             "by_level_m": {lv: round(_med(hdiff_vals[lv]) or 0.0, 4) for lv in levels}},
        "delta_inliers": {"mean": (sum(dcount.values()) / max(1, n_total)),
                          "by_level_m": {lv: round(dcount[lv] / iou_n[lv], 4) for lv in levels}},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-hyp", type=int, default=256)
    ap.add_argument("--sigma-max", type=float, default=2.0)
    ap.add_argument("--reproj-thresh", type=float, default=2.0)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: <diag-dir>/compare_fields.json")
    args = ap.parse_args(argv)
    res = compute(args.diag_dir, device=args.device, n_hyp=args.n_hyp, sigma_max=args.sigma_max,
                  reproj_thresh=args.reproj_thresh, max_queries=args.max_queries, verbose=True)
    out = Path(args.out).expanduser() if args.out else Path(args.diag_dir).expanduser() / "compare_fields.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[fields] n={res['n_calls']} identical={res['identical_frac']} "
          f"mask_IoU_mean={res['mask_iou']['mean']:.4f} "
          f"H_reproj_median_px={res['H_reproj_diff_px']['median']} "
          f"meanΔinl={res['delta_inliers']['mean']:.4f}", flush=True)
    print(f"[fields] mask IoU by level: {res['mask_iou']['by_level_m']}", flush=True)
    print(f"[ok] -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
