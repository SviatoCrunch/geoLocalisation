"""Path B: low-RAM GPU MAGSAC++ dump of per-(cell, position, level) inlier scores + homographies.

For each query frame × its coarse shortlist cell, runs the GPU MAGSAC++ ([[project_magsacpp_torch]])
over ALL 25 positions × 8 levels of that cell and records the inlier-ratio per (position, level) + the
homography H. Reads the 138 GiB ``pyramid_newformat`` store **one cell at a time from S3** (Range GET,
~0.4 GiB/cell → bounded RAM), scoring each union cell against every frame that shortlisted it (so each
cell is read once). Emits the SAME dump schema as ``patch_rerank/search_pyramid_s3 --diag-dir`` so
``clip_search.agg_sweep`` reads path A (cv2) and path B (GPU) identically:

  aggregation.jsonl   per query×cell: positions[{position_id,pos_lat,pos_lon,level_scores{Lm:score}}]
  summary.jsonl       per query: gt
  arrays/<query>.h5   group "<cell>": "scores" (P,L) + "H" (P,L,3,3)   (homographies; --no-h to skip)

This is the RAM-safe way to run the real server verifier on a small instance; use ``--max-queries`` /
``--k`` for a tractable subset to study aggregation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _kv(a):
    c, p = a.split("=", 1)
    return c, p


def _safe(s):
    return s.replace(":", "_").replace("/", "_")


def _candidates(ind, qid):
    if isinstance(ind, dict) and "shortlist" in ind:
        return (ind["shortlist"].get(qid) or {}).get("cells", [])
    v = ind.get(qid, [])
    return v.get("cells", []) if isinstance(v, dict) else v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True, help="city=query_clips_d1024.h5")
    ap.add_argument("--indices", required=True, help="coarse shortlist JSON")
    ap.add_argument("--dataset-dir", required=True, help="pyramid_newformat store (s3:// or local)")
    ap.add_argument("--diag-dir", required=True, help="output dump dir (aggregation/summary jsonl + arrays)")
    ap.add_argument("--k", type=int, default=100, help="shortlist cells per frame to verify")
    ap.add_argument("--grid-batch", type=int, default=200, help="(pos*level) grids per MAGSAC batch")
    ap.add_argument("--no-h", dest="dump_h", action="store_false", help="don't store homographies")
    ap.add_argument("--reproj-thresh", type=float, default=2.0)
    ap.add_argument("--mpp-sigma-max", type=float, default=2.0)
    ap.add_argument("--mpp-hyps", type=int, default=256)
    ap.add_argument("--mpp-irls", type=int, default=1)
    ap.add_argument("--mpp-seed", type=int, default=0)
    ap.add_argument("--match-budget", type=int, default=50_000_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-queries", type=int, default=0)
    args = ap.parse_args(argv)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    repo = Path(__file__).resolve().parents[1]
    for p in (str(repo), str(repo / "experiments")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import h5py
    import numpy as np
    import torch
    from tqdm import tqdm

    from magsacpp_torch import MagsacppConfig
    from magsacpp_torch.batch import estimate_homography_magsacpp_batch
    from magsacpp_torch.search import _mutual_nn_padded
    from patch_rerank.matcher import grid_keypoints
    from patch_rerank.query_io import QueryGridStore
    from pyramid_newformat.reader import NewFormatReader

    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    cfg = MagsacppConfig(sigma_max=args.mpp_sigma_max, inlier_threshold=args.reproj_thresh,
                         max_hypotheses=args.mpp_hyps, irls_iters=args.mpp_irls,
                         minimal_solver="closed_form", dtype=torch.float32)
    gen = torch.Generator(device=dev).manual_seed(args.mpp_seed)

    r = NewFormatReader(args.dataset_dir)
    P, L, H, W, D = r.P, r.L, r.H, r.W, r.D
    levels = [int(x) for x in r.levels_m]
    kp = grid_keypoints(H, W)
    by_gidx = {int(c["global_cell_idx"]): cid for cid, c in r.cells.items()}

    def resolve(x):
        return by_gidx[int(x)] if (isinstance(x, (int, np.integer)) or str(x).lstrip("-").isdigit()) else x

    qs = QueryGridStore(dict([_kv(args.queries)]))
    ind = json.loads(Path(args.indices).expanduser().read_text())
    qids = [q for q in ind.get("shortlist", ind).keys() if qs.has(q)]
    if args.max_queries:
        qids = qids[:args.max_queries]

    # per-query feats on CPU; candidate cells (resolved, existing) + coarse rank
    Q, cand = {}, {}
    for qid in qids:
        feat, xy, lat, lon = qs.get(qid)
        cells = [c for c in (resolve(x) for x in _candidates(ind, qid)) if c in r.cells][:args.k]
        if not cells:
            continue
        Q[qid] = {"feat": feat, "xy": xy, "nq": int(feat.shape[0]), "gt": (lat, lon)}
        cand[qid] = {c: rank for rank, c in enumerate(cells)}
    if not Q:
        print("[fine_dump] no queries matched"); return 2

    union = sorted(set().union(*[set(c) for c in cand.values()]),
                   key=lambda c: (r.cells[c]["file"], int(r.cells[c]["local_cell_idx"])))
    need = {c: [q for q in Q if c in cand[q]] for c in union}
    print(f"[fine_dump] queries={len(Q)} union_cells={len(union)} P={P} L={L} grid={H}x{W}", flush=True)

    dd = Path(args.diag_dir).expanduser(); (dd / "arrays").mkdir(parents=True, exist_ok=True)
    # accumulate per-query cell rows + per-query arrays; write at the end (subset-sized)
    agg_rows = {q: [] for q in Q}
    h5files = {q: h5py.File(dd / "arrays" / f"{_safe(q)}.h5", "w") for q in Q} if args.dump_h else {}

    for cid in tqdm(union, desc="fine_dump", unit="cell"):
        blk = r.read_cell(cid)                                   # (P, L, H, W, D) float16, one S3 GET
        plat, plon = r.read_cell_coords(cid)                    # (P,), (P,)
        grids = torch.from_numpy(np.ascontiguousarray(blk).reshape(P * L, H * W, D)).to(dev)
        for q in need[cid]:
            p = Q[q]; feat = p["feat"].to(dev); nq = p["nq"]
            counts_all = np.zeros(P * L, np.float64)
            H_all = np.zeros((P * L, 3, 3), np.float64) if args.dump_h else None
            for b0 in range(0, P * L, args.grid_batch):
                sub = grids[b0:b0 + args.grid_batch]
                qm, rm, valid = _mutual_nn_padded(feat, sub, p["xy"], kp, dev, args.match_budget)
                out = estimate_homography_magsacpp_batch(rm, qm, valid, config=cfg, generator=gen,
                                                         return_counts=True)
                counts = out[0]
                counts_all[b0:b0 + sub.shape[0]] = counts.detach().double().cpu().numpy()
                if args.dump_h and len(out) > 1 and out[1] is not None:
                    Hm = np.asarray(out[1].detach().cpu().numpy())
                    if Hm.shape[-2:] == (3, 3):
                        H_all[b0:b0 + sub.shape[0]] = Hm.reshape(-1, 3, 3)[:sub.shape[0]]
            s = (counts_all / nq if nq else counts_all).reshape(P, L)
            positions = [{"position_id": i, "pos_lat": float(plat[i]), "pos_lon": float(plon[i]),
                          "level_scores": {str(levels[j]): float(s[i, j]) for j in range(L)}}
                         for i in range(P)]
            agg_rows[q].append({"query_id": q, "cell_id": cid, "coarse_rank": cand[q][cid],
                                "positions": positions, "n_positions": P})
            if args.dump_h:
                g = h5files[q].require_group(_safe(cid))
                g.create_dataset("scores", data=s.astype(np.float32))
                g.create_dataset("H", data=H_all.reshape(P, L, 3, 3).astype(np.float32))
        del grids
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()

    with open(dd / "aggregation.jsonl", "w", encoding="utf-8") as af, \
         open(dd / "summary.jsonl", "w", encoding="utf-8") as sf:
        for q in Q:
            for row in agg_rows[q]:
                af.write(json.dumps(row) + "\n")
            la, lo = Q[q]["gt"]
            gt = [la, lo] if (la == la and lo == lo) else None     # nan-safe
            sf.write(json.dumps({"query_id": q, "gt": gt}) + "\n")
    for f in h5files.values():
        f.close()
    (dd / "manifest.json").write_text(json.dumps(
        {"source": "fine_dump (GPU magsacpp_torch)", "dataset_dir": args.dataset_dir,
         "P": P, "L": L, "levels_m": levels, "grid": [H, W], "n_queries": len(Q),
         "n_union_cells": len(union)}), encoding="utf-8")
    print(f"[ok] fine_dump -> {dd} (aggregation.jsonl + summary.jsonl"
          + (" + arrays/*.h5 with H" if args.dump_h else "") + ")", flush=True)
    r.close(); qs.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
