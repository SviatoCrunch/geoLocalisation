"""Lightweight LOCAL fine rerank: GPU MAGSAC++ between clip-frame tokens and the COARSE gallery cell
grids — real homography inlier-ratio per frame↔cell WITHOUT the 138 GiB pyramid store.

Why this exists: the full ``pyramid_newformat`` store (8 levels × 25 positions × 32×32) is 138 GiB and
needs S3 + lots of RAM/GPU. But the geometric match only needs *a* token grid per cell with 2D positions,
and the coarse gallery (``kram20_prodaction.h5``, ~0.37 GiB, one 16×16 d1024 grid per cell) already has
that. All cells fit on the GPU at once (~0.4 GB), so we mutual-NN each clip frame against its coarse
shortlist cells and run the exact same GPU MAGSAC++ primitives as ``pyramid_newformat.locate``
([[project_magsacpp_torch]]). Output is the SAME locate-schema JSON (``per_query[qid].topk`` with
``cell_score`` = inlier ratio + cell-centroid lat/lon), so ``clip_search.aggregate`` consumes it directly
(now with a REAL geometric confidence instead of the flat coarse cosine).

Run (from ~/work/geoLocalisation)::

    uv run --python 3.11 --with "torch==2.5.1" --with h5py --with numpy --with tqdm python -m clip_search.fine_local \
      --queries kram=/…/clips/query_clips_d1024.h5 \
      --gallery kram=/…/out/gallery_h5/kram20_prodaction.h5 \
      --shortlist /…/clips/shortlist_clips_k100.json \
      --city kram --k 100 --out /…/clips/locate_fine_local.json --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _kv(a):
    c, p = a.split("=", 1)
    return c, p


def _gallery_latlon(gallery_h5: str, city: str) -> dict:
    import h5py
    out = {}
    with h5py.File(Path(gallery_h5).expanduser(), "r") as f:
        for k in f.keys():
            g = f[k]
            if hasattr(g, "keys") and "ift_dino" in g:
                out[f"{city}:{k}"] = (float(g.attrs.get("lat", float("nan"))),
                                      float(g.attrs.get("lon", float("nan"))))
    return out


def _candidates(ind, qid):
    if isinstance(ind, dict) and "shortlist" in ind:
        return (ind["shortlist"].get(qid) or {}).get("cells", [])
    v = ind.get(qid, [])
    return v.get("cells", []) if isinstance(v, dict) else v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True, help="city=query_clips_d1024.h5 (clip frame tokens)")
    ap.add_argument("--gallery", required=True, help="city=kram20_prodaction.h5 (coarse cell grids)")
    ap.add_argument("--shortlist", required=True, help="coarse shortlist JSON (per-qid candidate cells)")
    ap.add_argument("--city", required=True)
    ap.add_argument("--k", type=int, default=100, help="coarse cells per frame to geometrically verify")
    ap.add_argument("--gpu-cells", type=int, default=16, help="cells per MAGSAC batch (bounds VRAM)")
    ap.add_argument("--reproj-thresh", type=float, default=2.0)
    ap.add_argument("--mpp-sigma-max", type=float, default=2.0)
    ap.add_argument("--mpp-hyps", type=int, default=256)
    ap.add_argument("--mpp-irls", type=int, default=1)
    ap.add_argument("--mpp-seed", type=int, default=0)
    ap.add_argument("--match-budget", type=int, default=50_000_000)
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
    import numpy as np
    import torch
    from tqdm import tqdm

    from magsacpp_torch import MagsacppConfig
    from magsacpp_torch.batch import estimate_homography_magsacpp_batch
    from magsacpp_torch.search import _mutual_nn_padded
    from patch_rerank.matcher import grid_keypoints
    from patch_rerank.query_io import QueryGridStore
    from geo_e2c_train.data import TileGridLoader

    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    cfg = MagsacppConfig(sigma_max=args.mpp_sigma_max, inlier_threshold=args.reproj_thresh,
                         max_hypotheses=args.mpp_hyps, irls_iters=args.mpp_irls,
                         minimal_solver="closed_form", dtype=torch.float32)
    gen = torch.Generator(device=dev).manual_seed(args.mpp_seed)

    gcity, gpath = _kv(args.gallery)
    latlon = _gallery_latlon(gpath, gcity)
    tile_ids = sorted(latlon.keys())
    tiles = TileGridLoader({gcity: gpath}, grid_size=None, cache=True)

    # preload ALL coarse cell grids onto the GPU once: (n_cells, Hc*Wc, D) + grid keypoints
    g0 = tiles.grid(tile_ids[0]); Hc, Wc, D = g0.shape
    cell_feats = torch.stack([tiles.grid(t).reshape(Hc * Wc, D) for t in tile_ids]).to(dev)   # (Nc, HW, D)
    kp = grid_keypoints(Hc, Wc)
    row = {t: i for i, t in enumerate(tile_ids)}
    print(f"[gallery] {len(tile_ids)} cells, grid {Hc}x{Wc}, D={D} -> GPU "
          f"({cell_feats.element_size() * cell_feats.nelement() / 1e9:.2f} GB)", flush=True)

    qs = QueryGridStore(dict([_kv(args.queries)]))
    ind = json.loads(Path(args.shortlist).expanduser().read_text())
    qids = [q for q in ind.get("shortlist", ind).keys() if qs.has(q)]
    if args.max_queries:
        qids = qids[:args.max_queries]
    print(f"[queries] {len(qids)} frames to verify", flush=True)

    per_query = {}
    for qid in tqdm(qids, desc="fine_local", unit="frame"):
        feat, xy, lat, lon = qs.get(qid)
        feat = feat.to(dev); nq = int(feat.shape[0])
        cells = [c for c in _candidates(ind, qid)[:args.k] if c in row]
        if not cells or nq == 0:
            per_query[qid] = {"topk": [], "gt": None}
            continue
        scored = []
        for g0i in range(0, len(cells), args.gpu_cells):
            sub = cells[g0i:g0i + args.gpu_cells]
            grids = cell_feats[[row[c] for c in sub]]                 # (b, HW, D) already on GPU
            qm, rm, valid = _mutual_nn_padded(feat, grids, xy, kp, dev, args.match_budget)
            counts, _ = estimate_homography_magsacpp_batch(rm, qm, valid, config=cfg, generator=gen,
                                                           return_counts=True)
            ratio = (counts.double() / nq).cpu().numpy()
            for c, r in zip(sub, ratio):
                la, lo = latlon.get(c, (float("nan"), float("nan")))
                scored.append({"cell_id": c, "cell_score": float(r), "lat": la, "lon": lo})
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()
        scored.sort(key=lambda d: d["cell_score"], reverse=True)
        per_query[qid] = {"topk": scored, "gt": None}

    out = {"meta": {"gallery": gpath, "queries": args.queries, "k": args.k,
                    "verifier": "magsacpp_torch (local coarse-grid)", "grid": [Hc, Wc]},
           "per_query": per_query}
    Path(args.out).expanduser().write_text(json.dumps(out), encoding="utf-8")
    print(f"[ok] fine_local ({len(qids)} frames) -> {args.out}", flush=True)
    tiles.close(); qs.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
