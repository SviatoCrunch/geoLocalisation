"""Locate frame(s) over a LOCAL packed gallery (картотека) given the top-K cell indices we pass in.

One engine, two front-doors:

  # in code
  from pyramid_newformat.locate import locate
  res = locate(
      queries={"kup": "/…/query_kup_d1024.h5"},              # name -> query H5 (or a QueryGridStore)
      indices="/…/shortlist_kup_prod_k100.json",             # path OR {qid: [idx…]} OR shortlist dict
      dataset_dir="/home/ubuntu/work/mpp_local/kup",         # local packed new-format dir (or S3)
      topk=5, device="cuda")
  res["per_query"]["<qid>"]["topk"]       # ranked cells: cell_id, cell_score, lat, lon, level_m, rank, dist_m
  res["summary"]                          # top1/topK distR@{250,500,1000}m (when GT present)
  res["speed"]                            # wall_s / read_s / gpu_s

  # CLI
  python -m pyramid_newformat.locate --queries kup=/…/query.h5 \
      --indices /…/shortlist.json --dataset-dir /…/kup --out out.json

``idx`` may be a ``global_cell_idx`` (int/str) or a ``cell_id``. Read path = minimum-reads shard sweep
(coalesced sequential runs into a reusable buffer, optional 1-shard-ahead prefetch); GPU scoring =
faithful MAGSAC++ (magsacpp_torch), identical math/aggregation to the batch harness. Each union cell is
read once across queries.
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

try:
    from tqdm import tqdm as _tqdm
except Exception:                                             # progress bar optional
    def _tqdm(x, **k):
        return x


def _haversine_m(a, b, c, d):
    r = 6371000.0
    p1, p2 = math.radians(a), math.radians(c)
    dp, dl = math.radians(c - a), math.radians(d - b)
    x = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(x)))


def _candidates(ind, qid):
    """Top-K cell indices for one query, accepting either a shortlist dict or a plain {qid:[idx…]}."""
    if isinstance(ind, dict) and "shortlist" in ind:
        return (ind["shortlist"].get(qid) or {}).get("cells", [])
    v = ind.get(qid, [])
    return v.get("cells", []) if isinstance(v, dict) else v


def _load_indices(indices):
    if isinstance(indices, (str, Path)):
        return json.loads(Path(indices).expanduser().read_text())
    return indices


def _make_query_store(queries):
    """queries -> (store, owns). Accepts {name: h5_path} (built here, owned) or a ready QueryGridStore."""
    from patch_rerank.query_io import QueryGridStore
    if isinstance(queries, dict):
        return QueryGridStore(dict(queries)), True
    return queries, False                                      # a pre-built store: caller keeps ownership


def locate(queries, indices, dataset_dir, *, k_coarse=1000, topk=5, gpu_cells=4,
           level_agg="sum", cell_agg="mean", prefetch=True, device="cuda",
           reproj_thresh=2.0, mpp_sigma_max=2.0, mpp_hyps=256, mpp_irls=1,
           mpp_dtype="float32", mpp_solver="closed_form", mpp_seed=0,
           match_budget=50_000_000, max_queries=0, out=None, progress=True) -> dict:
    """Search ``queries`` over the local packed gallery at ``dataset_dir`` using the supplied top-K
    ``indices``. Returns ``{meta, summary, per_query, speed}`` and (if ``out``) writes it as JSON."""
    def say(*a):
        if progress:
            print(*a, flush=True)

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

    from .reader import NewFormatReader

    dev = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
    cfg = MagsacppConfig(sigma_max=mpp_sigma_max, inlier_threshold=reproj_thresh,
                         max_hypotheses=mpp_hyps, irls_iters=mpp_irls, minimal_solver=mpp_solver,
                         dtype=torch.float64 if mpp_dtype == "float64" else torch.float32)
    gen = torch.Generator(device=dev).manual_seed(mpp_seed)

    r = NewFormatReader(dataset_dir)
    P, L, H, W, D = r.P, r.L, r.H, r.W, r.D
    levels = r.levels_m
    kp = grid_keypoints(H, W)
    by_gidx = {int(c["global_cell_idx"]): cid for cid, c in r.cells.items()}

    def resolve(x):
        return by_gidx[int(x)] if (isinstance(x, (int, np.integer)) or str(x).lstrip("-").isdigit()) else x

    qs, owns = _make_query_store(queries)
    ind = _load_indices(indices)
    Q, cand = {}, {}
    for qid in list(ind.get("shortlist", ind).keys()):
        if not qs.has(qid):
            continue
        if max_queries and len(Q) >= max_queries:
            break
        feat, xy, lat, lon = qs.get(qid)
        cells = [c for c in (resolve(x) for x in _candidates(ind, qid)) if c in r.cells][:k_coarse]
        if not cells:
            continue
        Q[qid] = {"feat": feat.to(dev), "xy": xy, "nq": int(feat.shape[0]), "lat": lat, "lon": lon}
        cand[qid] = set(cells)
    if not Q:
        say("[locate] no queries matched")
        r.close()
        if owns:
            qs.close()
        return {"summary": {"n_images": 0}, "per_query": {}, "speed": {}}

    union = set().union(*cand.values())
    byfile = {}                                               # file -> sorted [(local_idx, cell_id)]
    for cid in union:
        c = r.cells[cid]; byfile.setdefault(c["file"], []).append((int(c["local_cell_idx"]), cid))
    for f in byfile:
        byfile[f].sort()
    qneed = {cid: [q for q in Q if cid in cand[q]] for cid in union}
    qcell = {q: {} for q in Q}
    files = sorted(byfile)
    say(f"[locate] queries={len(Q)} cells={len(union)} shards={len(files)} "
        f"(cells/shard={[len(byfile[f]) for f in files]}) gpu_cells={gpu_cells} prefetch={prefetch}")

    # Reusable bounded buffers sized to the largest shard — no per-shard realloc, no np.stack copy.
    # Prefetch needs two (one filling ahead while the other is on the GPU); else one.
    maxcells = max(len(byfile[f]) for f in files)
    nbuf = 2 if prefetch else 1
    bufs = [np.empty((maxcells,) + r.block_shape, np.dtype("<f2")) for _ in range(nbuf)]

    def load_shard(fname, buf):                               # read this shard's needed cells (coalesced runs)
        cids = [c for _, c in byfile[fname]]
        _, order = r.read_cells_into(cids, buf)               # buf[:len(order)] filled; order = read order
        co = [r.read_cell_coords(c) for c in order]
        return order, buf, co

    t_read = t_gpu = 0.0
    ex = ThreadPoolExecutor(max_workers=1) if prefetch else None
    # Only arm the 1-shard-ahead future when prefetch is enabled; otherwise the initial submit would
    # read shard 0 in the background AND load_shard(fname) would read it again (double read).
    fut = ex.submit(load_shard, files[0], bufs[0]) if (prefetch and files) else None
    t0 = time.perf_counter()
    bar = _tqdm(files, desc="locate", unit="shard", disable=not progress)
    for fi, fname in enumerate(bar):
        tr = time.perf_counter()
        cids, arr, co = fut.result() if prefetch else load_shard(fname, bufs[0])
        sr = time.perf_counter() - tr; t_read += sr
        if prefetch:
            fut = (ex.submit(load_shard, files[fi + 1], bufs[(fi + 1) % 2])
                   if fi + 1 < len(files) else None)
        tg = time.perf_counter()
        n = len(cids)
        for cs in range(0, n, gpu_cells):
            ce = min(cs + gpu_cells, n); nc = ce - cs
            sub_cids = cids[cs:ce]
            qset = {q for c in sub_cids for q in qneed[c]}
            accq = {q: np.zeros((nc, P, L), np.float64) for q in qset}
            for li_ in range(L):
                crops = np.ascontiguousarray(arr[cs:ce, :, li_]).reshape(nc * P, H * W, D)
                grids = torch.from_numpy(crops).to(dev)
                for q in qset:
                    p = Q[q]
                    qm, rm, valid = _mutual_nn_padded(p["feat"], grids, p["xy"], kp, dev, match_budget)
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
                    pyr = s.sum(1) if level_agg == "sum" else s.max(1)
                    bi = int(pyr.argmax())
                    cs_ = {"mean": float(pyr.mean()), "min": float(pyr.min()), "max": float(pyr.max())}[cell_agg]
                    lvl_i = int(s[bi].argmax())
                    qcell[q][cid] = {"cell_id": cid, "cell_score": cs_, "lat": float(plat[bi]),
                                     "lon": float(plon[bi]), "level_m": int(levels[lvl_i])}
        del arr
        if str(dev).startswith("cuda"):
            torch.cuda.synchronize()
        sg = time.perf_counter() - tg; t_gpu += sg
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(cells=n, read=f"{sr:.1f}s", gpu=f"{sg:.1f}s")
        else:
            say(f"  shard {fi + 1}/{len(files)} {fname}: cells={n} read={sr:.1f}s gpu={sg:.1f}s")
    wall = time.perf_counter() - t0
    if ex is not None:
        ex.shutdown()

    thr = (250.0, 500.0, 1000.0)
    per_query, d1, dk = {}, [], []
    for q in Q:
        p = Q[q]; has_gt = math.isfinite(p["lat"]) and math.isfinite(p["lon"])
        ranked = sorted(qcell[q].values(), key=lambda v: v["cell_score"], reverse=True)[:topk]
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
        summary[f"top{topk}"] = {f"distR@{int(t)}m": float((np.asarray(dk) <= t).mean()) for t in thr}
    result = {"meta": {"dataset_dir": dataset_dir, "gpu_cells": gpu_cells, "prefetch": prefetch,
                       "topk": topk, "level_agg": level_agg, "cell_agg": cell_agg, "device": str(dev)},
              "summary": summary, "per_query": per_query,
              "speed": {"wall_s": wall, "read_s": t_read, "gpu_s": t_gpu, "n_shards": len(files),
                        "n_cells": len(union)}}
    if out:
        Path(out).expanduser().write_text(json.dumps(result), encoding="utf-8")
    say(f"[locate] DONE cells={len(union)} shards={len(files)} | wall={wall:.1f}s read={t_read:.1f}s gpu={t_gpu:.1f}s"
        + (f" | top{topk} distR@250m={summary[f'top{topk}']['distR@250m']:.3f}" if d1 else ""))
    if out:
        say(f"[ok] -> {out}")
    r.close()
    if owns:
        qs.close()
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", nargs="+", required=True, help="name=path query H5 (one or more)")
    ap.add_argument("--indices", required=True, help="shortlist JSON or {qid:[idx...]} JSON")
    ap.add_argument("--dataset-dir", required=True, help="local packed new-format dir (or s3://)")
    ap.add_argument("--gpu-cells", type=int, default=4, dest="gpu_cells",
                    help="cells per per-level GPU batch (crops = gpu_cells*n_positions)")
    ap.add_argument("--no-prefetch", dest="prefetch", action="store_false",
                    help="disable 1-shard-ahead read (low RAM)")
    ap.add_argument("--k-coarse", type=int, default=1000, dest="k_coarse")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--level-agg", choices=["sum", "max"], default="sum", dest="level_agg")
    ap.add_argument("--cell-agg", choices=["mean", "min", "max"], default="mean", dest="cell_agg")
    ap.add_argument("--reproj-thresh", type=float, default=2.0, dest="reproj_thresh")
    ap.add_argument("--mpp-sigma-max", type=float, default=2.0, dest="mpp_sigma_max")
    ap.add_argument("--mpp-hyps", type=int, default=256, dest="mpp_hyps")
    ap.add_argument("--mpp-irls", type=int, default=1, dest="mpp_irls")
    ap.add_argument("--mpp-dtype", choices=["float32", "float64"], default="float32", dest="mpp_dtype")
    ap.add_argument("--mpp-solver", choices=["svd", "closed_form"], default="closed_form", dest="mpp_solver")
    ap.add_argument("--mpp-seed", type=int, default=0, dest="mpp_seed")
    ap.add_argument("--match-budget", type=int, default=50_000_000, dest="match_budget")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-queries", type=int, default=0, dest="max_queries")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    res = locate(queries=dict(s.split("=", 1) for s in a.queries), indices=a.indices,
                 dataset_dir=a.dataset_dir, k_coarse=a.k_coarse, topk=a.topk, gpu_cells=a.gpu_cells,
                 level_agg=a.level_agg, cell_agg=a.cell_agg, prefetch=a.prefetch, device=a.device,
                 reproj_thresh=a.reproj_thresh, mpp_sigma_max=a.mpp_sigma_max, mpp_hyps=a.mpp_hyps,
                 mpp_irls=a.mpp_irls, mpp_dtype=a.mpp_dtype, mpp_solver=a.mpp_solver, mpp_seed=a.mpp_seed,
                 match_budget=a.match_budget, max_queries=a.max_queries, out=a.out)
    return 0 if res["summary"].get("n_images") else 2


if __name__ == "__main__":
    raise SystemExit(main())
