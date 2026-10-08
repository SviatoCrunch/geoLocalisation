"""Faithful-MAGSAC++ GPU rerank over the packed new-format store, driven by EXTERNAL top-K indices.

Takes 1..N query frames + a per-query candidate index list (coarse top-K, supplied from outside) + a
new-format dataset DIRECTORY, and reranks each query's candidate cells on the GPU. Reuses
``magsacpp_torch.search.search`` verbatim (same scoring / level-sum / cell-mean / dist metrics); only
the cell store is swapped to :class:`pyramid_newformat.loader.NewFormatCellStore`.

``--indices`` JSON is either ``{query_id: [idx, ...]}`` or a shortlist ``{"shortlist": {query_id:
{"cells": [...]}}}``; idx is a ``global_cell_idx`` (int) or a ``cell_id`` (str).

Run: see README.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _candidates(indices_obj, qid):
    if isinstance(indices_obj, dict) and "shortlist" in indices_obj:
        e = indices_obj["shortlist"].get(qid)
        return (e or {}).get("cells", [])
    v = indices_obj.get(qid, [])
    return v.get("cells", []) if isinstance(v, dict) else v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", nargs="+", required=True, help="city=query_grids.h5 (1..N frames)")
    ap.add_argument("--indices", required=True, help="JSON: {qid:[idx...]} or {'shortlist':{qid:{cells}}}")
    ap.add_argument("--dataset-dir", required=True, help="new-format store dir (local or s3://)")
    ap.add_argument("--k-coarse", type=int, default=1000, help="cap candidates per query")
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
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]               # geoLocalisation
    for p in (str(repo), str(repo / "experiments")):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch

    from magsacpp_torch import MagsacppConfig
    from magsacpp_torch.search import search
    from patch_rerank.query_io import QueryGridStore

    from .loader import NewFormatCellStore

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    cfg = MagsacppConfig(sigma_max=args.mpp_sigma_max, inlier_threshold=args.reproj_thresh,
                         max_hypotheses=args.mpp_hyps, irls_iters=args.mpp_irls,
                         minimal_solver=args.mpp_solver,
                         dtype=torch.float64 if args.mpp_dtype == "float64" else torch.float32)
    gen = torch.Generator(device=device).manual_seed(args.mpp_seed)

    qstore = QueryGridStore(dict(a.split("=", 1) for a in args.queries))
    store = NewFormatCellStore(args.dataset_dir)
    ind = json.loads(Path(args.indices).expanduser().read_text())

    qids = ind.get("shortlist", ind).keys() if isinstance(ind, dict) else []
    queries, shortlist = {}, {"shortlist": {}}
    for qid in qids:
        if not qstore.has(qid):
            continue
        if args.max_queries and len(queries) >= args.max_queries:
            break
        feat, xy, lat, lon = qstore.get(qid)
        cands = [store.resolve(x) for x in _candidates(ind, qid)][:args.k_coarse]
        cands = [c for c in cands if store.has(c)]
        if not cands:
            continue
        queries[qid] = {"feat": feat.to(device), "xy": xy, "lat": lat, "lon": lon}
        shortlist["shortlist"][qid] = {"cells": cands}
    if not queries:
        print("[search_newformat] no queries matched indices/store/query-h5"); return 2

    per_query, summary, speed = search(queries, shortlist, store, config=cfg, k_coarse=args.k_coarse,
                                       topk=args.topk, level_agg=args.level_agg, cell_agg=args.cell_agg,
                                       device=device, generator=gen, flat=True)
    out = {"meta": {"dataset_dir": args.dataset_dir, "k_coarse": args.k_coarse, "topk": args.topk,
                    "level_agg": args.level_agg, "cell_agg": args.cell_agg, "device": str(device),
                    "mpp": {"sigma_max": args.mpp_sigma_max, "hyps": args.mpp_hyps, "irls": args.mpp_irls,
                            "dtype": args.mpp_dtype, "solver": args.mpp_solver}},
           "summary": summary, "per_query": per_query, "speed": speed}
    Path(args.out).expanduser().write_text(json.dumps(out), encoding="utf-8")
    if summary.get("top1"):
        print(f"[search_newformat] images={summary['n_images']} with_gt={summary['n_with_gt']} "
              f"top1 distR@250m={summary['top1']['distR@250m']:.3f} "
              f"top{args.topk} distR@250m={summary['topk']['distR@250m']:.3f}", flush=True)
    else:
        print(f"[search_newformat] images={summary['n_images']} (no GT)", flush=True)
    print(f"[speed] fetch={speed['fetch_s']:.1f}s match={speed['match_s']:.1f}s verify={speed['verify_s']:.1f}s", flush=True)
    print(f"[ok] -> {args.out}", flush=True)
    store.close(); qstore.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
