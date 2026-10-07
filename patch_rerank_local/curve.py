"""Top-K localisation curve from a :mod:`rerank_local` run (``--out`` + ``--dump-scores``).

For each query, ranks its candidate cells by the dumped cell score (``mean`` = cell-agg mean of the
per-position level-aggregated scores — matches the ranking the reranker used under ``--cell-agg mean``),
then reports, per K in {1,5,10,15,20,50,100}:

  * ``distR@{250,500,1000}m`` — fraction of frames whose best top-K predicted point is within the
    threshold of GT;
  * ``gtcell_recall@K`` — fraction of frames whose GT-closest candidate ranks within top-K.

Pure data (no conclusions). Writes ``rank_curves.json`` next to ``--out`` by default.

Run::  python -m patch_rerank_local.curve --search …/kup_search_sum.json --dump …/kup_scores_sum.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

_KS = (1, 5, 10, 15, 20, 50, 100)
_THR = (250.0, 500.0, 1000.0)


def _haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def compute(search_path: str, dump_path: str, rank_key: str = "mean") -> dict:
    search = json.loads(Path(search_path).expanduser().read_text(encoding="utf-8"))
    dump = json.loads(Path(dump_path).expanduser().read_text(encoding="utf-8"))
    pq = search.get("per_query", search)
    hit = {k: {t: 0 for t in _THR} for k in _KS}
    gtr = {k: 0 for k in _KS}
    n = 0
    for q, cells in dump.items():
        gt = pq.get(q, {}).get("gt")
        if not gt or gt[0] is None:
            continue
        n += 1
        ranked = sorted(cells.values(), key=lambda v: v[rank_key], reverse=True)
        dists = [_haversine_m(gt[0], gt[1], v["lat"], v["lon"]) for v in ranked]
        gi = 1 + min(range(len(dists)), key=lambda i: dists[i]) if dists else 10 ** 9
        for k in _KS:
            dk = min(dists[:k]) if dists else 10 ** 18
            for t in _THR:
                if dk <= t:
                    hit[k][t] += 1
            if gi <= k:
                gtr[k] += 1
    curve = {}
    for k in _KS:
        curve[k] = {f"distR@{int(t)}m": (hit[k][t] / n if n else None) for t in _THR}
        curve[k]["gtcell_recall"] = (gtr[k] / n if n else None)
    return {"n_frames": n, "rank_key": rank_key, "by_topK": curve}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", required=True, help="rerank_local --out JSON (has per_query gt)")
    ap.add_argument("--dump", required=True, help="rerank_local --dump-scores JSON (all candidates)")
    ap.add_argument("--rank-key", choices=["mean", "best"], default="mean",
                    help="cell score to rank by (mean = cell-agg mean, the reranker default)")
    ap.add_argument("--out", default=None, help="default: <search dir>/rank_curves.json")
    args = ap.parse_args(argv)
    res = compute(args.search, args.dump, rank_key=args.rank_key)
    out = Path(args.out).expanduser() if args.out else Path(args.search).expanduser().with_name("rank_curves.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    n = res["n_frames"]
    print(f"n_frames={n}  (rank_key={res['rank_key']})", flush=True)
    print(f"{'topK':>4} | {'distR@250':>9} {'distR@500':>9} {'distR@1000':>10} | {'gtcell_recall':>13}", flush=True)
    for k in _KS:
        c = res["by_topK"][k]
        f = lambda v: "None" if v is None else f"{v:.3f}"  # noqa: E731
        print(f"{k:>4} | {f(c['distR@250m']):>9} {f(c['distR@500m']):>9} "
              f"{f(c['distR@1000m']):>10} | {f(c['gtcell_recall']):>13}", flush=True)
    print(f"[ok] -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
