"""Top-K ranking curves from a diagnostics archive — recomputed offline (no re-run).

All per-level scores are stored in ``aggregation.jsonl``, so the cell ranking can be recomputed under
any aggregation and read off at any K, directly from the archive. For each K (1/5/10/15/20/50/100) and
each ``level_agg`` (``sum`` = the kup recipe that worked, ``max`` = best-level) it reports:

  * **loc distR@{250,500,1000}m @top-K** — fraction of frames whose BEST top-K predicted point is within
    the threshold of GT (the usual localisation recall curve);
  * **gtcell_recall@K** — fraction of frames whose GT-closest candidate cell ranks within top-K
    (how well ranking surfaces the right cell).

Ranking rule = the service one: position_score = agg over its levels, cell_score = mean over positions,
predicted point = the best-scoring position of the cell.

Run::  python -m patch_rerank.diag_rank_curves --diag-dir /…/diag_kram_test
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from .diag_stats import _haversine_m

_THR = (250.0, 500.0, 1000.0)
_KS = (1, 5, 10, 15, 20, 50, 100)


def _rank(cells, agg):
    """cells = list of aggregation rows for one query → ranked [(cell_id, score, lat, lon)] desc."""
    out = []
    for c in cells:
        ps, lat, lon = [], [], []
        for p in c["positions"]:
            lv = [v for v in p["level_scores"].values() if v is not None]
            ps.append((sum(lv) if agg == "sum" else (max(lv) if lv else 0.0)))
            lat.append(p["pos_lat"]); lon.append(p["pos_lon"])
        bi = max(range(len(ps)), key=lambda i: ps[i])
        out.append((c["cell_id"], sum(ps) / len(ps), lat[bi], lon[bi]))   # cell_agg = mean
    out.sort(key=lambda r: r[1], reverse=True)
    return out


def compute(diag_dir, aggs=("sum", "max")) -> dict:
    d = Path(diag_dir).expanduser()
    gt = {}
    for line in (d / "summary.jsonl").read_text(encoding="utf-8").splitlines():
        s = json.loads(line)
        if s.get("gt") and None not in s["gt"]:
            gt[s["query_id"]] = s["gt"]
    cells = collections.defaultdict(list)
    for line in (d / "aggregation.jsonl").read_text(encoding="utf-8").splitlines():
        a = json.loads(line)
        if a["query_id"] in gt:
            cells[a["query_id"]].append(a)

    res = {}
    for agg in aggs:
        dist_at = {K: [] for K in _KS}             # per-frame best top-K predicted distance
        gtrank = []                                 # per-frame rank of GT-closest cell
        for q, cs in cells.items():
            ranked = _rank(cs, agg)
            g = gt[q]
            dists = [_haversine_m(g[0], g[1], la, lo) for (_, _, la, lo) in ranked]
            for K in _KS:
                dist_at[K].append(min(dists[:K]) if dists else None)
            gtrank.append(1 + min(range(len(dists)), key=lambda i: dists[i]))  # rank of closest cell
        n = len(cells)
        curve = {}
        for K in _KS:
            dd = [x for x in dist_at[K] if x is not None]
            curve[K] = {f"distR@{int(t)}m": round(sum(x <= t for x in dd) / len(dd), 4) if dd else None
                        for t in _THR}
            curve[K]["gtcell_recall"] = round(sum(r <= K for r in gtrank) / n, 4) if n else None
        res[agg] = {"n_frames": n, "by_topK": curve}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag-dir", required=True)
    ap.add_argument("--aggs", default="sum,max", help="comma list of level_agg to compare (sum,max)")
    ap.add_argument("--out", default=None, help="default: <diag-dir>/rank_curves.json")
    args = ap.parse_args(argv)
    res = compute(args.diag_dir, aggs=tuple(a.strip() for a in args.aggs.split(",")))
    out = Path(args.out).expanduser() if args.out else Path(args.diag_dir).expanduser() / "rank_curves.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    for agg, r in res.items():
        print(f"\n== level_agg={agg}  (n={r['n_frames']}) ==", flush=True)
        print(f"{'topK':>5} | {'distR@250':>9} {'distR@500':>9} {'distR@1000':>10} | {'gtcell_recall':>13}",
              flush=True)
        for K in _KS:
            c = r["by_topK"][K]
            print(f"{K:>5} | {str(c['distR@250m']):>9} {str(c['distR@500m']):>9} "
                  f"{str(c['distR@1000m']):>10} | {str(c['gtcell_recall']):>13}", flush=True)
    print(f"\n[ok] -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
