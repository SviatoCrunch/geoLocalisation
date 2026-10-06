"""Discrimination diagnosis for a diagnostics archive — why does fine rerank miss?

From ``summary.jsonl`` (the full reranked list, every candidate with its cell_score + distance to GT),
per frame it finds the GT-closest candidate (the best the rerank COULD pick) and compares it to the
actual rerank top-1. Aggregates answer two things:

  * **fine ceiling** — fraction of frames that have ANY candidate position within distR of GT
    (if low: not a ranking problem — the right place isn't even a good candidate position / matching
    is off).
  * **flat vs wrong** — median rerank-rank of the GT-closest candidate and the score GAP to top-1
    (small gap + scattered rank → scores don't discriminate = "flat"; GT cell low score + high rank →
    its matches are genuinely weak = "wrong").

Run::  python -m patch_rerank.diag_gap --diag-dir /…/diag_kram_test
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

_THR = (250.0, 500.0, 1000.0)


def _median(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def compute(diag_dir) -> dict:
    d = Path(diag_dir).expanduser()
    best_rank, best_dist, best_score, top1_score, top1_dist, gap, rel_gap = ([] for _ in range(7))
    n = 0
    for line in (d / "summary.jsonl").read_text(encoding="utf-8").splitlines():
        s = json.loads(line)
        gt = s.get("gt")
        if not gt or None in gt:
            continue
        rr = [r for r in (s.get("reranked_top") or []) if r.get("dist_m") is not None]
        if not rr:
            continue
        n += 1
        top1 = min(rr, key=lambda r: r.get("rerank_rank", 10 ** 9))
        best = min(rr, key=lambda r: r["dist_m"])            # closest candidate to GT = the fine ceiling
        best_rank.append(best.get("rerank_rank"))
        best_dist.append(best["dist_m"]); best_score.append(best["cell_score"])
        top1_score.append(top1["cell_score"]); top1_dist.append(top1["dist_m"])
        g = top1["cell_score"] - best["cell_score"]
        gap.append(g)
        rel_gap.append(g / top1["cell_score"] if top1["cell_score"] else None)

    ceiling = {f"best_candidate_within_{int(t)}m": (sum(x <= t for x in best_dist) / n if n else None)
               for t in _THR}
    return {
        "n_frames": n,
        "fine_ceiling": ceiling,                              # could rerank EVER hit it?
        "gt_closest_candidate": {
            "median_rerank_rank": _median(best_rank),         # where the right cell actually ranks
            "is_rank1_frac": (sum(r == 1 for r in best_rank) / n if n else None),
            "median_cell_score": _median(best_score)},
        "top1": {"median_cell_score": _median(top1_score), "median_dist_m": _median(top1_dist)},
        "score_gap_top1_minus_gtcell": {"median": _median(gap), "median_relative": _median(rel_gap)},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag-dir", required=True)
    ap.add_argument("--out", default=None, help="default: <diag-dir>/gap.json")
    args = ap.parse_args(argv)
    r = compute(args.diag_dir)
    out = Path(args.out).expanduser() if args.out else Path(args.diag_dir).expanduser() / "gap.json"
    out.write_text(json.dumps(r, indent=2), encoding="utf-8")
    c, gc, sg = r["fine_ceiling"], r["gt_closest_candidate"], r["score_gap_top1_minus_gtcell"]
    print(f"[ceiling] best candidate within 250m={c['best_candidate_within_250m']} "
          f"500m={c['best_candidate_within_500m']} 1000m={c['best_candidate_within_1000m']}", flush=True)
    print(f"[gt-cell] median rerank_rank={gc['median_rerank_rank']} is_rank1={gc['is_rank1_frac']} "
          f"median_score={gc['median_cell_score']}", flush=True)
    print(f"[flat?]  top1 median_score={r['top1']['median_cell_score']}  "
          f"gap(top1-gtcell) median={sg['median']} rel={sg['median_relative']}", flush=True)
    print(f"[ok] -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
