"""§7 positive-label + false-negative audit (geometry only; never auto-relabels)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import write_json, _COS


def run(ctx) -> dict:
    sr = ctx.sr["train"]
    thr = 250.0
    rows, n_susp = [], 0
    for i, qid in enumerate(sr.query_ids):
        pos = sorted(int(x) for x in sr.relevance.pos_of(i))
        d = np.linalg.norm(sr.tile_xy - sr.q_xy[i], axis=1) * _COS          # (M,) true metres
        pos_d = [float(d[r]) for r in pos]
        near = set(int(r) for r in np.nonzero(d <= thr)[0])
        susp = sorted(near - set(pos))                                      # close but NOT positive
        n_susp += len(susp)
        rows.append({"query_id": qid, "n_positives": len(pos),
                     "min_pos_dist_m": (min(pos_d) if pos_d else None),
                     "n_false_negative_suspects_within_250m": len(susp),
                     "false_negative_suspect_rows": susp[:20]})
    npos = [r["n_positives"] for r in rows]
    out = {"n_queries": len(sr.query_ids), "threshold_m": thr,
           "mean_positives_per_query": float(np.mean(npos)) if npos else 0.0,
           "queries_with_zero_positives": int(sum(1 for r in rows if r["n_positives"] == 0)),
           "total_false_negative_suspects": n_susp,
           "note": "suspects = gallery tiles within 250 m of the query GT that are NOT labeled "
                   "positive (overlapping-tile near-duplicates the loss may penalise). NOT relabeled."}
    write_json(Path(ctx.args.output_dir) / "labels_audit.json", {**out, "per_query": rows[:200]})
    return {**out, "_per_query": {"labels_train": rows}}
