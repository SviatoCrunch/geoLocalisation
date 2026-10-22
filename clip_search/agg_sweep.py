"""Offline sweep of SMALL→BIG aggregations (25 positions × 8 levels → one 1000x1000 cell score).

Reads a rerank DIAGNOSTICS dump (``aggregation.jsonl`` + ``summary.jsonl``) produced by EITHER
``patch_rerank/search_pyramid_s3 --diag-dir`` (cv2, path A) OR ``clip_search/fine_dump`` (GPU MAGSAC++,
path B) — both emit the same schema — and re-aggregates the per-(position, level) inlier scores every way
WITHOUT re-running any MAGSAC. For each (level-agg × position-agg × position-predictor) it ranks the
candidate cells per query and reports 1000x1000 accuracy (gt-cell top-1/top-K + distR), so we can pick
the aggregation that maximizes the 1000 m cell choice.

Aggregators (training-free): sum, max, mean, ``lse`` (LogMeanExp, β-tunable mean↔max dial), ``softmax``
(self-weighted Σ softmax(βx)·x). Position predictor: ``argmax`` (best sub-position lat/lon → sub-cell) or
``centroid`` (softmax-weighted mean of position lat/lon).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

THRESH_M = (250.0, 500.0, 1000.0)


def _hav(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    x = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(x)))


def _reduce(x: np.ndarray, how: str, beta: float, axis=None):
    """Reduce along ``axis`` by one of sum/max/mean/lse/softmax (all training-free)."""
    if how == "sum":
        return x.sum(axis)
    if how == "max":
        return x.max(axis)
    if how == "mean":
        return x.mean(axis)
    if how == "lse":                                  # LogMeanExp: β→0 ~ mean, β→∞ ~ max
        m = x.max(axis, keepdims=True)
        n = x.shape[axis] if axis is not None else x.size
        out = (m.squeeze(axis) if axis is not None else m) + \
            np.log(np.exp(beta * (x - m)).sum(axis) / n) / beta
        return out
    if how == "softmax":                              # self-weighted Σ softmax(βx)·x
        m = x.max(axis, keepdims=True)
        w = np.exp(beta * (x - m)); w = w / w.sum(axis, keepdims=True)
        return (w * x).sum(axis)
    raise ValueError(how)


def load_dump(diag_dir: str):
    """→ {query_id: {"gt": (lat,lon)|None, "cells": {cell_id: {"S": (P,L) array, "lat": (P,), "lon": (P,),
    "levels": [L...]}}}}. Reads aggregation.jsonl (per query×cell positions×level_scores) + summary.jsonl (gt)."""
    d = Path(diag_dir).expanduser()
    gt = {}
    for line in (d / "summary.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line); g = r.get("gt")
        gt[r["query_id"]] = (tuple(g) if g else None)
    queries: dict = {}
    for line in (d / "aggregation.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        pos = r["positions"]
        levels = sorted({int(L) for p in pos for L in p["level_scores"]}, reverse=True)
        P, L = len(pos), len(levels)
        S = np.zeros((P, L), np.float64); lat = np.zeros(P); lon = np.zeros(P)
        for i, p in enumerate(pos):
            lat[i] = p["pos_lat"]; lon[i] = p["pos_lon"]
            for j, Lm in enumerate(levels):
                v = p["level_scores"].get(str(Lm))
                S[i, j] = 0.0 if v is None else float(v)
        q = queries.setdefault(r["query_id"], {"gt": gt.get(r["query_id"]), "cells": {}})
        q["cells"][r["cell_id"]] = {"S": S, "lat": lat, "lon": lon, "levels": levels}
    return queries


def score_cell(cell: dict, level_agg: str, pos_agg: str, pos_pred: str, beta: float):
    """(cell_score, pred_lat, pred_lon) for one cell under the chosen aggregation."""
    S = cell["S"]                                      # (P, L)
    per_pos = _reduce(S, level_agg, beta, axis=1)      # (P,) collapse levels
    cell_score = float(_reduce(per_pos, pos_agg, beta, axis=0))
    if pos_pred == "centroid":
        m = per_pos.max(); w = np.exp(beta * (per_pos - m)); w = w / w.sum()
        return cell_score, float((w * cell["lat"]).sum()), float((w * cell["lon"]).sum())
    bi = int(per_pos.argmax())                         # argmax position → sub-cell lat/lon
    return cell_score, float(cell["lat"][bi]), float(cell["lon"][bi])


def evaluate(queries: dict, level_agg, pos_agg, pos_pred, beta, topk=5):
    d1, dk = [], []
    for q in queries.values():
        gt = q["gt"]
        if not gt or not q["cells"]:
            continue
        ranked = []
        for cid, cell in q["cells"].items():
            cs, la, lo = score_cell(cell, level_agg, pos_agg, pos_pred, beta)
            ranked.append((cs, la, lo))
        ranked.sort(key=lambda t: t[0], reverse=True)
        top = ranked[:topk]
        d1.append(_hav(gt[0], gt[1], top[0][1], top[0][2]))
        dk.append(min(_hav(gt[0], gt[1], la, lo) for _, la, lo in top))
    d1 = np.asarray(d1); dk = np.asarray(dk)
    return {"n": int(d1.size),
            "top1": {int(t): float((d1 <= t).mean()) if d1.size else 0.0 for t in THRESH_M},
            f"top{topk}": {int(t): float((dk <= t).mean()) if dk.size else 0.0 for t in THRESH_M}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag-dir", required=True, help="dump dir (aggregation.jsonl + summary.jsonl), A or B")
    ap.add_argument("--level-aggs", nargs="+", default=["sum", "max", "mean", "lse", "softmax"])
    ap.add_argument("--pos-aggs", nargs="+", default=["max", "mean", "sum", "lse"])
    ap.add_argument("--pos-pred", nargs="+", default=["argmax", "centroid"])
    ap.add_argument("--beta", type=float, default=5.0, help="β for lse/softmax (mean↔max dial)")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", default=None, help="optional JSON of the full sweep")
    args = ap.parse_args(argv)

    queries = load_dump(args.diag_dir)
    nq = sum(1 for q in queries.values() if q["gt"] and q["cells"])
    print(f"[agg_sweep] {len(queries)} queries ({nq} with gt+cells) from {args.diag_dir} | beta={args.beta}", flush=True)
    print(f"{'level':8} {'pos':6} {'pred':8} | " + " ".join(f"t1@{int(t)}" for t in THRESH_M)
          + " | " + " ".join(f"t{args.topk}@{int(t)}" for t in THRESH_M))
    rows = []
    for la in args.level_aggs:
        for pa in args.pos_aggs:
            for pp in args.pos_pred:
                r = evaluate(queries, la, pa, pp, args.beta, topk=args.topk)
                rows.append({"level_agg": la, "pos_agg": pa, "pos_pred": pp, **r})
                t1 = " ".join(f"{r['top1'][int(t)]:.3f}" for t in THRESH_M)
                tk = " ".join(f"{r[f'top{args.topk}'][int(t)]:.3f}" for t in THRESH_M)
                print(f"{la:8} {pa:6} {pp:8} | {t1} | {tk}")
    if args.out:
        Path(args.out).expanduser().write_text(json.dumps(
            {"diag_dir": args.diag_dir, "beta": args.beta, "topk": args.topk, "rows": rows}), encoding="utf-8")
        print(f"[ok] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
