"""Clip aggregation over the GPU FINE per-frame result → top-10 cells + KMZ + distR.

Input = the JSON written by ``pyramid_newformat.locate`` (run with ``--topk 100`` so every frame keeps
up to 100 fine-scored cells) + the ``clips_manifest.json`` from :mod:`clip_search.sample`. For each clip
we group its frames, then sweep the per-frame cut ``top_n ∈ {10,20,30,40,50}`` and aggregate each cell
TWO ways across the clip's frames:

  * **sum**   — Σ of the cell's fine ``cell_score`` over the frames where it is in that frame's top-N;
  * **votes** — how many frames have the cell in their top-N (score-agnostic).

Each clip → the top-10 cells per (agg, N); a cell's predicted lat/lon is taken from the frame where it
scored highest. We write one KMZ per (agg, N) (all clips) and a summary JSON with top-1 / top-10
``distR@{250,500,1000}m`` for every (agg, N) — the sweep the user asked for.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

TOPN_SWEEP = (10, 20, 30, 40, 50)
TOPK = 10
THRESH_M = (250.0, 500.0, 1000.0)


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    x = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(x)))


def aggregate_clip(frames_topk: list[list[dict]], top_n: int) -> dict:
    """frames_topk = per-frame ranked cell lists (each sorted desc by cell_score, as locate emits).

    → {"sum": [cells...], "votes": [cells...]} each a ranked list of
    {cell_id, score_sum, votes, lat, lon, level_m}. ``sum`` ranks by Σscore, ``votes`` by (#frames, Σscore).
    A cell's lat/lon = the position from the frame where it scored highest.
    """
    acc: dict[str, dict] = {}
    for cells in frames_topk:
        for c in cells[:top_n]:
            cid = c["cell_id"]
            sc = float(c.get("cell_score", 0.0))
            a = acc.get(cid)
            if a is None:
                acc[cid] = {"cell_id": cid, "score_sum": sc, "votes": 1,
                            "lat": c["lat"], "lon": c["lon"], "level_m": c.get("level_m"),
                            "_best": sc}
            else:
                a["score_sum"] += sc
                a["votes"] += 1
                if sc > a["_best"]:                       # keep the strongest frame's position
                    a["_best"] = sc; a["lat"] = c["lat"]; a["lon"] = c["lon"]; a["level_m"] = c.get("level_m")
    cells = list(acc.values())
    for c in cells:
        c.pop("_best", None)
    by_sum = sorted(cells, key=lambda c: c["score_sum"], reverse=True)
    by_votes = sorted(cells, key=lambda c: (c["votes"], c["score_sum"]), reverse=True)
    return {"sum": by_sum, "votes": by_votes}


def _ranked_entry(clip_id, gt, ranked, topk):
    """A patch_rerank.kmz entry: GT + rank-1 pred + the rest of the top-K as a ranked list."""
    top = ranked[:topk]
    for i, c in enumerate(top):
        c["rank"] = i + 1
        c["dist_m"] = haversine_m(gt[0], gt[1], c["lat"], c["lon"]) if gt else None
    pred = (top[0]["lat"], top[0]["lon"]) if top else gt
    d = top[0]["dist_m"] if (top and gt) else float("nan")
    return {"name": clip_id, "gt": tuple(gt) if gt else (float("nan"), float("nan")),
            "pred": pred, "dist_m": d if d is not None else float("nan"), "ranked": top}


def run(locate_json: str, manifest_json: str, out_dir: str, *, topn_sweep=TOPN_SWEEP, topk=TOPK,
        write_kmz=True, log=print) -> dict:
    loc = json.loads(Path(locate_json).expanduser().read_text())
    man = json.loads(Path(manifest_json).expanduser().read_text())
    per_q = loc.get("per_query", {})
    out_dir = Path(out_dir).expanduser(); out_dir.mkdir(parents=True, exist_ok=True)

    clips = man["clips"]
    results: dict = {"meta": {"locate": locate_json, "manifest": manifest_json,
                              "topn_sweep": list(topn_sweep), "topk": topk,
                              "n_clips": len(clips)}, "per_clip": {}, "summary": {}}
    # per (agg, N): collect top-1 and top-K distances across clips with GT
    dist1 = {(agg, n): [] for agg in ("sum", "votes") for n in topn_sweep}
    distk = {(agg, n): [] for agg in ("sum", "votes") for n in topn_sweep}
    kmz_entries = {(agg, n): [] for agg in ("sum", "votes") for n in topn_sweep}

    n_no_frames = 0
    for clip in clips:
        cid = clip["clip_id"]
        gt = [clip["gt_lat"], clip["gt_lon"]]
        frames_topk = [per_q[f["qid"]]["topk"] for f in clip.get("frames", []) if f["qid"] in per_q]
        n_hit = len(frames_topk)
        rec = {"n_frames": len(clip.get("frames", [])), "n_frames_scored": n_hit, "gt": gt, "by_n": {}}
        if n_hit == 0:
            n_no_frames += 1
            results["per_clip"][cid] = rec
            continue
        for n in topn_sweep:
            agg = aggregate_clip(frames_topk, n)
            rec["by_n"][str(n)] = {}
            for name in ("sum", "votes"):
                ranked = [dict(c) for c in agg[name][:topk]]
                for i, c in enumerate(ranked):
                    c["rank"] = i + 1
                    c["dist_m"] = haversine_m(gt[0], gt[1], c["lat"], c["lon"])
                rec["by_n"][str(n)][name] = ranked
                if ranked:
                    dist1[(name, n)].append(ranked[0]["dist_m"])
                    distk[(name, n)].append(min(c["dist_m"] for c in ranked))
                    kmz_entries[(name, n)].append(_ranked_entry(cid, gt, [dict(c) for c in ranked], topk))
        results["per_clip"][cid] = rec

    import numpy as np
    for (agg, n) in dist1:
        d1 = np.asarray(dist1[(agg, n)]); dk = np.asarray(distk[(agg, n)])
        key = f"{agg}_top{n}"
        results["summary"][key] = {
            "n_eval": int(d1.size),
            "top1": {f"distR@{int(t)}m": (float((d1 <= t).mean()) if d1.size else None) for t in THRESH_M},
            f"top{topk}": {f"distR@{int(t)}m": (float((dk <= t).mean()) if dk.size else None) for t in THRESH_M},
        }

    (out_dir / "clip_search_results.json").write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

    if write_kmz:
        import sys
        repo = Path(__file__).resolve().parents[1]
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from patch_rerank.kmz import write_kmz as _write_kmz
        kdir = out_dir / "kmz"; kdir.mkdir(exist_ok=True)
        for (agg, n), entries in kmz_entries.items():
            if entries:
                _write_kmz(kdir / f"{agg}_top{n}.kmz", entries)

    log(f"[ok] {len(clips)} clips ({n_no_frames} with no scored frames) -> {out_dir}/clip_search_results.json")
    # compact table to stdout
    log(f"{'agg_topN':16} {'n':>4} | " + " ".join(f"t1@{int(t)}" for t in THRESH_M)
        + " | " + " ".join(f"t{topk}@{int(t)}" for t in THRESH_M))
    for agg in ("sum", "votes"):
        for n in topn_sweep:
            s = results["summary"][f"{agg}_top{n}"]
            t1 = " ".join(f"{(s['top1'][f'distR@{int(t)}m'] or 0):.3f}" for t in THRESH_M)
            tk = " ".join(f"{(s[f'top{topk}'][f'distR@{int(t)}m'] or 0):.3f}" for t in THRESH_M)
            log(f"{agg+'_top'+str(n):16} {s['n_eval']:>4} | {t1} | {tk}")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--locate", required=True, help="pyramid_newformat.locate output JSON (run with --topk 100)")
    ap.add_argument("--manifest", required=True, help="clips_manifest.json from clip_search.sample")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--topn", type=int, nargs="+", default=list(TOPN_SWEEP), help="per-frame cut sweep")
    ap.add_argument("--topk", type=int, default=TOPK, help="final cells per clip (→ KMZ)")
    ap.add_argument("--no-kmz", dest="write_kmz", action="store_false")
    args = ap.parse_args(argv)
    run(args.locate, args.manifest, args.out_dir, topn_sweep=tuple(args.topn), topk=args.topk,
        write_kmz=args.write_kmz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
