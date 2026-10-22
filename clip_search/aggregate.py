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


METHODS = ("sum", "votes", "wsum", "max", "gated")


def aggregate_clip(frames_topk: list[list[dict]], top_n: int, *, gate_frac: float = 0.9) -> dict:
    """frames_topk = per-frame ranked cell lists (each sorted desc by cell_score).

    Returns a ranked cell list per method (all training-free; a frame's CONFIDENCE = its max cell_score
    in top-N — the coarse stand-in for a geometric inlier-ratio weight):
      * **sum**   — Σ cell_score over frames having the cell in top-N;
      * **votes** — #frames with the cell in top-N (score-agnostic);
      * **wsum**  — Σ (conf_f · cell_score): down-weights low-confidence (off-GT) frames [idea #1];
      * **max**   — max cell_score across frames (strongest frame wins, no dilution) [coarse #3];
      * **gated** — Σ cell_score but only over frames whose conf_f ≥ gate_frac·max_f(conf) — a
                    variable-shape window kept by score [idea #2].
    Each cell carries lat/lon from the frame where it scored highest.
    """
    frames = [cells[:top_n] for cells in frames_topk]
    conf = [max((float(c.get("cell_score", 0.0)) for c in cells), default=0.0) for cells in frames]
    cmax = max(conf, default=0.0)
    keep = [cf >= gate_frac * cmax for cf in conf] if cmax > 0 else [True] * len(frames)

    acc: dict[str, dict] = {}
    for cells, cf, gk in zip(frames, conf, keep):
        for c in cells:
            cid = c["cell_id"]; sc = float(c.get("cell_score", 0.0))
            a = acc.get(cid)
            if a is None:
                a = acc[cid] = {"cell_id": cid, "score_sum": 0.0, "votes": 0, "wsum": 0.0,
                                "score_max": -1e30, "gated_sum": 0.0,
                                "lat": c["lat"], "lon": c["lon"], "level_m": c.get("level_m"), "_best": -1e30}
            a["score_sum"] += sc
            a["votes"] += 1
            a["wsum"] += cf * sc
            a["score_max"] = max(a["score_max"], sc)
            if gk:
                a["gated_sum"] += sc
            if sc > a["_best"]:                           # strongest frame's position
                a["_best"] = sc; a["lat"] = c["lat"]; a["lon"] = c["lon"]; a["level_m"] = c.get("level_m")
    cells = list(acc.values())
    for c in cells:
        c.pop("_best", None)
    return {
        "sum":   sorted(cells, key=lambda c: c["score_sum"], reverse=True),
        "votes": sorted(cells, key=lambda c: (c["votes"], c["score_sum"]), reverse=True),
        "wsum":  sorted(cells, key=lambda c: c["wsum"], reverse=True),
        "max":   sorted(cells, key=lambda c: c["score_max"], reverse=True),
        "gated": sorted([c for c in cells if c["gated_sum"] > 0], key=lambda c: c["gated_sum"], reverse=True),
    }


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
        write_kmz=True, gt_frame_only=False, gate_frac=0.9, methods=METHODS, log=print) -> dict:
    loc = json.loads(Path(locate_json).expanduser().read_text())
    man = json.loads(Path(manifest_json).expanduser().read_text())
    per_q = loc.get("per_query", {})
    out_dir = Path(out_dir).expanduser(); out_dir.mkdir(parents=True, exist_ok=True)

    clips = man["clips"]
    results: dict = {"meta": {"locate": locate_json, "manifest": manifest_json,
                              "topn_sweep": list(topn_sweep), "topk": topk,
                              "gt_frame_only": gt_frame_only, "gate_frac": gate_frac,
                              "methods": list(methods),
                              "n_clips": len(clips)}, "per_clip": {}, "summary": {}}
    # per (method, N): collect top-1 and top-K distances across clips with GT
    dist1 = {(agg, n): [] for agg in methods for n in topn_sweep}
    distk = {(agg, n): [] for agg in methods for n in topn_sweep}
    kmz_entries = {(agg, n): [] for agg in methods for n in topn_sweep}

    n_no_frames = 0
    for clip in clips:
        cid = clip["clip_id"]
        gt = [clip["gt_lat"], clip["gt_lon"]]
        frames = clip.get("frames", [])
        if gt_frame_only:                              # baseline: only the central GT frame (1-frame search)
            gfi = clip.get("gt_frame_index")
            frames = [f for f in frames if f.get("frame_index") == gfi] or frames[:1]
        frames_topk = [per_q[f["qid"]]["topk"] for f in frames if f["qid"] in per_q]
        n_hit = len(frames_topk)
        rec = {"n_frames": len(clip.get("frames", [])), "n_frames_scored": n_hit, "gt": gt, "by_n": {}}
        if n_hit == 0:
            n_no_frames += 1
            results["per_clip"][cid] = rec
            continue
        for n in topn_sweep:
            agg = aggregate_clip(frames_topk, n, gate_frac=gate_frac)
            rec["by_n"][str(n)] = {}
            for name in methods:
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
    for agg in methods:
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
    ap.add_argument("--gt-frame-only", action="store_true",
                    help="baseline: localize using ONLY each clip's central GT frame (single-frame "
                         "search) instead of aggregating the clip's neighbours")
    ap.add_argument("--gate", type=float, default=0.9,
                    help="gated method: keep frames with confidence >= gate*max_conf (0..1)")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS),
                    help="which aggregation methods to evaluate")
    args = ap.parse_args(argv)
    run(args.locate, args.manifest, args.out_dir, topn_sweep=tuple(args.topn), topk=args.topk,
        write_kmz=args.write_kmz, gt_frame_only=args.gt_frame_only, gate_frac=args.gate,
        methods=tuple(args.methods))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
