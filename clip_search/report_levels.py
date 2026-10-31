"""Join the per-frame FINE result + per-video aggregation into one readable report.

Pure stdlib (no numpy/torch/cv2). Combines:
  * ``pyramid_newformat.locate`` output (``--dump-levels``): per-frame top-K cells with the full
    per-level breakdown (``level_scores`` / ``position_scores`` / ``winning_position_idx``);
  * ``clip_search.aggregate`` output (``clip_search_results.json``): per-video aggregated ranking +
    ``distR`` summary;
  * (optional) the ``clips_manifest.json`` to map each video → its frames → their ``qid``.

Writes:
  * ``report.csv``  — one row per video: GT, predicted (rank-1 of the chosen agg×N), ``dist_m``,
    hit@{250,500,1000} m, and the dominant pyramid level;
  * ``report.json`` — per video: GT, prediction, distR, a level histogram, and (if the manifest is
    given) each frame's detailed top-K (cell_id, cell_score, lat, lon, level_m, level_scores).

    python -m clip_search.report_levels --locate locate_k100.json \
        --results out/clip_search_results.json --manifest clips_manifest.json --out-dir out
"""
from __future__ import annotations

import argparse
import csv as _csv
import json
from collections import Counter
from pathlib import Path

THRESH_M = (250.0, 500.0, 1000.0)


def _rank1(results: dict, clip_id: str, agg: str, n: int):
    """Rank-1 aggregated cell for a clip at (agg, N), or None."""
    rec = results.get("per_clip", {}).get(clip_id, {})
    ranked = rec.get("by_n", {}).get(str(n), {}).get(agg, [])
    return ranked[0] if ranked else None


def _cell_detail(cell: dict, levels_m) -> dict:
    """One top-K cell with its per-level scores labelled by level (``{level_m: score}``)."""
    d = {k: cell.get(k) for k in
         ("cell_id", "cell_score", "lat", "lon", "level_m", "dist_m", "winning_position_idx")}
    ls = cell.get("level_scores")
    if ls is not None:
        if levels_m is not None and len(ls) == len(levels_m):
            d["levels"] = {str(int(lm)): float(sc) for lm, sc in zip(levels_m, ls)}
        else:
            d["levels"] = {str(i): float(sc) for i, sc in enumerate(ls)}
    return d


def build_report(locate: dict, results: dict, manifest: dict | None, *, agg: str = "sum",
                 n: int = 10, topk_detail: int = 100) -> dict:
    """Assemble the per-video report dict (see module docstring)."""
    per_q = locate.get("per_query", {})
    levels_m = locate.get("meta", {}).get("levels_m")
    clips_by_id = {c["clip_id"]: c for c in (manifest or {}).get("clips", [])}

    out: dict = {"meta": {"agg": agg, "topn": n, "levels_m": levels_m,
                          "summary": results.get("summary", {}).get(f"{agg}_top{n}")},
                 "videos": []}
    clip_ids = list(results.get("per_clip", {}).keys()) or list(clips_by_id.keys())
    for cid in clip_ids:
        prec = results.get("per_clip", {}).get(cid, {})
        gt = prec.get("gt")
        r1 = _rank1(results, cid, agg, n)
        clip = clips_by_id.get(cid, {})

        frames_detail = []
        level_hist = Counter()
        for fm in clip.get("frames", []):
            tk = per_q.get(fm["qid"], {}).get("topk", [])
            if tk:
                level_hist[tk[0].get("level_m")] += 1          # each frame's top-1 level
            frames_detail.append({
                "qid": fm["qid"], "frame_index": fm.get("frame_index"),
                "file": fm.get("file"), "quality": fm.get("quality"),
                "n_cells": len(tk),
                "topk": [_cell_detail(cell, levels_m) for cell in tk[:topk_detail]],
            })

        out["videos"].append({
            "clip_id": cid, "video_id": clip.get("video_id"),
            "gt": gt, "pred": ({"lat": r1["lat"], "lon": r1["lon"], "dist_m": r1.get("dist_m"),
                                "level_m": r1.get("level_m"), "cell_id": r1["cell_id"]} if r1 else None),
            "n_frames_scored": prec.get("n_frames_scored"),
            "level_hist": {str(k): v for k, v in sorted(level_hist.items(), key=lambda kv: -kv[1])},
            "frames": frames_detail,
        })
    return out


def _csv_rows(report: dict):
    for v in report["videos"]:
        gt = v.get("gt") or [None, None]
        pred = v.get("pred") or {}
        d = pred.get("dist_m")
        hist = v.get("level_hist") or {}
        best_level = next(iter(hist), pred.get("level_m"))
        yield {
            "clip_id": v["clip_id"], "video_id": v.get("video_id"),
            "n_frames_scored": v.get("n_frames_scored"),
            "gt_lat": gt[0], "gt_lon": gt[1],
            "pred_lat": pred.get("lat"), "pred_lon": pred.get("lon"), "dist_m": d,
            "hit250": None if d is None else int(d <= 250.0),
            "hit500": None if d is None else int(d <= 500.0),
            "hit1000": None if d is None else int(d <= 1000.0),
            "best_level_m": best_level,
        }


def run(locate_json, results_json, out_dir, *, manifest_json=None, agg="sum", n=10,
        topk_detail=100, log=print) -> dict:
    locate = json.loads(Path(locate_json).expanduser().read_text(encoding="utf-8"))
    results = json.loads(Path(results_json).expanduser().read_text(encoding="utf-8"))
    manifest = (json.loads(Path(manifest_json).expanduser().read_text(encoding="utf-8"))
                if manifest_json else None)
    report = build_report(locate, results, manifest, agg=agg, n=n, topk_detail=topk_detail)

    out_dir = Path(out_dir).expanduser(); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    rows = list(_csv_rows(report))
    cols = ["clip_id", "video_id", "n_frames_scored", "gt_lat", "gt_lon", "pred_lat", "pred_lon",
            "dist_m", "hit250", "hit500", "hit1000", "best_level_m"]
    with (out_dir / "report.csv").open("w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    hits = [r["hit1000"] for r in rows if r["hit1000"] is not None]
    log(f"[ok] {len(rows)} videos -> {out_dir}/report.csv + report.json "
        f"(agg={agg} top{n}; {sum(hits)}/{len(hits)} hit@1000m)" if hits else
        f"[ok] {len(rows)} videos -> {out_dir}/report.csv + report.json (no GT distances)")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--locate", required=True, help="pyramid_newformat.locate JSON (ideally --dump-levels)")
    ap.add_argument("--results", required=True, help="clip_search.aggregate clip_search_results.json")
    ap.add_argument("--manifest", default=None, help="clips_manifest.json (for per-frame detail + video_id)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--agg", default="sum", help="aggregation method to pick the prediction from")
    ap.add_argument("--topn", type=int, default=10, help="per-frame cut N (must be in the aggregate sweep)")
    ap.add_argument("--topk-detail", type=int, default=100, help="cells/frame kept in report.json")
    args = ap.parse_args(argv)
    run(args.locate, args.results, args.out_dir, manifest_json=args.manifest, agg=args.agg,
        n=args.topn, topk_detail=args.topk_detail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
