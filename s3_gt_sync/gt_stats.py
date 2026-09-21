"""GT statistics: distances between real GT anchors and how many frames can be
extracted between them IF the linear interpolation (see ``interpolate_gt.py``) is
treated as truth. JSON-only post-process over the subvideo_index / interpolate_gt
outputs — no video download, no disk artifacts unless you ask for --csv/--summary.

The unit of interpolation is a *sub-video* (a continuous good/usable run); linear
lat/lon ramps run only between consecutive GT anchors *within* one sub-video (never
across a bad-frame gap), exactly like ``interpolate_gt.interpolate_subvideo``. So
the extractable count here equals what interpolate_gt would (or did) generate.

    python -m s3_gt_sync.gt_stats --json-dir /home/ubuntu/work/gt_lyman
    python -m s3_gt_sync.gt_stats --json /home/ubuntu/work/gt_lyman/lyman_3.0.json
    # every city at once + machine-readable outputs:
    python -m s3_gt_sync.gt_stats --json-dir /home/ubuntu/work --glob "gt_*/*.json" \
        --csv gt_pairs.csv --summary gt_summary.json

Definitions
    anchor            real NCC-matched GT frame (lat/lon + frame index)
    pair              two consecutive anchors WITHIN one sub-video
    n_between         frames strictly between a pair = extractable interpolated GT
    extractable_total anchors + sum(n_between) = all usable "GT" positions
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import statistics as st
from pathlib import Path

EARTH_R_M = 6371000.0


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in metres between two (lat, lon) points."""
    la1, lo1, la2, lo2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    h = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 2 * EARTH_R_M * math.asin(math.sqrt(h))


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {}
    return {"min": round(min(xs), 2), "median": round(st.median(xs), 2),
            "mean": round(sum(xs) / len(xs), 2), "max": round(max(xs), 2)}


def anchor_pairs(rec: dict) -> list[dict]:
    """One row per consecutive-anchor pair within a sub-video, with the distance
    and how many frames sit between them (extractable if linear == truth)."""
    city, version = rec.get("city", "?"), rec.get("version", "?")
    rows: list[dict] = []
    for v in rec.get("videos", []):
        fps = v.get("fps") or None
        vname = str(v.get("video", "")).split("/")[-1]
        for sv in v.get("sub_videos", []):
            gts = sorted((f for f in sv.get("gt_frames", [])
                          if "frame_index_in_subvideo" in f),
                         key=lambda f: f["frame_index_in_subvideo"])
            for a, b in zip(gts, gts[1:]):
                span = b["frame_index_in_subvideo"] - a["frame_index_in_subvideo"]
                if span <= 0:
                    continue
                dm = haversine_m((a["lat"], a["lon"]), (b["lat"], b["lon"]))
                sec = span / fps if fps else None
                rows.append({
                    "city": city, "version": version, "video": vname,
                    "sub_index": sv.get("sub_index"),
                    "anchor_a_n": a.get("n"), "anchor_b_n": b.get("n"),
                    "dist_m": round(dm, 1), "frame_gap": span,
                    "sec_gap": round(sec, 2) if sec else None,
                    "speed_mps": round(dm / sec, 2) if sec else None,
                    "n_between": span - 1,
                    "m_per_frame": round(dm / span, 3) if span else None,
                })
    return rows


def summarize(rec: dict, pairs: list[dict] | None = None) -> dict:
    """Per-json roll-up. ``pairs`` may be passed to avoid recomputation."""
    if pairs is None:
        pairs = anchor_pairs(rec)
    n_anchors = 0
    single_anchor_videos = 0
    subs_no_anchor = 0
    for v in rec.get("videos", []):
        v_anchor = 0
        for sv in v.get("sub_videos", []):
            g = sv.get("gt_frames", [])
            n_anchors += len(g)
            v_anchor += len(g)
            if not g:
                subs_no_anchor += 1
        if v_anchor == 1:
            single_anchor_videos += 1

    n_between = sum(p["n_between"] for p in pairs)
    dists = [p["dist_m"] for p in pairs]
    speeds = [p["speed_mps"] for p in pairs if p["speed_mps"] is not None]
    mpf = [p["m_per_frame"] for p in pairs if p["m_per_frame"] is not None]
    return {
        "city": rec.get("city", "?"), "version": rec.get("version", "?"),
        "n_videos": len(rec.get("videos", [])), "n_anchors": n_anchors,
        "n_interpolatable_pairs": len(pairs),
        "n_between_total_extractable": n_between,
        "extractable_total_with_anchors": n_anchors + n_between,
        "declared_n_interpolated_total": rec.get("n_interpolated_total"),
        "videos_single_anchor": single_anchor_videos,
        "sub_videos_without_anchor": subs_no_anchor,
        "anchor_gap_distance_m": _stats(dists),
        "implied_speed_mps": _stats(speeds),
        "ground_sampling_m_per_frame": _stats(mpf),
        "total_interpolated_track_m": round(sum(dists), 1),
    }


def analyze_file(path: Path) -> tuple[dict, list[dict]]:
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    pairs = anchor_pairs(rec)
    summary = summarize(rec, pairs)
    summary["file"] = str(path)
    return summary, pairs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=None, help="a single GT json")
    ap.add_argument("--json-dir", default=None, help="dir of GT jsons")
    ap.add_argument("--glob", default="*.json", help="glob under --json-dir")
    ap.add_argument("--csv", default=None, help="write per-pair rows here")
    ap.add_argument("--summary", default=None, help="write per-json summaries json here")
    args = ap.parse_args(argv)

    if args.json:
        paths = [Path(args.json).expanduser()]
    elif args.json_dir:
        paths = [Path(p) for p in sorted(
            glob.glob(str(Path(args.json_dir).expanduser() / args.glob)))]
    else:
        raise SystemExit("need --json or --json-dir")

    all_pairs: list[dict] = []
    summaries: list[dict] = []
    for p in paths:
        try:
            s, pr = analyze_file(p)
        except Exception as e:  # noqa: BLE001 — skip a malformed json, keep going
            print(f"[skip] {p.name}: {e}")
            continue
        summaries.append(s)
        all_pairs.extend(pr)
        print(f"\n=== {s['city']} v{s['version']}  ({p.name}) ===")
        print(f"  videos={s['n_videos']}  anchors={s['n_anchors']}  "
              f"pairs={s['n_interpolatable_pairs']}  single-anchor videos={s['videos_single_anchor']}")
        print(f"  extractable between anchors (linear=truth): {s['n_between_total_extractable']}"
              f"  (+anchors = {s['extractable_total_with_anchors']})")
        print(f"  anchor-gap dist m: {s['anchor_gap_distance_m']}")
        print(f"  implied speed m/s: {s['implied_speed_mps']}")
        print(f"  ground sampling m/frame: {s['ground_sampling_m_per_frame']}")

    if args.csv and all_pairs:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_pairs[0].keys()))
            w.writeheader()
            w.writerows(all_pairs)
        print(f"\nwrote {args.csv} ({len(all_pairs)} pairs)")
    if args.summary:
        Path(args.summary).write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"wrote {args.summary}")

    tot_anc = sum(s["n_anchors"] for s in summaries)
    tot_ext = sum(s["n_between_total_extractable"] for s in summaries)
    print(f"\n== TOTAL over {len(summaries)} json: anchors={tot_anc}  "
          f"extractable_between={tot_ext}  grand_total={tot_anc + tot_ext} ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
