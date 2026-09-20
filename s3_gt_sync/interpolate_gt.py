"""Densify GT: linearly interpolate lat/lon for EVERY frame between adjacent GT points, WITHIN a
sub-video (a continuous good/usable run). JSON-only post-process over the subvideo_index outputs —
no frame extraction, no video download, no disk artifacts.

Assumption (per user): intra-sub-video motion over a short run is ~uniform, so a linear lat/lon ramp
by frame index is a reasonable label. Every generated point carries ``interpolated: true`` +
provenance (the GT pair it sits between and the blend factor t), so it is never confused with a real
GT label. Interpolation is done ONLY within a sub-video (never across a bad-frame gap).

    python -m s3_gt_sync.interpolate_gt --json-dir /home/ubuntu/work/gt_cramatorsc
    python -m s3_gt_sync.interpolate_gt --json /home/ubuntu/work/gt_cramatorsc/kram_2.7.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def interpolate_subvideo(sub: dict, fps: float | None) -> list[dict]:
    """Return interpolated frame dicts for every frame strictly between consecutive GT points.

    Frames are contiguous inside a sub-video, so ``frame_index_in_chunk`` increments 1:1 with
    ``frame_index_in_subvideo`` — we carry both."""
    gts = sorted((f for f in sub.get("gt_frames", []) if "frame_index_in_subvideo" in f),
                 key=lambda f: f["frame_index_in_subvideo"])
    out: list[dict] = []
    for fa, fb in zip(gts, gts[1:]):
        a, b = fa["frame_index_in_subvideo"], fb["frame_index_in_subvideo"]
        span = b - a
        if span <= 1:
            continue
        ca = fa["frame_index_in_chunk"]
        for k in range(1, span):
            t = k / span
            idx = a + k
            chunk_idx = ca + k
            out.append({
                "frame_index_in_subvideo": idx,
                "frame_index_in_chunk": chunk_idx,
                "lat": round(fa["lat"] + t * (fb["lat"] - fa["lat"]), 10),
                "lon": round(fa["lon"] + t * (fb["lon"] - fa["lon"]), 10),
                "time_s": round(chunk_idx / fps, 3) if fps else None,
                "interpolated": True,
                "between": [fa["n"], fb["n"]],
                "t": round(t, 4),
            })
    return out


def process_json(path: Path) -> int:
    rec = json.loads(path.read_text(encoding="utf-8"))
    total = 0
    for v in rec.get("videos", []):
        fps = v.get("fps")
        for sub in v.get("sub_videos", []):
            interp = interpolate_subvideo(sub, fps)
            sub["interpolated_frames"] = interp
            sub["n_interpolated"] = len(interp)
            total += len(interp)
    rec["n_interpolated_total"] = total
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-dir", default=None, help="dir of subvideo_index JSONs (augmented in place)")
    ap.add_argument("--json", default=None, help="a single subvideo_index JSON")
    ap.add_argument("--glob", default="*.json")
    args = ap.parse_args(argv)

    if args.json:
        paths = [Path(args.json).expanduser()]
    elif args.json_dir:
        paths = [Path(p) for p in sorted(glob.glob(str(Path(args.json_dir).expanduser() / args.glob)))]
    else:
        raise SystemExit("need --json or --json-dir")
    grand = 0
    for p in paths:
        try:
            n = process_json(p)
        except Exception as e:  # noqa: BLE001 — skip a malformed json, keep going
            print(f"[skip] {p.name}: {e}")
            continue
        grand += n
        print(f"[ok] {p.name}: +{n} interpolated frames")
    print(f"== total interpolated: {grand} across {len(paths)} JSON ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
