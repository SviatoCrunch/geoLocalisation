"""How often must a mask be drawn? Data-driven keyframe interval from the json.

For each sub-video (a continuous, valid propagation interval per subvideo_index)
we greedily place keyframes: start one at the sub-video start, and open a NEW one
whenever the ORB+RANSAC overlap (inlier ratio) to the current keyframe drops below
--min-overlap. The count answers "1 mask per N frames" per sub-video / video /
city, adapting to real motion (fast flight -> denser). Sub-video boundaries always
force a fresh keyframe (homography invalid across cuts/dropouts).

    python -m s3_gt_sync.keyframe_interval --json /home/ubuntu/work/gt_cramatorsc/kram_2.7.json \
        --min-overlap 0.5 --stride 3 --out kf_kram_2.7.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2

from .subvideo_index import overlap_ratio


# ------------------------- pure core (testable) -------------------------

def place_keyframes(indices, ratio_of, min_overlap: float) -> list[int]:
    """Greedy keyframe placement over sampled frame indices of ONE sub-video.

    A new keyframe opens when overlap to the CURRENT keyframe drops below
    min_overlap. ratio_of(key, t) -> inlier ratio. Returns keyframe indices."""
    if not indices:
        return []
    keys = [indices[0]]
    key = indices[0]
    for t in indices[1:]:
        if ratio_of(key, t) < min_overlap:
            keys.append(t)
            key = t
    return keys


def _bad_set(vrec: dict) -> set:
    bad = set()
    for s, e in vrec.get("bad_frames", []):
        bad.update(range(s, e + 1))
    return bad


def _sample_indices(sub, bad, stride) -> list[int]:
    a, b = sub["start_frame"], sub["end_frame"]
    return [i for i in range(a, b + 1, stride) if i not in bad]


# ------------------------------- I/O runner -------------------------------

def _descriptors(video_path, wanted: set, orb, downscale):
    """Sequential decode; ORB (kp, desc) only for wanted frame indices."""
    cap = cv2.VideoCapture(str(video_path))
    out: dict[int, tuple] = {}
    idx = -1
    maxw = max(wanted) if wanted else -1
    while True:
        ok, frame = cap.read()
        if not ok or idx >= maxw:
            break
        idx += 1
        if idx not in wanted:
            continue
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if downscale > 1:
            g = cv2.resize(g, (g.shape[1] // downscale, g.shape[0] // downscale))
        out[idx] = orb.detectAndCompute(g, None)
    cap.release()
    return out


def analyze(json_path, min_overlap=0.5, stride=3, orb_features=1200, downscale=2,
            ransac_thr=4.0, log=print):
    rec = json.loads(Path(json_path).read_text(encoding="utf-8"))
    fps_default = 30.0
    orb = cv2.ORB_create(nfeatures=orb_features)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    report = {"json": str(json_path), "min_overlap": min_overlap, "stride": stride,
              "videos": []}
    tot_frames = tot_keys = 0

    with tempfile.TemporaryDirectory() as _td:
        td = Path(_td)
        for v in rec.get("videos", []):
            subs = v.get("sub_videos", [])
            if not subs:
                continue
            url = v["video"]; fps = v.get("fps", fps_default)
            local = td / Path(url).name
            if not local.exists():
                subprocess.run(["aws", "s3", "cp", url, str(local)], capture_output=True)
            if not local.exists():
                continue
            bad = _bad_set(v)
            wanted = set()
            sampled = {}
            for sub in subs:
                s = _sample_indices(sub, bad, stride)
                sampled[sub["sub_index"]] = s
                wanted.update(s)
            desc = _descriptors(local, wanted, orb, downscale)

            def ratio_of(a, b):
                da, db = desc.get(a), desc.get(b)
                if not da or not db:
                    return 0.0
                return overlap_ratio(da[1], da[0], db[1], db[0], matcher, ransac_thr)

            vout = {"video": Path(url).name, "fps": fps, "sub_videos": []}
            for sub in subs:
                idxs = sampled[sub["sub_index"]]
                keys = place_keyframes(idxs, ratio_of, min_overlap)
                span = sub["end_frame"] - sub["start_frame"] + 1
                nk = max(len(keys), 1)
                vout["sub_videos"].append({
                    "sub_index": sub["sub_index"], "span_frames": span,
                    "sampled": len(idxs), "keyframes_needed": nk,
                    "avg_interval_frames": round(span / nk, 1),
                    "avg_interval_s": round(span / nk / fps, 2) if fps else None,
                    "boundary": sub.get("boundary")})
                tot_frames += span; tot_keys += nk
            report["videos"].append(vout)
            log(f"  {Path(url).name}: {len(subs)} subs -> "
                f"{sum(s['keyframes_needed'] for s in vout['sub_videos'])} masks")

    report["totals"] = {
        "total_subvideo_frames": tot_frames,
        "total_masks_needed": tot_keys,
        "one_mask_per_frames": round(tot_frames / tot_keys, 1) if tot_keys else None,
    }
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True)
    ap.add_argument("--min-overlap", type=float, default=0.5,
                    help="open a new keyframe when inlier ratio to current key drops below this")
    ap.add_argument("--stride", type=int, default=3, help="sample every k-th frame (speed)")
    ap.add_argument("--orb-features", type=int, default=1200)
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--ransac-thr", type=float, default=4.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    rep = analyze(args.json, args.min_overlap, args.stride, args.orb_features,
                  args.downscale, args.ransac_thr)
    t = rep["totals"]
    print(f"\n== {Path(args.json).name}: {t['total_masks_needed']} masks for "
          f"{t['total_subvideo_frames']} frames  = 1 mask / {t['one_mask_per_frames']} frames "
          f"(overlap>= {args.min_overlap}) ==")
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
