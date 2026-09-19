"""Build the KMZ → video → GT-frames JSON, recording each frame's matched video index.

For every GT KMZ under an S3 prefix: parse its placemarks (``<N>_<lat>,<lon>`` + chunk in the
description), pair each with ``GT_flat/<N>_*.jpg`` by N, group frames by their video chunk, download
each chunk once, and match every still against the decoded frames → the video frame index. Writes
one JSON grouped ``kmz → videos → frames`` to the output dir.

Run (from ~/work/geoLocalisation, cv2/numpy via uv; aws CLI for S3)::

    uv run --with opencv-python-headless --with numpy python -m kmz_frame_locator.cli \
      --kmz s3://geo-reference/gt/raw/kram/ \
      --gt-flat /home/ubuntu/work/gt_cramatorsc/GT_flat \
      --out /home/ubuntu/work/kram_frame_index.json \
      [--limit 5] [--preview-dir /home/ubuntu/work/frame_preview] [--min-ncc 0.9]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from kmz_video_audit.core import extract_kml_bytes
from .core import match_still, parse_placemarks

_GT_RE = re.compile(r"^(\d+)_")


def _aws_bytes(s3_url: str) -> bytes:
    return subprocess.run(["aws", "s3", "cp", s3_url, "-"], capture_output=True).stdout


def _list_kmz(prefix_url: str) -> list[str]:
    """List every .kmz key under an s3:// prefix (single .kmz url is returned as-is)."""
    if prefix_url.lower().endswith(".kmz"):
        return [prefix_url]
    m = re.match(r"s3://([^/]+)/(.*)$", prefix_url)
    if not m:
        raise SystemExit(f"bad --kmz S3 URL: {prefix_url}")
    bucket, prefix = m.group(1), m.group(2)
    out = subprocess.run(
        ["aws", "s3api", "list-objects-v2", "--bucket", bucket, "--prefix", prefix,
         "--query", "Contents[?ends_with(Key, `.kmz`)].Key", "--output", "text"],
        capture_output=True, text=True).stdout.split()
    return [f"s3://{bucket}/{k}" for k in out]


def _gt_index(gt_flat: Path) -> dict:
    """N -> path for every ``GT_flat/<N>_*.jpg``."""
    idx = {}
    for p in sorted(gt_flat.glob("*.jpg")):
        m = _GT_RE.match(p.name)
        if m:
            idx.setdefault(int(m.group(1)), p)
    return idx


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kmz", required=True, help="s3:// prefix of KMZ (or a single .kmz URL)")
    ap.add_argument("--gt-flat", required=True, help="local GT_flat dir (<N>_<lat>_<lon>.jpg)")
    ap.add_argument("--out", required=True, help="output JSON path (e.g. /home/ubuntu/work/....json)")
    ap.add_argument("--limit", type=int, default=0, help="cap frames matched (0=all; smoke run)")
    ap.add_argument("--min-ncc", type=float, default=0.9, help="flag/preview matches below this NCC")
    ap.add_argument("--preview-dir", default=None, help="save still|best-frame previews here (QC)")
    args = ap.parse_args(argv)

    import cv2
    import numpy as np

    gt_flat = Path(args.gt_flat).expanduser()
    if not gt_flat.is_dir():
        raise SystemExit(f"GT_flat not found: {gt_flat}")
    gt = _gt_index(gt_flat)
    print(f"[gt] {len(gt)} frames in {gt_flat}", flush=True)

    kmz_keys = _list_kmz(args.kmz)
    print(f"[kmz] {len(kmz_keys)} KMZ under {args.kmz}", flush=True)

    preview = Path(args.preview_dir).expanduser() if args.preview_dir else None
    if preview:
        preview.mkdir(parents=True, exist_ok=True)

    result = {"gt_flat": str(gt_flat), "kmz_prefix": args.kmz, "min_ncc": args.min_ncc, "kmz": []}
    n_done = 0
    low_conf = 0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for kurl in kmz_keys:
            try:
                pms = parse_placemarks(extract_kml_bytes(_aws_bytes(kurl)))
            except Exception as e:                             # noqa: BLE001 — skip a bad KMZ, keep going
                print(f"[kmz] {kurl} ERR {e}", flush=True)
                continue
            # group frames (that have a video AND a GT file) by their video URL
            by_video = defaultdict(list)
            for pm in pms:
                if pm.video_url and pm.n in gt:
                    by_video[pm.video_url].append(pm)
            if not by_video:
                continue
            kmz_rec = {"kmz": kurl, "videos": []}
            for vurl, frames in by_video.items():
                if args.limit and n_done >= args.limit:
                    break                                      # stop before downloading more chunks
                local = td / Path(vurl).name
                if not local.exists():
                    subprocess.run(["aws", "s3", "cp", vurl, str(local)],
                                   capture_output=True)
                vrec = {"video": vurl, "frames": []}
                for pm in frames:
                    if args.limit and n_done >= args.limit:
                        break
                    m = match_still(gt[pm.n], local) if local.exists() else None
                    if m is None:
                        vrec["frames"].append({"n": pm.n, "file": gt[pm.n].name,
                                               "lat": pm.lat, "lon": pm.lon, "error": "no match/video"})
                        continue
                    idx, fps, t_s, ncc, frame, nfr = m
                    rec = {"n": pm.n, "file": gt[pm.n].name, "lat": pm.lat, "lon": pm.lon,
                           "frame_index": idx, "fps": round(fps, 3), "time_s": round(t_s, 3),
                           "ncc": round(ncc, 4), "video_frames": nfr}
                    vrec["frames"].append(rec)
                    n_done += 1
                    if ncc < args.min_ncc:
                        low_conf += 1
                    if preview is not None:
                        h = 240
                        rz = lambda im: cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
                        tag = "LOW_" if ncc < args.min_ncc else ""
                        cv2.imwrite(str(preview / f"{tag}{pm.n}_idx{idx}_ncc{ncc:.3f}.jpg"),
                                    np.hstack([rz(cv2.imread(str(gt[pm.n]))), rz(frame)]))
                if local.exists():
                    local.unlink()                             # free disk — chunks are re-downloadable
                kmz_rec["videos"].append(vrec)
            result["kmz"].append(kmz_rec)
            nfr = sum(len(v["frames"]) for v in kmz_rec["videos"])
            print(f"[kmz] {kurl}: {len(kmz_rec['videos'])} videos, {nfr} frames", flush=True)
            if args.limit and n_done >= args.limit:
                break

    result["n_frames_matched"] = n_done
    result["n_low_confidence"] = low_conf
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] {n_done} frames matched ({low_conf} below NCC {args.min_ncc}) -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
