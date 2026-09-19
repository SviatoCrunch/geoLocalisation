"""Score an FPV video for corruption, list bad segments, and dump a visual QC montage.

Decodes the video, computes per-frame metrics (fpv_video_qc.core), flags bad frames
(``line_noise > --line-thr`` OR ``tdiff > --tdiff-thr``), groups them into segments, writes a JSON
timeline, and saves a montage of the worst vs the cleanest frames (score-labelled) so the thresholds
can be tuned by eye. No cutting yet — validate detection first.

Run (from ~/work/geoLocalisation; cv2/numpy via uv; s3 via aws CLI)::

    uv run --with opencv-python-headless --with numpy python -m fpv_video_qc.cli \
      --video s3://mediamtx-recordings-crunch/live/104/filtered/29.07.2026/chunk_104_..._c01.mp4 \
      --out /home/ubuntu/work/fpv_qc/chunk.json --montage /home/ubuntu/work/fpv_qc/chunk_montage.jpg \
      [--line-thr 0.15] [--tdiff-thr 40] [--every 1]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from .core import FRAME_KEYS, bad_segments, frame_score


def _fetch(video: str, td: Path) -> Path:
    if video.startswith("s3://"):
        local = td / Path(video).name
        subprocess.run(["aws", "s3", "cp", video, str(local)], capture_output=True, check=False)
        return local
    return Path(video).expanduser()


def _montage(cv2, np, frames, labels, cols=4, cell_h=160):
    tiles = []
    for fr, lab in zip(frames, labels):
        h = cell_h
        w = int(fr.shape[1] * h / fr.shape[0])
        t = cv2.resize(fr, (w, h))
        cv2.rectangle(t, (0, 0), (w - 1, 18), (0, 0, 0), -1)
        cv2.putText(t, lab, (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        tiles.append(t)
    if not tiles:
        return None
    w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, 0, 0, w - t.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
             for t in tiles]
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
            for r in rows]
    return np.vstack(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="s3:// URL or local path to the FPV video")
    ap.add_argument("--out", required=True, help="output JSON (per-frame metrics + bad segments)")
    ap.add_argument("--montage", default=None, help="save a worst-vs-clean frame montage here (jpg)")
    ap.add_argument("--line-thr", type=float, default=0.15, help="bad if line_noise > this (frac rows)")
    ap.add_argument("--tdiff-thr", type=float, default=40.0, help="bad if temporal diff > this (0..255)")
    ap.add_argument("--min-len", type=int, default=2, help="min frames for a bad segment")
    ap.add_argument("--merge-gap", type=int, default=3, help="merge bad runs <= this many good frames apart")
    ap.add_argument("--every", type=int, default=1, help="score every Nth frame (speed; index still real)")
    ap.add_argument("--montage-n", type=int, default=12, help="how many worst (and clean) frames to show")
    args = ap.parse_args(argv)

    import cv2
    import numpy as np

    tmp = tempfile.TemporaryDirectory()
    td = Path(tmp.name)
    path = _fetch(args.video, td)
    if not Path(path).exists():
        raise SystemExit(f"video not available: {args.video}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    per_frame, prev_gray, i = [], None, 0                     # scoring pass: keep metrics only, no frames
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if args.every > 1 and (i % args.every):
            i += 1
            continue
        m, prev_gray = frame_score(fr, prev_gray)
        m["frame"] = i
        per_frame.append(m)
        i += 1
    cap.release()

    flags = [(m["line_noise"] > args.line_thr) or (m["tdiff"] > args.tdiff_thr) for m in per_frame]
    for m, b in zip(per_frame, flags):
        m["bad"] = bool(b)
    idx = [m["frame"] for m in per_frame]
    segs = bad_segments(flags, min_len=args.min_len, merge_gap=args.merge_gap)
    # map scored-position segments back to real frame indices
    seg_real = [{"start": idx[s], "end": idx[e], "n_frames": e - s + 1,
                 "start_s": round(idx[s] / fps, 2) if fps else None,
                 "end_s": round(idx[e] / fps, 2) if fps else None} for s, e in segs]
    n_bad = sum(flags)
    report = {"video": args.video, "fps": round(fps, 3), "frames_scored": len(per_frame),
              "bad_frames": n_bad, "bad_frac": round(n_bad / max(1, len(per_frame)), 4),
              "n_bad_segments": len(seg_real),
              "thresholds": {"line_thr": args.line_thr, "tdiff_thr": args.tdiff_thr,
                             "min_len": args.min_len, "merge_gap": args.merge_gap},
              "bad_segments": seg_real, "frames": per_frame}
    out = Path(args.out).expanduser(); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] {len(per_frame)} frames | bad {n_bad} ({report['bad_frac']*100:.1f}%) | "
          f"{len(seg_real)} segments -> {out}", flush=True)
    for s in seg_real[:10]:
        print(f"    seg {s['start']}-{s['end']} ({s['n_frames']}f, {s['start_s']}-{s['end_s']}s)")

    if args.montage and per_frame:
        k = args.montage_n
        by_score = sorted(per_frame, key=lambda m: m["score"])
        want = {m["frame"]: f"BAD idx{m['frame']} s{m['score']:.2f}" for m in by_score[-k:]}
        want.update({m["frame"]: f"ok idx{m['frame']} s{m['score']:.2f}" for m in by_score[:k]})
        order = [m["frame"] for m in by_score[-k:][::-1]] + [m["frame"] for m in by_score[:k]]
        grabbed = {}
        cap = cv2.VideoCapture(str(path))                     # 2nd pass: fetch only the needed frames
        j = 0
        while grabbed.keys() != want.keys():
            ok, fr = cap.read()
            if not ok:
                break
            if j in want:
                grabbed[j] = fr
            j += 1
        cap.release()
        frames = [grabbed[fi] for fi in order if fi in grabbed]
        labels = [want[fi] for fi in order if fi in grabbed]
        mtg = _montage(cv2, np, frames, labels)
        if mtg is not None:
            mp = Path(args.montage).expanduser(); mp.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mp), mtg)
            print(f"[ok] montage ({k} worst + {k} clean) -> {mp}", flush=True)
    tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
