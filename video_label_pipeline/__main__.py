"""CLI: build a video-annotation manifest for a date range.

    # S3 (mediamtx layout), stations 104 and 138, a date range:
    python -m video_label_pipeline --source s3 --station 104,138 \
        --date-from 10.10.2026 --date-to 15.10.2026 --out manifest.json

    # local folder of chunk_*.mp4:
    python -m video_label_pipeline --source local --dir /path/to/videos \
        --date-from 2026-10-10 --date-to 2026-10-15 --out manifest.json

Needs ``quality_cls`` importable (put ``quality-cls/src`` on PYTHONPATH) and its
``checkpoints/head.pt`` available — unless ``--dry-structure`` is used, which
skips quality/embedding (no model) and only reports the split.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import SplitRecipe, build_frames_manifest, build_manifest
from .source import LocalSource, S3Source


class _StructureOnlyBackend:
    """No-model backend for --dry-structure: marks every frame 'good', embeds by
    mean BGR so diversity still runs. Lets you inspect the split without torch."""

    def score(self, frames_bgr):
        return [("good", 1.0) for _ in frames_bgr]

    def embed(self, frames_bgr):
        import numpy as np
        return np.stack([f.reshape(-1, f.shape[-1]).mean(axis=0) for f in frames_bgr]) \
            if frames_bgr else np.zeros((0, 3), dtype=np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("s3", "local", "frames"), required=True)
    ap.add_argument("--date-from", help="required for s3/local")
    ap.add_argument("--date-to", help="required for s3/local")
    # s3
    ap.add_argument("--station", help="s3: station number, comma list, or 'all'")
    ap.add_argument("--bucket", default="mediamtx-recordings-crunch")
    # local videos / frames
    ap.add_argument("--dir", help="local: dir of chunk_*.mp4;  frames: frames_good root")
    # selection / tagging
    ap.add_argument("--k", type=int, default=5, help="good frames per sub-video (default 5)")
    ap.add_argument("--daynight-thr", type=float, default=12.0)
    ap.add_argument("--every", type=int, default=1,
                    help="decode every N-th frame (default 1; keep 1 for faithful split)")
    ap.add_argument("--limit", type=int, default=None, help="process at most N videos")
    # split recipe (defaults == frames_good)
    ap.add_argument("--adaptive-thr", type=float, default=3.0)
    ap.add_argument("--min-content", type=float, default=15.0)
    ap.add_argument("--content-window", type=int, default=2)
    ap.add_argument("--content-reduce", type=int, default=4)
    ap.add_argument("--min-seg", type=int, default=2)
    ap.add_argument("--width", type=int, default=480)
    # model
    ap.add_argument("--checkpoint", default=None, help="quality_cls head.pt (default: its config)")
    ap.add_argument("--config", default=None, help="quality_cls YAML config")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-structure", action="store_true",
                    help="no model: split-only (every frame counted 'good')")
    ap.add_argument("--out", required=True, help="output manifest JSON path")
    args = ap.parse_args(argv)

    if args.source in ("s3", "local") and not (args.date_from and args.date_to):
        ap.error("--date-from and --date-to are required for --source s3/local")

    if args.dry_structure:
        backend = _StructureOnlyBackend()
    else:
        from .quality import QualityClsBackend
        backend = QualityClsBackend(checkpoint=args.checkpoint, config=args.config,
                                    device=args.device)

    def progress(i, rec):
        n = rec.get("n_subvideos", 0)
        sel = sum(len(s.get("selected", [])) for s in rec.get("subvideos", []))
        print(f"[{i + 1}] {rec['chunk']}: {n} sub-videos, {sel} frames selected"
              f"{' (' + rec['status'] + ')' if rec.get('status') else ''}", flush=True)

    if args.source == "frames":
        if not args.dir:
            ap.error("--dir (frames_good root) is required for --source frames")
        manifest = build_frames_manifest(args.dir, backend, k=args.k, thr=args.daynight_thr,
                                         limit=args.limit, on_video=progress)
    else:
        if args.source == "s3":
            if not args.station:
                ap.error("--station is required for --source s3")
            source = S3Source(args.station, args.date_from, args.date_to, bucket=args.bucket)
        else:
            if not args.dir:
                ap.error("--dir is required for --source local")
            source = LocalSource(args.dir, args.date_from, args.date_to)
        recipe = SplitRecipe(adaptive_thr=args.adaptive_thr, min_content=args.min_content,
                             content_window=args.content_window, content_reduce=args.content_reduce,
                             min_seg=args.min_seg, width=args.width)
        manifest = build_manifest(source, backend, recipe, date_from=args.date_from,
                                  date_to=args.date_to, k=args.k, thr=args.daynight_thr,
                                  every=args.every, limit=args.limit, on_video=progress)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nwrote {manifest['n_videos']} video(s) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
