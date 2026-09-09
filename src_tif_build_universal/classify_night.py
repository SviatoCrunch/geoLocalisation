#!/usr/bin/env python
"""Classify GT frames day/night by chroma and tag night ones in the filename.

Rule-based, no weights/network (ports main_try/chroma_day_night.ipynb):
    downscale to 256 px long side (INTER_AREA),
    chroma = mean over pixels & channels of |channel - pixel_gray|,
    NIGHT if chroma < THRESHOLD (default 12; validated acc 0.986).

Night frames are renamed by inserting a ``night`` token after the id, e.g.
    50_48.9038201043_37.8398358686.jpg
 -> 50_night_48.9038201043_37.8398358686.jpg
so parse_gt (last two ``_``-parts = lat, lon) still works and day frames are
exactly the ones without ``night`` in the name.

Dry-run by default; pass --apply to actually rename.

    python main_try/src_tif_build_universal/classify_night.py \
        --frames-dir /home/ubuntu/work/gt_liman/GT_flat            # preview
    python main_try/src_tif_build_universal/classify_night.py \
        --frames-dir /home/ubuntu/work/gt_liman/GT_flat --apply    # rename
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

THRESHOLD = 12.0


def chroma(path: Path, long_side: int = 256) -> float | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR uint8
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(
            img, (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    f = img.astype(np.float32)
    gray = f.mean(axis=2, keepdims=True)          # per-pixel gray = mean of B,G,R
    return float(np.abs(f - gray).mean())         # mean |channel - gray|


def night_name(name: str) -> str:
    p = Path(name)
    parts = p.stem.split("_")
    if len(parts) < 3:                            # keep last two as lat/lon
        return name
    return "_".join([parts[0], "night", *parts[1:]]) + p.suffix


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Tag night GT frames (chroma rule) in their filename.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--thr", type=float, default=THRESHOLD,
                    help="Night if chroma < thr.")
    ap.add_argument("--apply", action="store_true",
                    help="Actually rename night frames (default: dry-run).")
    a = ap.parse_args()

    files = sorted(Path(a.frames_dir).glob(a.glob))
    n_day = n_night = n_already = n_bad = 0
    renames = []
    print(f"{'chroma':>7}  {'class':<6}  file")
    for f in files:
        if "night" in f.stem.split("_"):
            n_already += 1
            print(f"{'--':>7}  NIGHT*  {f.name}   (already tagged)")
            continue
        c = chroma(f)
        if c is None:
            n_bad += 1
            print(f"{'--':>7}  BAD     {f.name}   (unreadable)")
            continue
        is_night = c < a.thr
        print(f"{c:7.2f}  {'NIGHT' if is_night else 'DAY':<6}  {f.name}")
        if is_night:
            n_night += 1
            renames.append((f, f.with_name(night_name(f.name))))
        else:
            n_day += 1

    print("-" * 60)
    print(f"day={n_day}  night={n_night}  already_night={n_already}  bad={n_bad}  "
          f"total={len(files)}  (thr={a.thr})")
    if renames:
        print(f"\n{'APPLYING' if a.apply else 'WOULD RENAME'} {len(renames)} night frames:")
        for src, dst in renames:
            print(f"  {src.name}  ->  {dst.name}")
            if a.apply:
                src.rename(dst)
        if not a.apply:
            print("\n(dry-run — pass --apply to rename)")


if __name__ == "__main__":
    main()
