#!/usr/bin/env python
"""Download ESRI World Imagery into a GeoTIFF (no API key).

AOI can be given directly as a bbox, or computed from a directory of GT frames
named ``..._<lat>_<lon>.<ext>`` (so the map is guaranteed to cover every point).
Use ``--day-only`` to skip night frames (files whose name contains ``night``,
as tagged by classify_night.py).

Examples
--------
    # explicit bbox (south west north east), zoom 18
    python main_try/src_tif_build_universal/download_esri.py \
        --bbox 48.84 37.68 48.98 37.96 --zoom 18 \
        --out /home/ubuntu/work/tif/Liman_esri_z18.tif

    # AOI from DAY frames only (+0.02° buffer)
    python main_try/src_tif_build_universal/download_esri.py \
        --frames-dir /home/ubuntu/work/gt_liman/GT_flat --day-only \
        --margin-deg 0.02 --zoom 18 \
        --out /home/ubuntu/work/tif/Liman_day_esri_z18.tif
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # put main_try/ on the path

from src_tif_build_universal import DownloadConfig, EsriSource, download_tiles


def parse_gt(name: str):
    parts = Path(name).stem.split("_")
    if len(parts) < 2:
        return None
    try:
        return float(parts[-2]), float(parts[-1])
    except ValueError:
        return None


def aoi_from_frames(frames_dir: str, glob: str, margin_deg: float,
                    exclude_substr: str | None = None):
    lats, lons, used, excluded = [], [], 0, 0
    for f in sorted(Path(frames_dir).glob(glob)):
        if exclude_substr and exclude_substr in f.name:
            excluded += 1
            continue
        gt = parse_gt(f.name)
        if gt:
            lats.append(gt[0])
            lons.append(gt[1])
            used += 1
    if not lats:
        raise SystemExit(f"No GT coordinates parsed under {frames_dir}/{glob}")
    s, n = min(lats) - margin_deg, max(lats) + margin_deg
    w, e = min(lons) - margin_deg, max(lons) + margin_deg
    print(f"AOI from {used} frames (excluded {excluded}): "
          f"S{s:.5f} W{w:.5f} N{n:.5f} E{e:.5f}  (margin {margin_deg}°)")
    return s, w, n, e


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download ESRI World Imagery to a GeoTIFF (no API key).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bbox", nargs=4, type=float,
                   metavar=("SOUTH", "WEST", "NORTH", "EAST"),
                   help="AOI bounds in WGS84 degrees.")
    g.add_argument("--frames-dir",
                   help="Compute AOI from GT frames named ..._<lat>_<lon>.<ext>.")
    ap.add_argument("--glob", default="*.jpg", help="Frame glob for --frames-dir.")
    ap.add_argument("--day-only", action="store_true",
                    help="Skip night frames (name contains 'night').")
    ap.add_argument("--exclude-substr", default=None,
                    help="Skip frames whose name contains this substring "
                         "(overrides --day-only if given).")
    ap.add_argument("--margin-deg", type=float, default=0.02,
                    help="Extra buffer around the GT bbox (degrees).")
    ap.add_argument("--zoom", type=int, default=18,
                    help="Esri LOD (18≈0.6 m/px, 19≈0.3, 20≈0.15).")
    ap.add_argument("--tile-px", type=int, default=2048, help="Server tile size (≤4096).")
    ap.add_argument("--workers", type=int, default=16, help="Parallel download threads.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-download from scratch instead of resuming.")
    ap.add_argument("--out", required=True, help="Output GeoTIFF path.")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if a.bbox:
        s, w, n, e = a.bbox
    else:
        exclude = a.exclude_substr if a.exclude_substr is not None else ("night" if a.day_only else None)
        s, w, n, e = aoi_from_frames(a.frames_dir, a.glob, a.margin_deg, exclude)

    aoi_points = [(s, w), (n, w), (n, e), (s, e)]  # (lat, lon) corners
    src = EsriSource(zoom=a.zoom, tile_px=a.tile_px)
    cfg = DownloadConfig(max_workers=a.workers, overwrite=a.overwrite)
    out = download_tiles(src, aoi_points, a.out, cfg)
    print("Saved:", out)


if __name__ == "__main__":
    main()
