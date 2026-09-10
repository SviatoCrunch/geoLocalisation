#!/usr/bin/env python
"""
check_map_coverage.py — how many GT frames fall inside a raster map (GeoTIFF/COG).

Parses lat/lon from GT file names (..._<lat>_<lon>.<ext>), projects them into the
raster's CRS and reports, per frame, whether it lands inside / on the nodata
border / outside the map. Read-only: nothing is downloaded or modified.

Usage
-----
    python -m s3_gt_sync.check_map_coverage \
        --tif        /path/to/lyman_map.tif \
        --frames-dir /home/ubuntu/work/gt_liman/GT_flat \
        --check-nodata

    # save the full per-frame breakdown as JSON too
    python -m s3_gt_sync.check_map_coverage \
        --tif ... --frames-dir ... --out lyman_coverage.json
"""
from __future__ import annotations

import argparse
import json

from .map_coverage import coverage_report


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Report how many GT frames fall inside a GeoTIFF/COG map.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--tif", required=True,
                    help="Raster map (GeoTIFF / COG) to test against.")
    ap.add_argument("--frames-dir", required=True,
                    help="Directory of GT frames named ..._<lat>_<lon>.<ext>.")
    ap.add_argument("--glob", default="*.jpg",
                    help="Frame glob within --frames-dir.")
    ap.add_argument("--check-nodata", action="store_true",
                    help="Also sample the valid-data mask; a point on the "
                         "nodata/alpha border counts as NOT covered.")
    ap.add_argument("--out", default=None,
                    help="Optional path to write the full per-frame JSON report.")
    ap.add_argument("--list-inside", action="store_true",
                    help="Also print the frames that are inside (not just misses).")
    args = ap.parse_args()

    rep = coverage_report(
        args.tif, args.frames_dir,
        frame_glob=args.glob, check_nodata=args.check_nodata,
    )
    c = rep.counts()
    u = rep.units
    left, bottom, right, top = rep.bbox

    print("=" * 60)
    print(f"  MAP    : {rep.tif}")
    print(f"  crs    : {rep.crs}   size {rep.width}x{rep.height}   units {u}")
    print(f"  bbox   : x[{left:.1f} .. {right:.1f}]  y[{bottom:.1f} .. {top:.1f}]")
    print(f"  FRAMES : {rep.frames_dir}  (glob {args.glob})")
    print("=" * 60)
    print(f"  total          : {rep.n_total}")
    print(f"  parseable      : {rep.n_parseable}   (unparseable: {c['bad']})")
    print(f"  INSIDE map     : {c['inside']}")
    print(f"  ON nodata      : {c['nodata']}"
          + ("" if args.check_nodata else "  (skipped; pass --check-nodata)"))
    print(f"  OUTSIDE bbox   : {c['outside']}")

    for f in sorted(rep.by_status("outside"), key=lambda t: -t.dist_out):
        print(f"    OUT     {f.filename}  ~{f.dist_out:.0f} {u} past edge"
              f"  (lat={f.lat}, lon={f.lon})")
    for f in rep.by_status("nodata"):
        print(f"    NODATA  {f.filename}  in bbox, margin {f.margin:.0f} {u}")
    for f in rep.by_status("bad"):
        print(f"    BAD     {f.filename}  (no lat/lon in name)")
    if args.list_inside:
        for f in rep.by_status("inside"):
            print(f"    IN      {f.filename}  margin {f.margin:.0f} {u}")

    covered = c["inside"]
    print("-" * 60)
    verdict = "усі потрапляють ✅" if covered == rep.n_parseable else "НЕ всі потрапляють ⚠️"
    print(f"  => {covered}/{rep.n_parseable} GT-точок реально на карті — {verdict}")
    print("=" * 60)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rep.to_dict(), fh, ensure_ascii=False, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
