#!/usr/bin/env python
"""
pull_from_s3.py — sync UAV GT frames from S3 into a local flat directory.

What it does (non-destructive on S3):
  * walks every sub-directory under the prefix EXCEPT the ones you --exclude
  * resolves each frame's CORRECT name (N + paired N_lat,lon -> N_lat_lon),
    reading the N_lat,lon screenshot for coordinates only — never deleting it
  * copies to the local output directory only files that are not already there

Usage
-----
    # dry-run: show what would be fetched, touch nothing
    python -m s3_gt_sync.pull_from_s3 \
        --s3-uri s3://my-bucket/uav/Kramatorsk \
        --dest  "C:/data/GT_flat" \
        --exclude raw --dry-run

    # real sync
    python -m s3_gt_sync.pull_from_s3 \
        --s3-uri s3://my-bucket/uav/Kramatorsk \
        --dest  "C:/data/GT_flat" \
        --exclude raw
"""
from __future__ import annotations

import argparse
import logging

from .core import DEFAULT_EXTS, pull


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sync UAV GT frames from an S3 prefix into a local flat "
                    "directory under correct names. Nothing is deleted on S3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--s3-uri", required=True,
                    help="Source prefix, e.g. s3://bucket/path/to/city")
    ap.add_argument("--dest", required=True,
                    help="Local output directory (flat, created if missing).")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="Sub-directory name(s) to skip anywhere in the tree.")
    ap.add_argument("--exts", nargs="+", default=sorted(DEFAULT_EXTS),
                    help="Image extensions to process.")
    ap.add_argument("--decimals", type=int, default=10,
                    help="Decimal places in output lat/lon (matches rename_gt).")
    ap.add_argument("--endpoint-url", default=None,
                    help="Custom S3 endpoint (MinIO/other); else AWS_ENDPOINT_URL.")
    ap.add_argument("--no-skip-existing", dest="skip_existing",
                    action="store_false",
                    help="Re-download even when a same-size local file exists.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List planned fetches without downloading.")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Enable DEBUG logging (per-file skip lines).")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.exts}

    stats = pull(
        args.s3_uri, args.dest, exts,
        exclude=set(args.exclude), decimals=args.decimals,
        skip_existing=args.skip_existing, dry_run=args.dry_run,
        endpoint_url=args.endpoint_url,
    )

    verb = "would download" if args.dry_run else "downloaded"
    print("\n" + "=" * 50)
    print(f"  {verb:14s}: {stats['downloaded']}")
    print(f"  skipped (exist): {stats['skipped']}")
    print(f"  unpaired (no coords): {stats['unpaired']}")
    print(f"  name conflicts : {stats['conflicts']}")
    if stats["unpaired_keys"]:
        print("\n  [!] plain-N frames with no paired N_lat,lon screenshot:")
        for k in stats["unpaired_keys"]:
            print("     ", k)
    if stats["conflict_pairs"]:
        print("\n  [!] name conflicts (kept the first source):")
        for src, name in stats["conflict_pairs"]:
            print(f"      {src}  ->  {name} (already taken)")
    print("=" * 50)


if __name__ == "__main__":
    main()
