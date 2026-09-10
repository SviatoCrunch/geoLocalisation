#!/usr/bin/env python
"""
push_to_s3.py — upload files from a local directory to an S3 prefix.

Mirrors the local tree under the target prefix (keys use POSIX separators).
Non-destructive: existing objects are never deleted; with the default
skip-existing an object already present at the same size is left untouched.

Usage
-----
    # dry-run
    python -m s3_gt_sync.push_to_s3 \
        --src   "C:/data/GT_flat" \
        --s3-uri s3://my-bucket/uav/Kramatorsk_GT \
        --dry-run

    # real upload
    python -m s3_gt_sync.push_to_s3 \
        --src   "C:/data/GT_flat" \
        --s3-uri s3://my-bucket/uav/Kramatorsk_GT
"""
from __future__ import annotations

import argparse
import logging

from .core import DEFAULT_EXTS, push


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Upload local files to an S3 prefix (non-destructive).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--src", required=True,
                    help="Local source directory.")
    ap.add_argument("--s3-uri", required=True,
                    help="Target prefix, e.g. s3://bucket/path/to/dest")
    ap.add_argument("--exts", nargs="+", default=sorted(DEFAULT_EXTS),
                    help="File extensions to upload.")
    ap.add_argument("--no-recursive", dest="recursive", action="store_false",
                    help="Upload only the top-level files of --src.")
    ap.add_argument("--endpoint-url", default=None,
                    help="Custom S3 endpoint (MinIO/other); else AWS_ENDPOINT_URL.")
    ap.add_argument("--no-skip-existing", dest="skip_existing",
                    action="store_false",
                    help="Re-upload even when a same-size object already exists.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List planned uploads without transferring.")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Enable DEBUG logging (per-file skip lines).")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.exts}

    stats = push(
        args.src, args.s3_uri, exts,
        recursive=args.recursive, skip_existing=args.skip_existing,
        dry_run=args.dry_run, endpoint_url=args.endpoint_url,
    )

    verb = "would upload" if args.dry_run else "uploaded"
    print("\n" + "=" * 50)
    print(f"  {verb:14s}: {stats['uploaded']}")
    print(f"  skipped (exist): {stats['skipped']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
