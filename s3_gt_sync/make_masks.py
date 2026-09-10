#!/usr/bin/env python
"""
make_masks.py — build per-class binary masks from a COCO ``result.json`` and
(optionally) push them to S3.

Reads a COCO annotation file, rasterises one bitwise PNG per (image, class) into
the output directory (by convention ``<gt-dir>/GT_flat_mask``), then — if a
target S3 URI is given — uploads them with the same non-destructive push used for
the frames. pull() already skips ``GT_flat`` and ``GT_flat_mask`` on S3, so these
masks never get re-ingested.

Usage
-----
    # build masks only (dry-run first)
    python -m s3_gt_sync.make_masks \
        --coco /home/ubuntu/work/gt_kup/result.json \
        --dest /home/ubuntu/work/gt_kup/GT_flat_mask --dry-run

    # build + push to S3
    python -m s3_gt_sync.make_masks \
        --coco /home/ubuntu/work/gt_kup/result.json \
        --dest /home/ubuntu/work/gt_kup/GT_flat_mask \
        --push-s3-uri s3://geo-reference/gt/raw/kup/GT_flat_mask
"""
from __future__ import annotations

import argparse
import logging

from .masks import masks_from_coco
from .core import push


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rasterise per-class binary masks from a COCO result.json "
                    "and optionally push them to S3 (non-destructive).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--coco", required=True,
                    help="COCO result.json (images + annotations + categories).")
    ap.add_argument("--dest", required=True,
                    help="Output directory for masks, e.g. .../GT_flat_mask")
    ap.add_argument("--push-s3-uri", default=None,
                    help="If set, upload the built masks to this S3 prefix.")
    ap.add_argument("--endpoint-url", default=None,
                    help="Custom S3 endpoint (MinIO/other); else AWS_ENDPOINT_URL.")
    ap.add_argument("--no-skip-existing", dest="skip_existing",
                    action="store_false",
                    help="Rebuild/re-upload even when a same-name file exists.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Plan only: write no PNGs and upload nothing.")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Enable DEBUG logging (per-file lines).")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    stats = masks_from_coco(
        args.coco, args.dest,
        skip_existing=args.skip_existing, dry_run=args.dry_run,
    )

    verb = "would write" if args.dry_run else "written"
    print("\n" + "=" * 50)
    print(f"  masks {verb:11s}: {stats['written']}")
    print(f"  skipped (exist) : {stats['skipped']}")
    print(f"  empty (no seg)  : {stats['empty']}")
    print(f"  images / anns   : {stats['images']} / {stats['annotations']}")
    print(f"  classes         : {', '.join(stats['classes']) or '-'}")
    print("=" * 50)

    if args.push_s3_uri:
        pstats = push(
            args.dest, args.push_s3_uri, {".png"},
            skip_existing=args.skip_existing, dry_run=args.dry_run,
            endpoint_url=args.endpoint_url,
        )
        pverb = "would upload" if args.dry_run else "uploaded"
        print(f"  masks {pverb:11s}: {pstats['uploaded']}")
        print(f"  push skipped    : {pstats['skipped']}")
        print("=" * 50)


if __name__ == "__main__":
    main()
