"""
s3_gt_sync — S3 <-> local sync for UAV GT frames.

Two directions, built on the same GT-naming rules as ``main_try/labels/rename_gt.py``:

  pull_from_s3.py : S3 prefix  ->  local flat directory
      * walks every "sub-directory" under the prefix EXCEPT one you exclude
      * copies only files that are not already present locally
      * writes each frame under its CORRECT name  (N + N_lat,lon  ->  N_lat_lon)
      * NEVER deletes anything on S3 (the N_lat,lon screenshot is read for its
        coordinates only, then left in place)

  push_to_s3.py   : local directory  ->  S3 prefix
      * uploads files that are not already present under the target prefix

  make_masks.py   : COCO result.json  ->  per-class binary masks (GT_flat_mask)
      * one bitwise PNG per (image, class), named to pair with its frame
      * optionally pushes them to S3; pull() always skips GT_flat / GT_flat_mask

Credentials come from the standard AWS chain (env vars / shared config / IAM
role); an optional custom endpoint (MinIO / other S3-compatible store) can be
supplied with --endpoint-url or the AWS_ENDPOINT_URL env var.
"""

from .core import (
    ALWAYS_EXCLUDE_SUBDIRS,
    GtTarget,
    parse_s3_uri,
    make_s3_client,
    plan_pull,
    pull,
    push,
)
from .masks import MASK_SUBDIR, MaskTarget, masks_from_coco
from .map_coverage import CoverageReport, FrameCoverage, coverage_report

__all__ = [
    "ALWAYS_EXCLUDE_SUBDIRS",
    "GtTarget",
    "parse_s3_uri",
    "make_s3_client",
    "plan_pull",
    "pull",
    "push",
    "MASK_SUBDIR",
    "MaskTarget",
    "masks_from_coco",
    "CoverageReport",
    "FrameCoverage",
    "coverage_report",
]
