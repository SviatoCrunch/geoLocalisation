"""
Core S3 <-> local sync helpers for UAV GT frames.

The GT-naming rules mirror ``main_try/labels/rename_gt.py`` so that a frame ends
up under the same "correct" name whether it was renamed on disk or resolved here
straight from S3:

  N            (plain frame)      -> N_lat_lon.ext   (lat/lon from paired N_lat,lon)
  N_lat,lon    (map screenshot)   -> coordinate SOURCE only, never copied/deleted
  N_lat_lon    (already GT)        -> copied as-is

Unlike the notebook pipeline this module is non-destructive on the remote side:
nothing is ever deleted from S3. The screenshot is only read for its filename.
"""
from __future__ import annotations

import logging
import os
import re
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import PurePosixPath

log = logging.getLogger(__name__)

# Same patterns as rename_gt.py (kept in sync intentionally).
_MAP_SHOT = re.compile(r"^(\d+)_(-?\d+\.\d+),(-?\d+\.\d+)$")   # N_lat,lon  (comma)
_PLAIN = re.compile(r"^\d+$")                                    # just N
_DONE = re.compile(r"^(\d+)_(-?\d+\.\d+)_(-?\d+\.\d+)$")       # N_lat_lon  (underscore)

DEFAULT_EXTS = {".jpg", ".jpeg", ".png"}

# Output sub-directories this tool produces on S3 (GT frames + their masks).
# pull() always skips them so a sync never re-ingests its own output.
ALWAYS_EXCLUDE_SUBDIRS = {"GT_flat", "GT_flat_mask"}


# ── URI / client helpers ──────────────────────────────────────────────────────

def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/prefix`` into ``(bucket, prefix)``.

    ``prefix`` keeps no leading slash and no trailing slash. Raises ValueError on
    anything that is not a well-formed ``s3://`` URI with a bucket.
    """
    parsed = urllib.parse.urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {s3_uri!r} (expected s3://bucket/prefix)")
    return parsed.netloc, parsed.path.strip("/")


def make_s3_client(endpoint_url: str | None = None):
    """Create a boto3 S3 client using the standard AWS credential chain.

    ``endpoint_url`` (or the AWS_ENDPOINT_URL env var) targets an S3-compatible
    store such as MinIO; when both are unset the real AWS endpoint is used.
    """
    import boto3

    endpoint_url = endpoint_url or os.environ.get("AWS_ENDPOINT_URL")
    return boto3.client(
        "s3",
        region_name=os.environ.get("AWS_REGION"),
        endpoint_url=endpoint_url or None,
    )


# ── GT-name resolution ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GtTarget:
    """One planned pull: fetch ``src_key`` from S3 and save it locally as
    ``dst_name``. ``size`` is the S3 object size (bytes) for skip-existing."""
    src_key: str
    dst_name: str
    size: int


def _folder_of(key: str) -> str:
    """The S3 "directory" of a key = everything before the last '/'."""
    return key.rsplit("/", 1)[0] if "/" in key else ""


def _is_excluded(rel_parts: tuple[str, ...], exclude: set[str]) -> bool:
    """True if any path component of the key (relative to the prefix) matches an
    excluded sub-directory name."""
    return any(part in exclude for part in rel_parts[:-1])


def resolve_folder_targets(objects: list[tuple[str, int]], exts: set[str],
                           decimals: int) -> tuple[list[GtTarget], list[str]]:
    """Apply the GT rules to the images of a single S3 folder.

    ``objects`` is ``[(key, size), ...]`` for one folder. Returns
    ``(targets, unpaired)`` where ``unpaired`` lists plain-N keys that had no
    paired screenshot (so no coordinates -> cannot be named, skipped).
    """
    images = [(k, s) for k, s in objects
              if PurePosixPath(k).suffix.lower() in exts]

    # N_lat,lon screenshots -> coordinate lookup keyed by N. Never emitted.
    coord_map: dict[str, tuple[str, str]] = {}
    for k, _ in images:
        m = _MAP_SHOT.match(PurePosixPath(k).stem)
        if m:
            n, lat, lon = m.group(1), float(m.group(2)), float(m.group(3))
            coord_map[n] = (f"{round(lat, decimals)}", f"{round(lon, decimals)}")

    targets: list[GtTarget] = []
    unpaired: list[str] = []
    for k, size in images:
        p = PurePosixPath(k)
        stem, suffix = p.stem, p.suffix
        if _DONE.match(stem):
            # Already correctly named -> copy verbatim.
            targets.append(GtTarget(k, p.name, size))
        elif _PLAIN.match(stem):
            coord = coord_map.get(stem)
            if coord is None:
                unpaired.append(k)
                continue
            lat, lon = coord
            targets.append(GtTarget(k, f"{stem}_{lat}_{lon}{suffix}", size))
        # _MAP_SHOT and anything else: not copied.
    return targets, unpaired


def plan_pull(s3_uri: str, exts: set[str] = DEFAULT_EXTS, *,
              exclude: set[str] | None = None, decimals: int = 10,
              endpoint_url: str | None = None) -> tuple[list[GtTarget], list[str]]:
    """List the prefix and resolve every GT target without downloading anything.

    Walks all sub-directories under the prefix except those named in ``exclude``.
    The tool's own output sub-directories (``GT_flat`` and ``GT_flat_mask``, see
    ``ALWAYS_EXCLUDE_SUBDIRS``) are skipped unconditionally so a sync never
    re-ingests what it previously pushed. Returns ``(targets, unpaired)`` — pure
    planning, no local writes, no S3 mutation. Useful for a dry-run.
    """
    exclude = (exclude or set()) | ALWAYS_EXCLUDE_SUBDIRS
    bucket, prefix = parse_s3_uri(s3_uri)
    s3 = make_s3_client(endpoint_url)
    paginator = s3.get_paginator("list_objects_v2")

    base = PurePosixPath(prefix) if prefix else None
    folders: dict[str, list[tuple[str, int]]] = {}
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = PurePosixPath(key).relative_to(base) if base else PurePosixPath(key)
            if _is_excluded(rel.parts, exclude):
                continue
            folders.setdefault(_folder_of(key), []).append((key, obj["Size"]))

    all_targets: list[GtTarget] = []
    all_unpaired: list[str] = []
    for objs in folders.values():
        targets, unpaired = resolve_folder_targets(objs, exts, decimals)
        all_targets.extend(targets)
        all_unpaired.extend(unpaired)
    return all_targets, all_unpaired


# ── Pull: S3 -> local ─────────────────────────────────────────────────────────

def pull(s3_uri: str, dest_dir: str, exts: set[str] = DEFAULT_EXTS, *,
         exclude: set[str] | None = None, decimals: int = 10,
         skip_existing: bool = True, dry_run: bool = False,
         endpoint_url: str | None = None) -> dict:
    """Sync a UAV-GT S3 prefix into a flat local directory under correct names.

    Non-destructive on S3. Only files whose target name is missing locally are
    downloaded (``skip_existing``; a same-size match counts as present). Returns
    a stats dict: downloaded / skipped / unpaired / conflicts.
    """
    bucket, _ = parse_s3_uri(s3_uri)
    targets, unpaired = plan_pull(s3_uri, exts, exclude=exclude,
                                  decimals=decimals, endpoint_url=endpoint_url)

    if not dry_run:
        os.makedirs(dest_dir, exist_ok=True)

    log.info("pull plan  targets=%d unpaired=%d dest=%s dry_run=%s",
             len(targets), len(unpaired), dest_dir, dry_run)
    t0 = time.monotonic()

    s3 = None if dry_run else make_s3_client(endpoint_url)
    downloaded = skipped = 0
    conflicts: list[tuple[str, str]] = []
    seen: dict[str, str] = {}  # dst_name -> src_key, to catch cross-folder clashes

    for t in targets:
        # Two different source keys mapping to the same output name = a real
        # collision (e.g. same frame number in two cities). Keep the first, warn.
        prev = seen.get(t.dst_name)
        if prev is not None and prev != t.src_key:
            conflicts.append((t.src_key, t.dst_name))
            log.warning("pull conflict  %s -> %s already claimed by %s",
                        t.src_key, t.dst_name, prev)
            continue
        seen[t.dst_name] = t.src_key

        local = os.path.join(dest_dir, t.dst_name)
        if (skip_existing and os.path.exists(local)
                and os.path.getsize(local) == t.size):
            skipped += 1
            log.debug("pull skip (exists)  %s", t.dst_name)
            continue

        if dry_run:
            log.info("pull would fetch  %s -> %s", t.src_key, t.dst_name)
            downloaded += 1
            continue

        log.info("pull fetch  %s -> %s (%.2f MB)",
                 t.src_key, t.dst_name, t.size / 1_048_576)
        s3.download_file(bucket, t.src_key, local)
        downloaded += 1

    stats = {
        "downloaded": downloaded,
        "skipped": skipped,
        "unpaired": len(unpaired),
        "conflicts": len(conflicts),
        "unpaired_keys": unpaired,
        "conflict_pairs": conflicts,
    }
    log.info("pull done  downloaded=%d skipped=%d unpaired=%d conflicts=%d elapsed=%.1fs",
             downloaded, skipped, len(unpaired), len(conflicts), time.monotonic() - t0)
    return stats


# ── Push: local -> S3 ─────────────────────────────────────────────────────────

_CONTENT_TYPE = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
}


def push(src_dir: str, s3_uri: str, exts: set[str] = DEFAULT_EXTS, *,
         recursive: bool = True, skip_existing: bool = True, dry_run: bool = False,
         endpoint_url: str | None = None) -> dict:
    """Upload local files into an ``s3://bucket/prefix`` target.

    Keys are ``<prefix>/<path-relative-to-src_dir>`` (POSIX separators). With
    ``skip_existing`` an object already present with the same size is left as-is
    (a HEAD is issued per file). Never deletes anything remote. Returns a stats
    dict: uploaded / skipped.
    """
    bucket, prefix = parse_s3_uri(s3_uri)
    src = os.path.abspath(src_dir)
    if not os.path.isdir(src):
        raise ValueError(f"Source directory not found: {src_dir!r}")

    s3 = make_s3_client(endpoint_url)

    files: list[str] = []
    if recursive:
        for dirpath, _, names in os.walk(src):
            for n in names:
                if os.path.splitext(n)[1].lower() in exts:
                    files.append(os.path.join(dirpath, n))
    else:
        for n in sorted(os.listdir(src)):
            full = os.path.join(src, n)
            if os.path.isfile(full) and os.path.splitext(n)[1].lower() in exts:
                files.append(full)

    log.info("push plan  files=%d src=%s dest=%s dry_run=%s",
             len(files), src, s3_uri, dry_run)
    t0 = time.monotonic()

    uploaded = skipped = 0
    for full in sorted(files):
        rel = os.path.relpath(full, src).replace(os.sep, "/")
        key = "/".join(p for p in (prefix, rel) if p)
        size = os.path.getsize(full)

        if skip_existing:
            try:
                head = s3.head_object(Bucket=bucket, Key=key)
                if head["ContentLength"] == size:
                    skipped += 1
                    log.debug("push skip (exists)  %s", key)
                    continue
            except Exception:
                pass  # not found / no access to HEAD -> attempt upload

        if dry_run:
            log.info("push would upload  %s -> s3://%s/%s", rel, bucket, key)
            uploaded += 1
            continue

        extra = {}
        ct = _CONTENT_TYPE.get(os.path.splitext(full)[1].lower())
        if ct:
            extra["ContentType"] = ct
        log.info("push upload  %s -> s3://%s/%s (%.2f MB)",
                 rel, bucket, key, size / 1_048_576)
        s3.upload_file(full, bucket, key, ExtraArgs=extra or None)
        uploaded += 1

    log.info("push done  uploaded=%d skipped=%d elapsed=%.1fs",
             uploaded, skipped, time.monotonic() - t0)
    return {"uploaded": uploaded, "skipped": skipped}
