"""Audit KMZ/KML files under an S3 prefix (or a local dir) for video references.

Reuses the S3 client + URI parser from :mod:`s3_gt_sync.core` (same credential chain, endpoint
override, paginator). Streams each object into memory (KMZs are small), classifies it, prints a
per-file line and an aggregate, and — with ``--schema`` — shows WHERE video links live across the
whole set so the real KML schema can be discovered before the detector is hardened.

Run::

    python -m kmz_video_audit.cli s3://geo-reference/some/kmz/prefix --schema
    python -m kmz_video_audit.cli s3://geo-reference/some/kmz/prefix --json out.json
    python -m kmz_video_audit.cli /local/dir/of/kmz --list-urls
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from .core import KmzAudit, audit_kmz_bytes

_KMZ_EXTS = (".kmz", ".kml")


def _iter_s3(s3_uri: str, endpoint_url: str | None):
    """Yield ``(key, bytes)`` for every ``.kmz``/``.kml`` object under the prefix."""
    from s3_gt_sync.core import make_s3_client, parse_s3_uri

    bucket, prefix = parse_s3_uri(s3_uri)
    s3 = make_s3_client(endpoint_url)
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(_KMZ_EXTS):
                body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                yield key, body


def _iter_local(root: str):
    """Yield ``(path, bytes)`` for every ``.kmz``/``.kml`` under a dir (or a single file)."""
    p = Path(root)
    files = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.suffix.lower() in _KMZ_EXTS)
    for f in files:
        yield str(f), f.read_bytes()


def audit_target(target: str, endpoint_url: str | None = None) -> list[KmzAudit]:
    """Audit every KMZ/KML under ``target`` (``s3://…`` prefix or local path)."""
    results = []
    if target.startswith("s3://"):
        for key, body in _iter_s3(target, endpoint_url):
            results.append(audit_kmz_bytes(body, source=key))
    else:
        for path, body in _iter_local(target):
            results.append(audit_kmz_bytes(body, source=path))   # sniffs zip vs bare KML
    return results


def _print_report(results: list[KmzAudit], *, schema: bool, list_urls: bool) -> None:
    for r in results:
        print(r.summary())
        if list_urls and r.urls:
            for u in r.urls:
                print(f"         url: {u}")

    ok = [r for r in results if r.error is None]
    err = [r for r in results if r.error is not None]
    with_v = [r for r in ok if r.has_video]
    print(f"\n== {len(results)} files: {len(with_v)} with video, "
          f"{len(ok) - len(with_v)} without, {len(err)} errors ==")

    if schema and ok:
        where_ctr: collections.Counter = collections.Counter()
        reason_ctr: collections.Counter = collections.Counter()
        extkey_ctr: collections.Counter = collections.Counter()
        for r in ok:
            for v in r.video_links:
                where_ctr[v.where] += 1
                reason_ctr[v.reason] += 1
            for k in r.ext_data_keys:
                extkey_ctr[k] += 1
        print("\n-- video links by KML location --")
        for w, c in where_ctr.most_common():
            print(f"   {c:4d}  {w}")
        print("-- video links by reason --")
        for w, c in reason_ctr.most_common():
            print(f"   {c:4d}  {w}")
        if extkey_ctr:
            print("-- ExtendedData <Data name> keys seen (schema hint) --")
            for w, c in extkey_ctr.most_common():
                print(f"   {c:4d}  {w}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="s3://bucket/prefix OR a local dir/file of .kmz/.kml")
    ap.add_argument("--endpoint-url", default=None, help="S3-compatible endpoint (e.g. MinIO)")
    ap.add_argument("--schema", action="store_true",
                    help="also print WHERE video links appear across all files (discover the schema)")
    ap.add_argument("--list-urls", action="store_true", help="print every URL found per file")
    ap.add_argument("--json", dest="json_out", default=None, help="write full per-file results to this JSON")
    args = ap.parse_args(argv)

    results = audit_target(args.target, args.endpoint_url)
    _print_report(results, schema=args.schema, list_urls=args.list_urls)

    if args.json_out:
        payload = [{"source": r.source, "has_video": r.has_video, "error": r.error,
                    "n_placemarks": r.n_placemarks, "urls": r.urls,
                    "ext_data_keys": r.ext_data_keys,
                    "video_links": [vars(v) for v in r.video_links]} for r in results]
        Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
        print(f"[ok] wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
