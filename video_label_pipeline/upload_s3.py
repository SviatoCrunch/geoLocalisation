"""Upload selected frames from a video_label_pipeline manifest to an S3 catalog.

Reads ``manifest.json`` (each sub-video has ``selected`` entries with a ``rank`` and an on-disk
``path``) and uploads the **top-K by rank** of every sub-video to an S3 prefix, mirroring the
``<station>/<date>/<chunk>/<sub_id>/<file>`` tree. Because the pipeline's farthest-point selection is
prefix-stable, ONE manifest built with ``--k <maxN>`` feeds BOTH densities: run this with ``--k N`` for
the dense catalog and ``--k 5`` for the sparse one, into two different ``--s3`` prefixes.

Idempotent: existing keys under the prefix are listed once and skipped, so re-running after adding new
days only uploads the new frames. The selection/key logic is pure (:func:`iter_uploads`) and unit-tested
with a fake uploader; only the CLI needs boto3.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _split_s3(uri: str):
    if not uri.startswith("s3://"):
        raise SystemExit(f"--s3 must be s3://bucket/prefix, got {uri!r}")
    b, _, k = uri[5:].partition("/")
    return b, k.strip("/")


def iter_uploads(manifest: dict, k: int, prefix_key: str):
    """Yield (local_path, s3_key) for the top-``k`` ranked frames of every sub-video (pure, testable).

    s3_key = ``<prefix_key>/<station>/<date>/<chunk>/<sub_id>/<file>``. Entries without an on-disk
    ``path`` or ``file`` are skipped (caller sees them via the returned/logged missing count only if it
    tracks them; here we simply don't yield them)."""
    pk = prefix_key.strip("/")
    for v in manifest.get("videos", []):
        station, date, chunk = str(v.get("station")), str(v.get("date")), str(v.get("chunk"))
        for sv in v.get("subvideos", []):
            sub_id = str(sv.get("sub_id"))
            chosen = sorted(sv.get("selected", []), key=lambda s: s.get("rank", 1_000_000))[:k]
            for s in chosen:
                path, fname = s.get("path"), s.get("file")
                if not path or not fname:
                    continue
                key = "/".join([p for p in (pk, station, date, chunk, sub_id, fname) if p])
                yield path, key


def upload(manifest: dict, s3_uri: str, k: int, *, put, existing=None, dry_run=False, log=print) -> dict:
    """Upload top-k frames to ``s3_uri``. ``put(local_path, key)`` does the transfer; ``existing`` is a
    set of already-present keys (skipped). Returns {uploaded, skipped, missing}."""
    bucket, prefix_key = _split_s3(s3_uri)
    existing = existing or set()
    up = skip = missing = 0
    for path, key in iter_uploads(manifest, k, prefix_key):
        if not Path(path).exists():
            missing += 1
            continue
        if key in existing:
            skip += 1
            continue
        if not dry_run:
            put(path, bucket, key)
        up += 1
        if up % 200 == 0:
            log(f"[upload] {up} sent ({skip} skipped, {missing} missing) ...")
    log(f"[done] k={k} -> {s3_uri}: uploaded={up} skipped={skip} missing_on_disk={missing}")
    return {"uploaded": up, "skipped": skip, "missing": missing}


def _list_existing(bucket: str, prefix_key: str) -> set:
    import boto3
    s3 = boto3.client("s3")
    keys, token = set(), None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix_key + "/" if prefix_key else ""}
        if token:
            kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        for o in r.get("Contents", []):
            keys.add(o["Key"])
        if not r.get("IsTruncated"):
            break
        token = r.get("NextContinuationToken")
    return keys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="video_label_pipeline manifest.json")
    ap.add_argument("--s3", required=True, help="target prefix, e.g. s3://geo-reference/frames/dense")
    ap.add_argument("--k", type=int, required=True, help="frames per sub-video (top-K by rank)")
    ap.add_argument("--dry-run", action="store_true", help="count only; no upload, no S3 listing")
    ap.add_argument("--no-skip", action="store_true", help="don't list existing keys (re-upload all)")
    args = ap.parse_args(argv)

    manifest = json.loads(Path(args.manifest).expanduser().read_text(encoding="utf-8"))
    kps = manifest.get("k_per_subvideo")
    if kps is not None and args.k > int(kps):
        print(f"[warn] manifest k_per_subvideo={kps} < --k {args.k}: only {kps} frames/sub exist; "
              f"regenerate the manifest with --k {args.k} for a true dense catalog.", flush=True)

    bucket, prefix_key = _split_s3(args.s3)
    existing = set()
    if not args.dry_run and not args.no_skip:
        existing = _list_existing(bucket, prefix_key)
        print(f"[s3] {len(existing)} existing objects under {args.s3} (will skip)", flush=True)

    put = None
    if not args.dry_run:
        import boto3
        s3 = boto3.client("s3")
        def put(local, b, key):                                   # noqa: E306
            s3.upload_file(local, b, key)

    upload(manifest, args.s3, args.k, put=put, existing=existing, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
