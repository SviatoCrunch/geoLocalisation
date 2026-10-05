"""Download a LOCAL subset of the S3 pyramid store for offline runs (no heredoc, no key bugs).

Pulls ``_index.json`` + only the cells in the shortlist union (top ``--k-coarse`` of the first
``--max-queries`` queries) into ``--dest``, preserving each cell's manifest ``key`` (which is the
FULL S3 object key — CellStoreS3 uses it directly), and writes a PRUNED ``_index.json`` listing only
the downloaded cells. The result is a directory LocalCellStore (evaluate --local-store) reads offline.

    python -m magsacpp_torch.fetch_local \
        --index-uri s3://…/<city>/_index.json --shortlist shortlist.json \
        --dest /home/ubuntu/work/kram_store_local --k-coarse 30 --max-queries 20
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index-uri", required=True, help="s3://…/<city>/_index.json")
    ap.add_argument("--shortlist", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--k-coarse", type=int, default=30, dest="k_coarse")
    ap.add_argument("--max-queries", type=int, default=0, dest="max_queries")
    ap.add_argument("--only-city", default=None, dest="only_city")
    a = ap.parse_args(argv)

    import boto3
    u = urllib.parse.urlparse(a.index_uri)
    bucket, ikey = u.netloc, u.path.lstrip("/")
    s3 = boto3.client("s3")
    idx = json.loads(s3.get_object(Bucket=bucket, Key=ikey)["Body"].read())
    sj = json.loads(Path(a.shortlist).expanduser().read_text())["shortlist"]

    need, n = set(), 0
    for q, e in sj.items():
        if a.only_city and q.split(":", 1)[0] != a.only_city:
            continue
        if a.max_queries and n >= a.max_queries:
            break
        n += 1
        need |= {c for c in e["cells"][:a.k_coarse] if c in idx["cells"]}

    dest = Path(a.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    pruned = {"config": idx.get("config", {}), "cells": {}}
    need = sorted(need)
    print(f"[fetch_local] queries={n} cells={len(need)} dest={dest}", flush=True)
    for i, c in enumerate(need, 1):
        key = idx["cells"][c]["key"]                 # FULL S3 key (used directly, like CellStoreS3)
        dst = dest / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            s3.download_file(bucket, key, str(dst))
        pruned["cells"][c] = idx["cells"][c]
        if i % 20 == 0 or i == len(need):
            print(f"  {i}/{len(need)}", flush=True)
    (dest / "_index.json").write_text(json.dumps(pruned))
    print(f"[ok] LOCAL STORE READY: {dest}  cells={len(pruned['cells'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
