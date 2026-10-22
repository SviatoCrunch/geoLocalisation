"""Location-agnostic IO: a 'uri' is either ``s3://bucket/key`` or a local filesystem path.

Lets the converter/reader/tests run identically against S3 or a local directory (synthetic tests use
local; production uses S3). boto3 is imported lazily so local-only use needs no AWS deps. S3 reads can
pin a specific object ``VersionId`` so a manifest byte-offset stays valid against an immutable object.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

_client = None

# Stream large S3 Range GETs in chunks and log progress (GB + MB/s). Off by default (zero overhead);
# set NF_S3_PROGRESS=1 to see long reads tick instead of looking hung. Chunk size via NF_S3_CHUNK_MB.
_S3_PROGRESS = os.environ.get("NF_S3_PROGRESS", "") not in ("", "0")
_S3_CHUNK = int(os.environ.get("NF_S3_CHUNK_MB", "64")) << 20
_S3_PROGRESS_MIN = int(os.environ.get("NF_S3_PROGRESS_MIN_MB", "512")) << 20

# ---- optional IO counters (bench.py turns these on; zero overhead otherwise) ----
COUNTERS = {"get": 0, "head": 0, "retry": 0, "get_bytes": 0, "range_calls": 0}
_count = False


def enable_counters(on: bool = True):
    global _count
    _count = bool(on)
    reset_counters()


def reset_counters():
    for k in COUNTERS:
        COUNTERS[k] = 0


def _bump(k, v=1):
    if _count:
        COUNTERS[k] += v


def is_s3(uri: str) -> bool:
    return str(uri).startswith("s3://")


def split_s3(uri: str):
    b, _, k = uri[5:].partition("/")
    return b, k


def client():
    global _client
    if _client is None:
        import boto3
        _client = boto3.client("s3")
        # count real botocore retries (fires per retry attempt) — the one signal we can't infer locally
        _client.meta.events.register("needs-retry.s3.GetObject", lambda **k: _bump("retry"))
        _client.meta.events.register("needs-retry.s3.HeadObject", lambda **k: _bump("retry"))
    return _client


def join(uri: str, *parts: str) -> str:
    uri = str(uri).rstrip("/")
    return uri + "/" + "/".join(p.strip("/") for p in parts)


def exists(uri: str) -> bool:
    if is_s3(uri):
        b, k = split_s3(uri)
        try:
            client().head_object(Bucket=b, Key=k); return True
        except Exception:
            return False
    return Path(uri).exists()


def size(uri: str) -> int:
    if is_s3(uri):
        b, k = split_s3(uri)
        return int(client().head_object(Bucket=b, Key=k)["ContentLength"])
    return Path(uri).stat().st_size


def version_id(uri: str):
    """S3 object VersionId (None if unversioned/local) — to bind byte-offset reads to an immutable obj."""
    if not is_s3(uri):
        return None
    b, k = split_s3(uri)
    _bump("head")
    return client().head_object(Bucket=b, Key=k).get("VersionId")


def head_identity(uri: str):
    """(version_id, etag) for an uploaded object — recorded in the manifest so readers pin identity
    WITHOUT a per-open HEAD. Local → (None, None). ETag quotes are preserved for an IfMatch header."""
    if not is_s3(uri):
        return None, None
    b, k = split_s3(uri)
    _bump("head")
    h = client().head_object(Bucket=b, Key=k)
    return h.get("VersionId"), h.get("ETag")


def read_json(uri: str) -> dict:
    if is_s3(uri):
        b, k = split_s3(uri)
        return json.loads(client().get_object(Bucket=b, Key=k)["Body"].read())
    return json.loads(Path(uri).read_text(encoding="utf-8"))


def write_json(uri: str, obj) -> None:
    body = json.dumps(obj, indent=2).encode("utf-8")
    if is_s3(uri):
        b, k = split_s3(uri)
        client().put_object(Bucket=b, Key=k, Body=body, ContentType="application/json")
    else:
        p = Path(uri); p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(body)


def download(uri: str, local: str) -> str:
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    if is_s3(uri):
        b, k = split_s3(uri)
        client().download_file(b, k, str(local))
    else:
        if str(Path(uri).resolve()) != str(Path(local).resolve()):
            shutil.copyfile(uri, local)
    return str(local)


def upload(local: str, uri: str) -> None:
    if is_s3(uri):
        b, k = split_s3(uri)
        client().upload_file(str(local), b, k)
    else:
        p = Path(uri); p.parent.mkdir(parents=True, exist_ok=True)
        if str(Path(local).resolve()) != str(p.resolve()):
            shutil.copyfile(local, p)


def range_get(uri: str, offset: int, length: int, version: str | None = None,
              etag: str | None = None) -> bytes:
    """Read ``length`` bytes starting at ``offset`` (inclusive). S3 Range GET or local pread.

    S3 (GetObject docs): one GET = one contiguous range. ``version`` pins an immutable object version;
    for unversioned buckets ``etag`` sets ``IfMatch`` so a replaced object fails loudly instead of
    returning mismatched bytes against the manifest offset."""
    _bump("range_calls"); _bump("get"); _bump("get_bytes", length)
    if is_s3(uri):
        b, k = split_s3(uri)
        kw = {"Bucket": b, "Key": k, "Range": f"bytes={offset}-{offset + length - 1}"}
        if version:
            kw["VersionId"] = version
        elif etag:
            kw["IfMatch"] = etag
        body = client().get_object(**kw)["Body"]
        if not (_S3_PROGRESS and length >= _S3_PROGRESS_MIN):
            return body.read()
        # chunked read with per-GB progress so a multi-GB Range GET visibly ticks (and a stall is obvious)
        buf = bytearray(length); mv = memoryview(buf); done = 0
        t0 = time.time(); next_mark = 1 << 30
        print(f"[s3] GET {k} range {length/1e9:.2f} GB @off {offset} ...", flush=True)
        while done < length:
            chunk = body.read(min(_S3_CHUNK, length - done))
            if not chunk:
                break
            mv[done:done + len(chunk)] = chunk; done += len(chunk)
            if done >= next_mark:
                el = time.time() - t0
                print(f"[s3]   {done/1e9:.1f}/{length/1e9:.1f} GB  {done/1e6/max(el, 1e-9):.0f} MB/s", flush=True)
                next_mark += 1 << 30
        el = time.time() - t0
        print(f"[s3] done {done/1e9:.2f} GB in {el:.0f}s ({done/1e6/max(el, 1e-9):.0f} MB/s)", flush=True)
        if done != length:
            raise IOError(f"short S3 read: got {done} of {length} bytes for {k}")
        return bytes(buf)
    with open(uri, "rb") as f:
        f.seek(offset)
        return f.read(length)
