"""Tests for the S3 catalog uploader (pure selection + idempotent skip; no boto3/S3)."""
from __future__ import annotations

from pathlib import Path

from video_label_pipeline.upload_s3 import iter_uploads, upload


def _manifest(tmp: Path) -> dict:
    # one video, one sub-video, 6 ranked frames with real files on disk
    paths = []
    for i in range(6):
        p = tmp / f"f{i}.jpg"; p.write_bytes(b"x"); paths.append(str(p))
    selected = [{"rank": i + 1, "file": f"f{i}.jpg", "path": paths[i]} for i in range(6)]
    return {"k_per_subvideo": 6, "videos": [{
        "station": "104", "date": "10.10.2026", "chunk": "chunk_104_x",
        "subvideos": [{"sub_id": "sub_01", "selected": selected}]}]}


def test_iter_uploads_topk_and_key(tmp_path):
    m = _manifest(tmp_path)
    ups = list(iter_uploads(m, k=3, prefix_key="frames/dense"))
    assert len(ups) == 3                                           # top-3 by rank
    _, key0 = ups[0]
    assert key0 == "frames/dense/104/10.10.2026/chunk_104_x/sub_01/f0.jpg"
    # ranks are 1,2,3 → files f0,f1,f2
    assert [Path(k).name for _, k in ups] == ["f0.jpg", "f1.jpg", "f2.jpg"]


def test_sparse_is_prefix_of_dense(tmp_path):
    m = _manifest(tmp_path)
    dense = [k for _, k in iter_uploads(m, k=5, prefix_key="d")]
    sparse = [k for _, k in iter_uploads(m, k=2, prefix_key="d")]
    assert sparse == dense[:2]                                     # same ranks → sparse ⊂ dense


def test_upload_counts_and_skip(tmp_path):
    m = _manifest(tmp_path)
    sent = []
    put = lambda local, b, key: sent.append(key)                  # noqa: E731
    # first run: 4 frames, nothing existing
    r1 = upload(m, "s3://bkt/frames/dense", k=4, put=put, existing=set(), log=lambda *a: None)
    assert r1 == {"uploaded": 4, "skipped": 0, "missing": 0}
    assert len(sent) == 4
    # second run: those 4 now exist → all skipped (idempotent)
    r2 = upload(m, "s3://bkt/frames/dense", k=4, put=put, existing=set(sent), log=lambda *a: None)
    assert r2["uploaded"] == 0 and r2["skipped"] == 4


def test_missing_on_disk_counted(tmp_path):
    m = _manifest(tmp_path)
    m["videos"][0]["subvideos"][0]["selected"][0]["path"] = str(tmp_path / "gone.jpg")
    r = upload(m, "s3://bkt/x", k=6, put=lambda *a: None, existing=set(), log=lambda *a: None)
    assert r["missing"] == 1 and r["uploaded"] == 5


def test_dry_run_no_put(tmp_path):
    m = _manifest(tmp_path)
    calls = []
    upload(m, "s3://bkt/x", k=3, put=lambda *a: calls.append(a), existing=set(), dry_run=True,
           log=lambda *a: None)
    assert calls == []                                            # dry-run never calls put
