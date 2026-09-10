"""Unit tests for s3_gt_sync.core — GT-name resolution + pull/push with a
fake in-memory S3 client (no network, no moto)."""
from __future__ import annotations

import os

import pytest

from s3_gt_sync import core


# ── fake S3 ───────────────────────────────────────────────────────────────────

class _Paginator:
    def __init__(self, store):
        self._store = store

    def paginate(self, Bucket, Prefix=""):
        contents = [{"Key": k, "Size": len(v)}
                    for k, v in sorted(self._store.get(Bucket, {}).items())
                    if k.startswith(Prefix)]
        # split into two pages to exercise pagination
        mid = len(contents) // 2 or len(contents)
        yield {"Contents": contents[:mid]}
        if contents[mid:]:
            yield {"Contents": contents[mid:]}


class FakeS3:
    """Minimal stand-in: objects live in ``store[bucket][key] = bytes``."""

    def __init__(self, store):
        self.store = store
        self.uploaded: list[tuple[str, str, int]] = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _Paginator(self.store)

    def download_file(self, Bucket, Key, Filename):
        data = self.store[Bucket][Key]
        with open(Filename, "wb") as fh:
            fh.write(data)

    def head_object(self, Bucket, Key):
        if Key not in self.store.get(Bucket, {}):
            raise KeyError(Key)
        return {"ContentLength": len(self.store[Bucket][Key])}

    def upload_file(self, Filename, Bucket, Key, ExtraArgs=None):
        with open(Filename, "rb") as fh:
            data = fh.read()
        self.store.setdefault(Bucket, {})[Key] = data
        self.uploaded.append((Bucket, Key, len(data)))


@pytest.fixture
def patch_client(monkeypatch):
    def _install(store):
        fake = FakeS3(store)
        monkeypatch.setattr(core, "make_s3_client", lambda endpoint_url=None: fake)
        return fake
    return _install


# ── parse_s3_uri ──────────────────────────────────────────────────────────────

def test_parse_s3_uri_ok():
    assert core.parse_s3_uri("s3://bucket/a/b/") == ("bucket", "a/b")
    assert core.parse_s3_uri("s3://bucket") == ("bucket", "")


@pytest.mark.parametrize("bad", ["", "http://x/y", "s3:///nobucket", "/local/path"])
def test_parse_s3_uri_bad(bad):
    with pytest.raises(ValueError):
        core.parse_s3_uri(bad)


# ── GT-name resolution ────────────────────────────────────────────────────────

def test_resolve_plain_paired_and_screenshot_not_copied():
    objs = [
        ("p/40/2.0/1.jpg", 10),
        ("p/40/2.0/1_48.5844637615,37.6091680481.jpg", 20),
    ]
    targets, unpaired = core.resolve_folder_targets(objs, core.DEFAULT_EXTS, 10)
    assert unpaired == []
    assert len(targets) == 1
    t = targets[0]
    assert t.src_key == "p/40/2.0/1.jpg"        # the frame, not the screenshot
    assert t.dst_name == "1_48.5844637615_37.6091680481.jpg"
    assert t.size == 10


def test_resolve_done_copied_verbatim():
    objs = [("p/2.0/9_48.59_37.57.jpg", 5)]
    targets, unpaired = core.resolve_folder_targets(objs, core.DEFAULT_EXTS, 10)
    assert unpaired == []
    assert targets[0].dst_name == "9_48.59_37.57.jpg"
    assert targets[0].src_key == "p/2.0/9_48.59_37.57.jpg"


def test_resolve_plain_without_screenshot_is_unpaired():
    objs = [("p/2.0/7.jpg", 5)]
    targets, unpaired = core.resolve_folder_targets(objs, core.DEFAULT_EXTS, 10)
    assert targets == []
    assert unpaired == ["p/2.0/7.jpg"]


def test_lone_screenshot_never_emitted():
    # leftover N_lat,lon with no plain N -> coordinate source only, dropped
    objs = [("p/2.3/1_48.87,37.76.jpg", 5)]
    targets, unpaired = core.resolve_folder_targets(objs, core.DEFAULT_EXTS, 10)
    assert targets == []
    assert unpaired == []


# ── plan_pull: exclusion + folder grouping ────────────────────────────────────

def test_plan_pull_excludes_subdir(patch_client):
    store = {"bkt": {
        "root/keep/1.jpg": b"x" * 3,
        "root/keep/1_48.1,37.1.jpg": b"y" * 4,
        "root/raw/2.jpg": b"z" * 3,
        "root/raw/2_48.2,37.2.jpg": b"w" * 4,
    }}
    patch_client(store)
    targets, unpaired = core.plan_pull("s3://bkt/root", exclude={"raw"})
    names = {t.dst_name for t in targets}
    assert names == {"1_48.1_37.1.jpg"}
    assert unpaired == []


def test_plan_pull_always_excludes_output_subdirs(patch_client):
    # The tool's own output (GT_flat / GT_flat_mask) is skipped even without
    # passing --exclude, so a sync never re-ingests what it pushed.
    store = {"bkt": {
        "root/src/1.jpg": b"x" * 3,
        "root/src/1_48.1,37.1.jpg": b"y" * 4,
        "root/GT_flat/9_48.9_37.9.jpg": b"aaa",
        "root/GT_flat_mask/9_48.9_37.9__road.png": b"bbb",
    }}
    patch_client(store)
    targets, _ = core.plan_pull("s3://bkt/root")
    names = {t.dst_name for t in targets}
    assert names == {"1_48.1_37.1.jpg"}  # GT_flat / GT_flat_mask contents dropped


# ── pull: skip-existing + conflicts ───────────────────────────────────────────

def test_pull_downloads_and_skips_existing(patch_client, tmp_path):
    store = {"bkt": {
        "root/a/1.jpg": b"abc",
        "root/a/1_48.1,37.1.jpg": b"screenshot",
    }}
    patch_client(store)
    dest = tmp_path / "out"

    s1 = core.pull("s3://bkt/root", str(dest))
    assert s1["downloaded"] == 1 and s1["skipped"] == 0
    out_file = dest / "1_48.1_37.1.jpg"
    assert out_file.read_bytes() == b"abc"

    # second run: same-size file present -> skipped, nothing re-fetched
    s2 = core.pull("s3://bkt/root", str(dest))
    assert s2["downloaded"] == 0 and s2["skipped"] == 1


def test_pull_dry_run_writes_nothing(patch_client, tmp_path):
    store = {"bkt": {"root/a/1.jpg": b"abc", "root/a/1_1.0,2.0.jpg": b"s"}}
    patch_client(store)
    dest = tmp_path / "out"
    s = core.pull("s3://bkt/root", str(dest), dry_run=True)
    assert s["downloaded"] == 1
    assert not dest.exists()


def test_pull_detects_cross_folder_name_conflict(patch_client, tmp_path):
    store = {"bkt": {
        "root/cityA/1_48.1_37.1.jpg": b"aaa",
        "root/cityB/1_48.1_37.1.jpg": b"bbb",
    }}
    patch_client(store)
    dest = tmp_path / "out"
    s = core.pull("s3://bkt/root", str(dest))
    assert s["downloaded"] == 1
    assert s["conflicts"] == 1
    assert (dest / "1_48.1_37.1.jpg").exists()


# ── push ──────────────────────────────────────────────────────────────────────

def test_push_uploads_and_skips_existing(patch_client, tmp_path):
    store = {"bkt": {}}
    fake = patch_client(store)
    src = tmp_path / "local"
    (src / "sub").mkdir(parents=True)
    (src / "a.jpg").write_bytes(b"one")
    (src / "sub" / "b.png").write_bytes(b"two")

    s1 = core.push(str(src), "s3://bkt/dest")
    assert s1["uploaded"] == 2 and s1["skipped"] == 0
    assert "dest/a.jpg" in store["bkt"]
    assert "dest/sub/b.png" in store["bkt"]

    s2 = core.push(str(src), "s3://bkt/dest")
    assert s2["uploaded"] == 0 and s2["skipped"] == 2


def test_push_dry_run_uploads_nothing(patch_client, tmp_path):
    store = {"bkt": {}}
    fake = patch_client(store)
    src = tmp_path / "local"
    src.mkdir()
    (src / "a.jpg").write_bytes(b"one")

    s = core.push(str(src), "s3://bkt/dest", dry_run=True)
    assert s["uploaded"] == 1
    assert store["bkt"] == {}
    assert fake.uploaded == []
