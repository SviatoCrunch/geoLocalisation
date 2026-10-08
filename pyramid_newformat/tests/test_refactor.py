"""Synthetic tests for the read-reduction / no-copy refactor (spec §6).

Covers: scheduler planning (dedup, cache exclusion, gap-merge budgets, byte cap, overread stats);
bounded version-keyed cache (hit/miss, eviction, identity key); reader dedup of duplicate ids,
arbitrary request order, the `out` buffer contract for read_cells_into / read_cell / read_cell_level;
true per-position coords from the manifest; and resume after a failed upload (written-but-not-uploaded
shard republishes before the manifest). No S3, no torch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pyramid_newformat import s3io                               # noqa: E402
from pyramid_newformat.cache import CellCache                    # noqa: E402
from pyramid_newformat.scheduler import plan_reads               # noqa: E402

P, LEVELS, H, W, D = 3, [1000, 900], 2, 2, 3
BPC = P * len(LEVELS) * H * W * D * 2


def _block(seed):
    return np.random.default_rng(seed).standard_normal((P, len(LEVELS), H, W, D)).astype(np.float16)


def _write_source(root: Path, n=5):
    src = root / "embeddings" / "kup" / "pyr" / "kup"
    (src / "cells").mkdir(parents=True)
    cells, blocks = {}, {}
    for i in range(n):
        cid = f"kup:{i}_lvl0"; fn = f"kup_{i}_lvl0.h5"; key = f"embeddings/kup/pyr/kup/cells/{fn}"
        blk = _block(i); blocks[cid] = blk
        with h5py.File(src / "cells" / fn, "w") as f:
            f.create_dataset("lat", data=np.arange(P) + i * 10.0)
            f.create_dataset("lon", data=np.arange(P) + i * 100.0)
            f.attrs.update({"n_positions": P, "levels_m": [float(x) for x in LEVELS], "D": D})
            for pi in range(P):
                g = f.create_group(f"p{pi}")
                for L in LEVELS:
                    g.create_dataset(f"l{int(L)}", data=blk[pi, LEVELS.index(L)])
        cells[cid] = {"key": key, "lat": 49.0 + i, "lon": 37.0 + i}
    idx = {"city": "kup", "s3_prefix": str(src),
           "config": {"D": D, "levels_m": LEVELS, "step_m": 250.0, "tile_size_m": 1000.0, "output_px": 448},
           "n_cells": n, "cells": cells}
    (src / "_index.json").write_text(json.dumps(idx))
    return str(src), blocks


def _convert(src, dst, work, cps=0, resume=False):
    from pyramid_newformat.convert import Converter
    c = Converter(src, dst, work, cells_per_shard=cps)
    return c.run(c.plan(), resume=resume)


# ---------------- scheduler ----------------
def test_plan_contiguous_single_read():
    needed = [("f.h5", 2, "c2"), ("f.h5", 0, "c0"), ("f.h5", 1, "c1")]
    reads, st = plan_reads(needed, BPC)
    assert st["n_reads"] == 1 and st["overread_bytes"] == 0          # 0,1,2 coalesce to one span
    assert reads[0].lo == 0 and reads[0].hi == 2


def test_plan_dedup_and_cache_exclusion():
    needed = [("f.h5", 0, "c0"), ("f.h5", 0, "c0"), ("f.h5", 1, "c1")]
    reads, st = plan_reads(needed, BPC, cached={"c1"})
    assert st["n_cells"] == 1 and reads[0].cells == [(0, "c0")]      # dup once, c1 cached out


def test_plan_gap_split_vs_merge():
    needed = [("f.h5", 0, "c0"), ("f.h5", 5, "c5")]                  # gap of 4 cells
    split, _ = plan_reads(needed, BPC, max_gap_cells=0)
    assert split[0].span == 1 and len(split) == 2                   # no merge → two reads
    merged, st = plan_reads(needed, BPC, max_gap_cells=4, max_overread_ratio=1.0)
    assert len(merged) == 1 and merged[0].span == 6                 # merged span covers the gap
    assert st["overread_bytes"] == 4 * BPC                          # 4 gap cells are overread


def test_plan_byte_budget_caps_read():
    needed = [("f.h5", i, f"c{i}") for i in range(10)]              # one contiguous run of 10
    reads, st = plan_reads(needed, BPC, max_read_bytes=3 * BPC)
    assert all(r.span <= 3 for r in reads)                          # no read exceeds the budget
    assert st["peak_read_bytes"] <= 3 * BPC and st["n_cells"] == 10


def test_plan_overread_ratio_blocks_bad_merge():
    needed = [("f.h5", 0, "c0"), ("f.h5", 9, "c9")]                 # merging → 8/10 overread
    reads, _ = plan_reads(needed, BPC, max_gap_cells=100, max_overread_ratio=0.5)
    assert len(reads) == 2                                           # ratio too high → not merged


# ---------------- cache ----------------
def test_cache_hit_miss_and_identity_key():
    c = CellCache(max_bytes=10 * BPC)
    k_v1 = CellCache.key("f.h5", "VER1", 3)
    k_v2 = CellCache.key("f.h5", "VER2", 3)                          # same offset, different object version
    arr = np.zeros((2, 2), np.float16)
    assert c.get(k_v1) is None                                      # miss
    c.put(k_v1, arr)
    assert c.get(k_v1) is arr                                       # hit
    assert c.get(k_v2) is None                                      # version change ≠ stale hit
    assert c.stats()["hits"] == 1 and c.stats()["misses"] == 2


def test_cache_lru_eviction_by_bytes():
    cell = np.zeros((P, len(LEVELS), H, W, D), np.float16)
    c = CellCache(max_bytes=2 * cell.nbytes)
    for i in range(3):
        c.put(CellCache.key("f.h5", None, i), cell.copy())
    assert len(c) == 2 and c.evictions == 1                        # oldest evicted
    assert CellCache.key("f.h5", None, 0) not in c
    assert c.nbytes <= 2 * cell.nbytes


def test_cache_disabled_when_zero_budget():
    c = CellCache(max_bytes=0)
    c.put(CellCache.key("f.h5", None, 0), np.zeros(4, np.float16))
    assert len(c) == 0


# ---------------- reader: dedup / order / out buffers ----------------
def test_read_cells_duplicates_read_once(tmp_path):
    from pyramid_newformat.reader import NewFormatReader
    src, blocks = _write_source(tmp_path / "s", n=3)
    dst = str(tmp_path / "d"); _convert(src, dst, str(tmp_path / "w"))
    r = NewFormatReader(dst)
    out, order = r.read_cells_into(["kup:0_lvl0", "kup:0_lvl0", "kup:1_lvl0"])
    assert order == ["kup:0_lvl0", "kup:1_lvl0"] and out.shape[0] == 2   # dup collapsed
    # request-order API still returns a row per requested id (duplicates share the one read)
    got = r.read_cells(["kup:1_lvl0", "kup:0_lvl0", "kup:0_lvl0"])
    assert np.array_equal(got[0].view(np.uint16), blocks["kup:1_lvl0"].view(np.uint16))
    assert np.array_equal(got[1].view(np.uint16), got[2].view(np.uint16))
    r.close()


def test_read_cells_into_out_buffer_reuse(tmp_path):
    from pyramid_newformat.reader import NewFormatReader
    src, blocks = _write_source(tmp_path / "s", n=3)
    dst = str(tmp_path / "d"); _convert(src, dst, str(tmp_path / "w"))
    r = NewFormatReader(dst)
    buf = np.empty((3,) + r.block_shape, np.dtype("<f2"))
    ids = ["kup:2_lvl0", "kup:0_lvl0", "kup:1_lvl0"]
    out, order = r.read_cells_into(ids, buf)
    assert out is buf                                               # filled in place, no new alloc
    for row, cid in enumerate(order):
        assert np.array_equal(buf[row].view(np.uint16), blocks[cid].view(np.uint16))
    with pytest.raises(ValueError):
        r.read_cells_into(ids, np.empty((1,) + r.block_shape, np.dtype("<f2")))   # too small
    r.close()


def test_read_cell_and_level_out_contract(tmp_path):
    from pyramid_newformat.reader import NewFormatReader
    src, blocks = _write_source(tmp_path / "s", n=2)
    dst = str(tmp_path / "d"); _convert(src, dst, str(tmp_path / "w"))
    r = NewFormatReader(dst)
    ob = np.empty(r.block_shape, np.dtype("<f2"))
    assert r.read_cell("kup:1_lvl0", ob) is ob
    assert np.array_equal(ob.view(np.uint16), blocks["kup:1_lvl0"].view(np.uint16))
    ol = np.empty((P, H, W, D), np.dtype("<f2"))
    r.read_cell_level("kup:1_lvl0", 900, ol)
    assert np.array_equal(ol.view(np.uint16), blocks["kup:1_lvl0"][:, 1].view(np.uint16))
    r.close()


def test_true_coords_from_manifest(tmp_path):
    from pyramid_newformat.reader import NewFormatReader
    src, _ = _write_source(tmp_path / "s", n=3)
    dst = str(tmp_path / "d"); man = _convert(src, dst, str(tmp_path / "w"))
    # manifest carries per-position coords (not cell-center fakes)
    assert man["cells"]["kup:2_lvl0"]["pos_lat"] == (np.arange(P) + 20.0).tolist()
    r = NewFormatReader(dst)
    lat, lon = r.read_cell_coords("kup:2_lvl0")
    assert np.allclose(lat, np.arange(P) + 20.0) and np.allclose(lon, np.arange(P) + 200.0)
    assert lat.shape == (P,) and not np.allclose(lat, lat[0])       # NOT a constant cell-center
    r.close()


# ---------------- resume after a failed upload ----------------
def test_resume_after_failed_upload(tmp_path):
    """Shard written+verified locally but manifest/files not recorded (upload died) → resume must
    republish that shard and only then write the manifest."""
    from pyramid_newformat.reader import NewFormatReader
    src, blocks = _write_source(tmp_path / "s", n=3)
    dst = str(tmp_path / "d"); work = str(tmp_path / "w")
    _convert(src, dst, work, cps=2)
    man = Path(dst) / "manifest.json"
    # simulate death right after the LAST shard's bytes were written but before it was published:
    ck = json.loads((Path(work) / "_convert_checkpoint.json").read_text())
    last = "features_00001.h5"
    ck["files"].pop(last, None)                                    # forget the upload record
    ck.setdefault("written", {})[last] = {"sha256": "stale"}       # keep written state
    (Path(work) / "_convert_checkpoint.json").write_text(json.dumps(ck))
    man.unlink()                                                   # manifest not yet written
    out = _convert(src, dst, work, cps=2, resume=True)             # resume → finish publish + manifest
    assert man.exists() and last in {f["name"] for f in out["files"]}
    r = NewFormatReader(dst)
    for cid, blk in blocks.items():
        assert np.array_equal(r.read_cell(cid).view(np.uint16), blk.view(np.uint16))
    r.close()


# ---------------- bench (local, cpu) ----------------
def test_bench_counters_and_variants(tmp_path):
    from pyramid_newformat.bench import bench
    src, _ = _write_source(tmp_path / "s", n=4)
    dst = str(tmp_path / "d"); work = str(tmp_path / "w")
    _convert(src, dst, work, cps=2)
    res = bench(src, dst, work, n_cells=4, device="cpu", run_old=True, out=str(tmp_path / "b.json"))
    assert set(res) >= {"old", "adjacent", "merge", "span", "meta"}
    for mode in ("adjacent", "merge", "span"):
        v = res[mode]
        assert v["n_cells"] == 4 and v["get"] == v["n_reads"]        # one physical read per planned range
        assert v["got_bytes"] == v["useful_bytes"] + v["overread_bytes"]   # bytes accounting is consistent
    # span mode reads whole shards (2 cells each) → 2 reads; adjacent coalesces contiguous → ≤ 2 too
    assert res["span"]["n_reads"] == 2
    assert (tmp_path / "b.json").exists()


def test_bench_cache_wired(tmp_path):
    from pyramid_newformat.bench import _new_variant
    from pyramid_newformat.reader import NewFormatReader
    src, _ = _write_source(tmp_path / "s", n=4)
    dst = str(tmp_path / "d"); _convert(src, dst, str(tmp_path / "w"), cps=4)
    r = NewFormatReader(dst)
    bpc = int(r.files[next(iter(r.files))]["bytes_per_cell"]); cids = list(r.cells)[:4]; r.close()
    # within one plan every cell is unique (plan dedups), so the cache fills but does not self-hit;
    # cross-call hits are covered by test_cache_hit_miss_and_identity_key.
    res = _new_variant(dst, cids, "adjacent", bpc, do_gpu=False, cache_bytes=10 * bpc)
    assert res["cache"]["n_cached"] == 4 and res["cache"]["hits"] == 0


def test_resume_refuses_zeros_when_shard_missing(tmp_path):
    """If the local shard is gone but cells are marked done and not uploaded, refuse (don't ship zeros)."""
    src, _ = _write_source(tmp_path / "s", n=3)
    dst = str(tmp_path / "d"); work = str(tmp_path / "w")
    _convert(src, dst, work, cps=2)
    ck = json.loads((Path(work) / "_convert_checkpoint.json").read_text())
    last = "features_00001.h5"
    ck["files"].pop(last, None)
    (Path(work) / "_convert_checkpoint.json").write_text(json.dumps(ck))
    (Path(work) / last).unlink()                                   # local shard removed
    (Path(dst) / "manifest.json").unlink()
    with pytest.raises(SystemExit):
        _convert(src, dst, work, cps=2, resume=True)
