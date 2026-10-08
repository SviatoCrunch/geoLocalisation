"""Synthetic end-to-end tests for the packed pyramid-embedding format (local, no S3, no torch).

Builds a tiny source per-cell store (+_index.json) in the '.../embeddings/...' layout, converts it,
and checks: deterministic mapping, numeric-position/explicit-level order, coords, bitwise float16
repack, shard boundaries, byte-offset Range read, resume idempotency, and structural errors.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

P, LEVELS, H, W, D = 3, [1000, 900], 2, 2, 3


def _cell_block(seed):                       # deterministic known values, fp16
    rng = np.random.default_rng(seed)
    return rng.standard_normal((P, len(LEVELS), H, W, D)).astype(np.float16)


def _write_source(root: Path, n=3, break_cell=None):
    """root/embeddings/kup/pyr/kup/{_index.json, cells/*.h5}. Returns source_prefix + blocks."""
    src = root / "embeddings" / "kup" / "pyr" / "kup"
    (src / "cells").mkdir(parents=True)
    cells = {}; blocks = {}
    for i in range(n):
        cid = f"kup:{i}_lvl0"; fn = f"kup_{i}_lvl0.h5"; key = f"embeddings/kup/pyr/kup/cells/{fn}"
        blk = _cell_block(i); blocks[cid] = blk
        p = src / "cells" / fn
        with h5py.File(p, "w") as f:
            f.create_dataset("lat", data=np.arange(P) + i * 10.0)
            f.create_dataset("lon", data=np.arange(P) + i * 100.0)
            f.create_dataset("px", data=np.zeros(P)); f.create_dataset("py", data=np.zeros(P))  # decoy non-pos ds
            f.attrs.update({"n_positions": P, "levels_m": [float(x) for x in LEVELS], "D": D,
                            "cell_id": cid, "cell_lat": 49.0 + i, "cell_lon": 37.0 + i})
            f.create_group("p_meta")                                 # decoy group: must be ignored
            order = list(range(P))
            for pi in order:
                g = f.create_group(f"p{pi}")
                for li, L in enumerate(LEVELS):
                    data = blk[pi, li]
                    if break_cell == cid and pi == 0 and L == 900:
                        continue                                     # missing level -> error case
                    g.create_dataset(f"l{int(L)}", data=data)
        cells[cid] = {"key": key, "lat": 49.0 + i, "lon": 37.0 + i, "n_pos": P,
                      "bytes": p.stat().st_size}
    idx = {"city": "kup", "s3_prefix": str(src),
           "config": {"model": "dinov2_vitg14", "patch": 14, "D": D, "n_prefix": 0, "step_m": 250.0,
                      "levels_m": LEVELS, "tile_size_m": 1000.0, "output_px": 448},
           "n_cells": n, "cells": cells}
    (src / "_index.json").write_text(json.dumps(idx))
    return str(src), blocks


def _convert(src, dst, work, cps=0, resume=False, max_cells=0):
    from pyramid_newformat.convert import Converter
    c = Converter(src, dst, work, cells_per_shard=cps)
    return c.run(c.plan(max_cells=max_cells), resume=resume)


def test_single_file_bitwise_and_mapping(tmp_path):
    from pyramid_newformat.reader import NewFormatReader
    src, blocks = _write_source(tmp_path / "s")
    dst = str(tmp_path / "d"); work = str(tmp_path / "w")
    man = _convert(src, dst, work)
    assert man["n_cells"] == 3 and man["n_positions"] == P and man["levels_m"] == LEVELS
    assert man["grid_hw"] == [H, W] and man["feature_dim"] == D
    r = NewFormatReader(dst)
    for cid, blk in blocks.items():
        got = r.read_cell(cid)
        assert got.shape == (P, len(LEVELS), H, W, D)
        assert np.array_equal(got.view(np.uint16), blk.view(np.uint16))         # bitwise fp16
        m = man["cells"][cid]
        assert m["file"] == "features.h5" and m["source_key"].endswith(f"{cid.replace(':','_')}.h5".replace("kup_", "kup_"))
    # explicit level order + numeric positions preserved
    cid0 = "kup:0_lvl0"
    assert np.array_equal(r.read_cell_level(cid0, 1000).view(np.uint16), blocks[cid0][:, 0].view(np.uint16))
    assert np.array_equal(r.read_cell_level(cid0, 900).view(np.uint16), blocks[cid0][:, 1].view(np.uint16))
    # coords
    with h5py.File(str(Path(dst) / "features.h5"), "r") as f:
        assert np.allclose(f["position_lat"][0], np.arange(P) + 0.0)
        assert np.allclose(f["cell_lat"][0], 49.0)
    r.close()


def test_sharding_boundary_and_order(tmp_path):
    from pyramid_newformat.reader import NewFormatReader
    src, blocks = _write_source(tmp_path / "s", n=3)
    dst = str(tmp_path / "d"); work = str(tmp_path / "w")
    man = _convert(src, dst, work, cps=2)
    assert len(man["files"]) == 2
    assert man["cells"]["kup:2_lvl0"]["file"] == "features_00001.h5"
    assert man["cells"]["kup:2_lvl0"]["local_cell_idx"] == 0
    r = NewFormatReader(dst)
    req = ["kup:2_lvl0", "kup:0_lvl0", "kup:1_lvl0"]          # cross-file, out of order
    got = r.read_cells(req)
    for blk, cid in zip(got, req):
        assert np.array_equal(blk.view(np.uint16), blocks[cid].view(np.uint16))
    r.close()


def test_byte_offset_range_read(tmp_path):
    from pyramid_newformat import s3io
    from pyramid_newformat.reader import NewFormatReader
    src, blocks = _write_source(tmp_path / "s")
    dst = str(tmp_path / "d"); work = str(tmp_path / "w")
    man = _convert(src, dst, work)
    r = NewFormatReader(dst)
    fm = man["files"][0]
    for cid, blk in blocks.items():
        li = man["cells"][cid]["local_cell_idx"]
        off = fm["features_byte_offset"] + li * fm["bytes_per_cell"]
        raw = s3io.range_get(s3io.join(dst, fm["name"]), off, fm["bytes_per_cell"])
        got = np.frombuffer(raw, "<f2").reshape((P, len(LEVELS), H, W, D))
        assert np.array_equal(got.view(np.uint16), blk.view(np.uint16))         # offset is correct & contiguous
    r.close()


def test_resume_idempotent(tmp_path):
    from pyramid_newformat.reader import NewFormatReader
    src, blocks = _write_source(tmp_path / "s", n=3)
    dst = str(tmp_path / "d"); work = str(tmp_path / "w")
    _convert(src, dst, work, cps=2)
    (Path(dst) / "manifest.json").unlink()                   # drop final manifest, keep checkpoint+shards
    man = _convert(src, dst, work, cps=2, resume=True)        # resume -> shards complete, manifest rewritten
    r = NewFormatReader(dst)
    for cid, blk in blocks.items():
        assert np.array_equal(r.read_cell(cid).view(np.uint16), blk.view(np.uint16))
    r.close()


def test_loader_chunks_and_store(tmp_path):
    from pyramid_newformat.loader import ChunkedCellLoader, NewFormatCellStore
    src, blocks = _write_source(tmp_path / "s", n=3)
    dst = str(tmp_path / "d"); work = str(tmp_path / "w")
    man = _convert(src, dst, work, cps=2)
    # resolve: int global_cell_idx and str cell_id both -> cell_id
    ld = ChunkedCellLoader(dst, chunk_cells=2, to_device=False)
    assert ld.resolve([0, "kup:1_lvl0", 2]) == ["kup:0_lvl0", "kup:1_lvl0", "kup:2_lvl0"]
    # chunked iteration: request order preserved, bitwise, 2 then 1
    seen = []
    for sub, arr in ld.iter_chunks([2, 0, 1]):          # out-of-order, cross-shard
        assert arr.dtype == np.float16
        for k, cid in enumerate(sub):
            assert np.array_equal(arr[k].view(np.uint16), blocks[cid].view(np.uint16))
        seen.append(tuple(sub))
    assert seen == [("kup:2_lvl0", "kup:0_lvl0"), ("kup:1_lvl0",)]
    ld.close()
    # store adapter: block + coords + levels
    st = NewFormatCellStore(dst)
    assert st.has("kup:0_lvl0") and st.resolve(2) == "kup:2_lvl0"
    cd = st.cell("kup:1_lvl0")
    assert cd.n_pos == P and cd.levels == LEVELS
    assert np.array_equal(cd._block.view(np.uint16), blocks["kup:1_lvl0"].view(np.uint16))
    assert np.allclose(cd.lat, np.arange(P) + 1 * 10.0)      # position_lat from file
    st.close()


def test_cell_to_crops(tmp_path):
    pytest.importorskip("torch")
    from pyramid_newformat.loader import cell_to_crops
    _, blocks = _write_source(tmp_path / "s", n=1)
    blk = blocks["kup:0_lvl0"]
    grids, keys, kp = cell_to_crops(blk, LEVELS)
    assert tuple(grids.shape) == (P * len(LEVELS), H * W, D)
    assert keys[0] == (0, 1000) and len(keys) == P * len(LEVELS)
    assert kp.shape == (H * W, 2)


def test_errors(tmp_path):
    from pyramid_newformat.convert import read_source_cell
    # missing level
    src, _ = _write_source(tmp_path / "s", n=1, break_cell="kup:0_lvl0")
    cellp = Path(src) / "cells" / "kup_0_lvl0.h5"
    with pytest.raises(ValueError):
        read_source_cell(str(cellp), LEVELS, expect_pos=P)
    # non-contiguous positions
    src2, _ = _write_source(tmp_path / "s2", n=1)
    cp2 = Path(src2) / "cells" / "kup_0_lvl0.h5"
    with h5py.File(cp2, "a") as f:
        del f["p1"]                                          # leaves p0,p2 -> not contiguous
    with pytest.raises(ValueError):
        read_source_cell(str(cp2), LEVELS)
