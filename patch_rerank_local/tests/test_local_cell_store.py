"""Synthetic tests for the isolated local reranker IO + curve (no cv2 / no cuda needed)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")


def _write_cell(path: Path, n_pos: int, levels: list[int], h: int, w: int, D: int, seed: int):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        f.create_dataset("lat", data=np.linspace(49.0, 49.1, n_pos))
        f.create_dataset("lon", data=np.linspace(37.0, 37.1, n_pos))
        f.attrs["levels_m"] = np.asarray(levels, np.int64)
        f.attrs["n_positions"] = n_pos
        for i in range(n_pos):
            for L in levels:
                f.create_dataset(f"p{i}/l{int(L)}", data=rng.standard_normal((h, w, D)).astype(np.float32))


def _make_store_dir(root: Path, with_index: bool):
    cells = root / "cells"
    cells.mkdir(parents=True)
    ids = ["kup:0_lvl0", "kup:1_lvl0"]
    files = {}
    for k, cid in enumerate(ids):
        fn = cid.replace(":", "_").replace("/", "_") + ".h5"
        _write_cell(cells / fn, n_pos=3, levels=[1000, 500], h=4, w=4, D=8, seed=k)
        files[cid] = fn
    if with_index:
        idx = {"config": {"tile_size_m": 1000, "patch": 14},
               "cells": {cid: {"key": f"emb/kup/cells/{fn}", "lat": 49.0, "lon": 37.0}
                         for cid, fn in files.items()}}
        (root / "_index.json").write_text(json.dumps(idx), encoding="utf-8")
    return ids


def test_store_with_index(tmp_path):
    from patch_rerank_local.local_cell_store import LocalCellStore
    ids = _make_store_dir(tmp_path, with_index=True)
    s = LocalCellStore(str(tmp_path))
    assert s.config.get("tile_size_m") == 1000
    assert all(s.has(c) for c in ids)
    assert not s.has("kup:999_lvl0")
    cd = s.cell(ids[0])
    assert cd.n_pos == 3 and cd.levels == [1000, 500]
    g = cd.grid(0, 1000)
    assert tuple(g.shape) == (4, 4, 8)
    assert g.dtype.is_floating_point
    s.close()


def test_store_discovery_no_index(tmp_path):
    from patch_rerank_local.local_cell_store import LocalCellStore
    _make_store_dir(tmp_path, with_index=False)
    s = LocalCellStore(str(tmp_path))
    # cell_id = filename stem in discovery mode
    assert s.has("kup_0_lvl0") and s.has("kup_1_lvl0")
    cd = s.cell("kup_0_lvl0")
    assert cd.n_pos == 3
    s.close()


def test_open_cap_evicts(tmp_path):
    from patch_rerank_local.local_cell_store import LocalCellStore
    ids = _make_store_dir(tmp_path, with_index=True)
    s = LocalCellStore(str(tmp_path), open_cap=1)
    s.cell(ids[0]); s.cell(ids[1])
    assert len(s._open) == 1          # LRU bound to 1 open handle
    s.close()


def test_missing_file_raises(tmp_path):
    from patch_rerank_local.local_cell_store import LocalCellStore
    _make_store_dir(tmp_path, with_index=True)
    (tmp_path / "cells").rename(tmp_path / "cells_moved")   # break file resolution, keep index
    s = LocalCellStore(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        s.cell("kup:0_lvl0")


def test_curve_math(tmp_path):
    from patch_rerank_local.curve import compute
    # one query at GT (0,0); cellA ~11 m east (top score), cellB ~157 km away
    search = {"per_query": {"q": {"gt": [0.0, 0.0]}}}
    dump = {"q": {"A": {"mean": 0.9, "best": 0.9, "lat": 0.0, "lon": 0.0001, "level_m": 500},
                  "B": {"mean": 0.4, "best": 0.4, "lat": 1.0, "lon": 1.0, "level_m": 500}}}
    (tmp_path / "s.json").write_text(json.dumps(search))
    (tmp_path / "d.json").write_text(json.dumps(dump))
    res = compute(str(tmp_path / "s.json"), str(tmp_path / "d.json"))
    assert res["n_frames"] == 1
    c = res["by_topK"]
    assert c[1]["distR@250m"] == 1.0          # top-1 (A) within 250 m
    assert c[1]["distR@1000m"] == 1.0
    assert c[1]["gtcell_recall"] == 1.0       # A is both top-ranked and GT-closest
