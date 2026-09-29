"""CellStoreS3 reader + pyramid search — CPU tests with S3 and the GPU scorer mocked.

Covers: manifest load + fetch-by-index + LRU eviction/download count; the max-level→mean-pyramid
aggregation; and end-to-end search (top-k ranking, GT distances, timings, KMZ) with _score_cell stubbed.
"""
import io
import json
import shutil
import zipfile

import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("torch")

from patch_rerank import search_pyramid_s3 as sps
from patch_rerank.cell_store_s3 import CellStoreS3


def _make_cell(path, n_pos=2, levels=(1000, 500), D=4, with_grids=True):
    import h5py
    with h5py.File(path, "w") as f:
        f.create_dataset("lat", data=np.array([49.70 + 0.001 * i for i in range(n_pos)]))
        f.create_dataset("lon", data=np.array([37.66 + 0.001 * i for i in range(n_pos)]))
        f.attrs["levels_m"] = [float(x) for x in levels]
        f.attrs["n_positions"] = n_pos
        if with_grids:
            for i in range(n_pos):
                for L in levels:
                    f.create_dataset(f"p{i}/l{int(L)}", data=np.zeros((4, 4, D), np.float16))


class _FakeS3:
    def __init__(self, manifest: bytes, files: dict):
        self._m, self._f = manifest, files

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self._m)}

    def download_file(self, Bucket, Key, local):
        shutil.copy(self._f[Key], local)


def _store(tmp_path, monkeypatch, cache_cap=256):
    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    _make_cell(a); _make_cell(b)
    manifest = {"city": "kup", "config": {"model": "dinov3sat", "step_m": 250},
                "n_cells": 2,
                "cells": {"kup:0_lvl0": {"key": "cells/a.h5", "lat": 49.70, "lon": 37.66},
                          "kup:100_lvl0": {"key": "cells/b.h5", "lat": 49.71, "lon": 37.67}}}
    s3 = _FakeS3(json.dumps(manifest).encode(), {"cells/a.h5": str(a), "cells/b.h5": str(b)})
    return CellStoreS3("s3://bkt/kup/_index.json", cache_dir=str(tmp_path / "cache"),
                       cache_cap=cache_cap, s3_client=s3)


def test_cellstore_fetch_and_data(tmp_path, monkeypatch):
    st = _store(tmp_path, monkeypatch)
    assert st.has("kup:0_lvl0") and not st.has("kup:999_lvl0")
    cd = st.cell("kup:0_lvl0")
    assert cd.n_pos == 2 and cd.levels == [1000, 500]
    assert tuple(cd.grid(0, 1000).shape) == (4, 4, 4)
    assert st.n_downloads == 1
    st.cell("kup:0_lvl0")                      # cached → no new download
    assert st.n_downloads == 1
    st.close()


def test_cellstore_lru_eviction(tmp_path, monkeypatch):
    st = _store(tmp_path, monkeypatch, cache_cap=1)
    st.cell("kup:0_lvl0"); st.cell("kup:100_lvl0")
    assert st.n_downloads == 2 and len(st._lru) == 1     # first evicted
    st.close()


def test_aggregate_maxlevel_mean():
    class _CD:
        lat = np.array([49.70, 49.71]); lon = np.array([37.66, 37.67])
    cs, bi, blat, blon, bL = sps._aggregate([0.5, 0.9], [1000, 500], _CD, "mean")
    assert abs(cs - 0.7) < 1e-9                  # mean of pyramids
    assert bi == 1 and bL == 500                 # best pyramid = point, at its best level
    assert (blat, blon) == (49.71, 37.67)
    assert abs(sps._aggregate([0.5, 0.9], [1, 1], _CD, "min")[0] - 0.5) < 1e-9
    assert abs(sps._aggregate([0.5, 0.9], [1, 1], _CD, "max")[0] - 0.9) < 1e-9


def _query_h5(path):
    import h5py
    with h5py.File(path, "w") as f:
        g = f.create_group("q0")
        g.create_dataset("ift_dino", data=np.zeros((4, 16), np.float32))   # (D=4, N=16) → 4x4 grid
        g.attrs["patch_grid_h"] = 4; g.attrs["patch_grid_w"] = 4
        g.attrs["lat"] = 49.7105; g.attrs["lon"] = 37.6705
        g.attrs["filename"] = "0_49.7105_37.6705.jpg"


def test_search_end_to_end(tmp_path, monkeypatch):
    qh5 = tmp_path / "q.h5"; _query_h5(qh5)
    # fake S3 for CellStoreS3 built inside main()
    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    _make_cell(a); _make_cell(b)
    manifest = {"city": "kup", "config": {}, "n_cells": 2,
                "cells": {"kup:0_lvl0": {"key": "cells/a.h5", "lat": 49.70, "lon": 37.66},
                          "kup:100_lvl0": {"key": "cells/b.h5", "lat": 49.71, "lon": 37.67}}}
    s3 = _FakeS3(json.dumps(manifest).encode(), {"cells/a.h5": str(a), "cells/b.h5": str(b)})
    monkeypatch.setattr("s3_gt_sync.core.make_s3_client", lambda: s3)
    # stub the GPU scorer: cell a strong (best pyramid #1), cell b weak
    monkeypatch.setattr(sps, "_score_cell",
                        lambda cd, qfeat, qxy, n_q, args, dev:
                        (([0.5, 0.9], [1000, 500]) if cd.cell_id == "kup:0_lvl0"
                         else ([0.1, 0.2], [1000, 1000])))
    sl = tmp_path / "sl.json"
    sl.write_text(json.dumps({"shortlist": {"kup:0_49.7105_37.6705":
                                            {"cells": ["kup:100_lvl0", "kup:0_lvl0"]}}}))
    outj, outk = tmp_path / "s.json", tmp_path / "s.kmz"
    rc = sps.main(["--queries", f"kup={qh5}", "--shortlist", str(sl),
                   "--index-uri", "s3://bkt/kup/_index.json", "--cache-dir", str(tmp_path / "c"),
                   "--k-coarse", "100", "--topk", "5", "--cell-agg", "mean", "--device", "cpu",
                   "--out", str(outj), "--kmz", str(outk)])
    assert rc == 0
    j = json.loads(outj.read_text())
    rec = j["per_query"]["kup:0_49.7105_37.6705"]
    assert [t["cell_id"] for t in rec["topk"]] == ["kup:0_lvl0", "kup:100_lvl0"]   # mean 0.7 > 0.15
    top = rec["topk"][0]
    assert top["level_m"] == 500 and abs(top["lat"] - 49.701) < 1e-6               # best pyramid #1
    assert rec["fine_dist_m"] is not None and rec["timings"]["downloads"] == 2
    assert j["meta"]["cell_agg"] == "mean" and "speed" in j
    with zipfile.ZipFile(outk) as z:                                              # KMZ has squares
        kml = z.read("doc.kml").decode()
    assert "<Polygon>" in kml and "GT" in kml
