"""Full cv2.MAGSAC++ diagnostics archive — CPU tests with cv2 faked.

Proves the task's acceptance criteria without needing OpenCV/GPU/S3:
  * adding the writer does NOT change scores / best position-level (diag-on == diag-off);
  * every candidate×position×level gets a record, including cv2-not-called skips;
  * each level score is reproducible from the archived mask + denominator;
  * position/cell aggregation is reproducible from the archived level scores;
  * H / mask / rm / qm round-trip through the arrays HDF5;
  * resume refuses to mix two configs (fingerprint guard).
"""
import json
import sys
import types

import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("torch")

import torch
import torch.nn.functional as F

from patch_rerank import rerank_diag
from patch_rerank import search_pyramid_s3 as sps
from patch_rerank.matcher import grid_keypoints


class _FakeCv2:
    """Deterministic stand-in: findHomography marks every other correspondence an inlier."""
    USAC_MAGSAC = 38
    USAC_DEFAULT = 32
    RANSAC = 8
    LMEDS = 4
    __version__ = "4.x-fake"
    error = Exception

    @staticmethod
    def findHomography(rm, qm, method, ransacReprojThreshold=2.0):
        n = len(rm)
        if n < 4:
            return None, None
        mask = np.zeros((n, 1), np.uint8)
        mask[::2] = 1
        return np.eye(3), mask


@pytest.fixture
def fake_cv2(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", _FakeCv2())
    return _FakeCv2


def _feats(seed, n=16, d=4):
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, d, generator=g), dim=1)


def _tiny_cell(cell_id="kram:5_lvl0"):
    """4 crops (2 positions × 2 levels), query == grids so mutual-NN matches all 16 patches."""
    q = _feats(0)
    kp = grid_keypoints(4, 4)
    grids = torch.stack([q, q, q, q])                       # (4 crops, 16, 4)
    keys = [(0, 1000), (0, 500), (1, 1000), (1, 500)]
    cd = types.SimpleNamespace(cell_id=cell_id, lat=np.array([1.0, 2.0]), lon=np.array([3.0, 4.0]),
                               levels=[1000, 500], n_pos=2)
    return grids, keys, kp, cd, q, kp.copy(), 16


def _args():
    return types.SimpleNamespace(verify_backend="cpu_magsac", reproj_thresh=2.0, grid_chunk=64,
                                 level_agg="sum", cell_agg="mean")


def _writer(tmp_path, fp="fp0"):
    return rerank_diag.DiagWriter(tmp_path, {"run_id": fp, "config": {}}, fp)


def _dctx(writer, cd, cell_id="kram:5_lvl0"):
    return {"writer": writer, "query_id": "kram:0_q", "cell_id": cell_id, "coarse_rank": 7,
            "method_int": 38,
            "base_meta": {"crop_key": "cells/x.h5", "cell_lat": 1.0, "cell_lon": 3.0, "tile_m": 1000.0,
                          "patch_px": 14, "query_grid_h": 4, "query_grid_w": 4, "query_n_keep": 16}}


def test_diag_on_off_scores_identical(tmp_path, fake_cv2):
    grids, keys, kp, cd, qf, qxy, n_q = _tiny_cell()
    a = _args()
    pyr0, blv0 = sps._score_loaded(grids, keys, kp, cd, qf, qxy, n_q, a, "cpu")
    w = _writer(tmp_path); w.begin_query("kram:0_q")
    pyr1, blv1 = sps._score_loaded(grids, keys, kp, cd, qf, qxy, n_q, a, "cpu", diag_ctx=_dctx(w, cd))
    w.finalize_cell(query_id="kram:0_q", cell_id="kram:5_lvl0", coarse_rank=7, pyr=pyr1, blv=blv1,
                    cd=cd, level_agg="sum")
    w.close()
    assert pyr0 == pyr1 and blv0 == blv1                    # writer did not perturb the result
    assert pyr0 == [1.0, 1.0] and blv0 == [1000, 1000]      # 0.5 per level, sum over 2 levels


def test_records_cover_all_and_reproduce_score(tmp_path, fake_cv2):
    grids, keys, kp, cd, qf, qxy, n_q = _tiny_cell()
    w = _writer(tmp_path); w.begin_query("kram:0_q")
    sps._score_loaded(grids, keys, kp, cd, qf, qxy, n_q, _args(), "cpu", diag_ctx=_dctx(w, cd))
    w.close()
    rows = [json.loads(l) for l in (tmp_path / "records.jsonl").read_text().splitlines()]
    assert len(rows) == 4                                   # 2 positions × 2 levels, all present
    assert {(r["position_id"], r["level_id"]) for r in rows} == {(0, 1000), (0, 500), (1, 1000), (1, 500)}
    for r in rows:
        assert r["cv2_called"] and r["status"] == "ok"
        assert r["n_inliers"] == 8 and r["n_query_patches"] == 16
        assert abs(r["level_score"] - r["n_inliers"] / r["n_query_patches"]) < 1e-12   # reproduced
        assert r["coarse_rank"] == 7 and r["crop_key"] == "cells/x.h5"


def test_arrays_roundtrip(tmp_path, fake_cv2):
    import h5py
    grids, keys, kp, cd, qf, qxy, n_q = _tiny_cell()
    w = _writer(tmp_path); w.begin_query("kram:0_q")
    sps._score_loaded(grids, keys, kp, cd, qf, qxy, n_q, _args(), "cpu", diag_ctx=_dctx(w, cd))
    w.close()
    with h5py.File(tmp_path / "arrays" / "kram_0_q.h5", "r") as f:
        g = f["kram_5_lvl0/p0/l1000"]
        qm, rm, mask, H = g["qm"][:], g["rm"][:], g["mask"][:], g["H"][:]
        assert qm.shape == (16, 2) and rm.shape == (16, 2)   # all mutual matches, in cv2 order
        assert mask.shape == (16,) and int(mask.sum()) == 8
        assert H.shape == (3, 3) and g.attrs["H_direction"].startswith("rm")
        assert g["q_idx"].shape == (16,) and g["residuals"].shape == (16,)
        assert int(g.attrs["n_inliers"]) == 8


def test_aggregation_reproduces(tmp_path, fake_cv2):
    grids, keys, kp, cd, qf, qxy, n_q = _tiny_cell()
    w = _writer(tmp_path); w.begin_query("kram:0_q")
    pyr, blv = sps._score_loaded(grids, keys, kp, cd, qf, qxy, n_q, _args(), "cpu", diag_ctx=_dctx(w, cd))
    w.finalize_cell(query_id="kram:0_q", cell_id="kram:5_lvl0", coarse_rank=7, pyr=pyr, blv=blv,
                    cd=cd, level_agg="sum")
    w.close()
    agg = json.loads((tmp_path / "aggregation.jsonl").read_text().splitlines()[0])
    # position_score = sum over its level scores; cell_score_mean = mean over positions
    for pos in agg["positions"]:
        assert abs(sum(pos["level_scores"].values()) - pos["position_score"]) < 1e-12
    assert abs(np.mean(agg["position_scores"]) - agg["cell_score_mean"]) < 1e-12
    assert agg["n_positions"] == 2 and agg["best_level_m"] in (500, 1000)


def test_skip_record_when_too_few_matches(tmp_path, fake_cv2):
    """Features that share NO mutual NN → n_mutual < 4 → cv2 NOT called, but a record still exists."""
    kp = grid_keypoints(4, 4)
    q = _feats(1)
    r = _feats(999)                                          # unrelated → at most the degenerate match
    grids = torch.stack([r])
    keys = [(0, 1000)]
    cd = types.SimpleNamespace(cell_id="kram:9_lvl0", lat=np.array([1.0]), lon=np.array([3.0]),
                               levels=[1000], n_pos=1)
    w = _writer(tmp_path); w.begin_query("kram:0_q")
    sps._score_loaded(grids, keys, kp, cd, q, kp.copy(), 16, _args(), "cpu",
                      diag_ctx=_dctx(w, cd, "kram:9_lvl0"))
    w.close()
    row = json.loads((tmp_path / "records.jsonl").read_text().splitlines()[0])
    if not row["cv2_called"]:                               # typical: unrelated features → <4 mutual
        assert row["status"] == "skipped_insufficient_matches" and row["level_score"] == 0.0
        assert row["h5_group"] is None and row["n_inliers"] == 0


def test_resume_fingerprint_guard(tmp_path, fake_cv2):
    w = rerank_diag.DiagWriter(tmp_path, {"config": {}}, "fpA")
    w.begin_query("q1")
    w.finalize_query(query_id="q1", gt=None, coarse_cells=["c0"], reranked=[], final_topk=[], metrics={})
    w.close()
    assert "q1" in set(json.loads((tmp_path / "completed.json").read_text())["queries"])
    # same fingerprint resumes and reports q1 done
    w2 = rerank_diag.DiagWriter(tmp_path, {"config": {}}, "fpA")
    assert w2.is_done("q1"); w2.close()
    # different fingerprint → refuse to mix configs
    with pytest.raises(RuntimeError, match="fingerprint"):
        rerank_diag.DiagWriter(tmp_path, {"config": {}}, "fpB")


def test_fingerprint_is_stable_and_order_independent():
    a = {"k": 1, "x": [1, 2], "s": {"b": 2, "a": 1}}
    b = {"s": {"a": 1, "b": 2}, "x": [1, 2], "k": 1}
    assert rerank_diag.config_fingerprint(a) == rerank_diag.config_fingerprint(b)


# ---- end-to-end main() with the real cpu_magsac diag path (S3 + cv2 faked) ----

class _FakeS3:
    def __init__(self, manifest, files):
        import io
        self._m, self._f, self._io = manifest, files, io

    def get_object(self, Bucket, Key):
        return {"Body": self._io.BytesIO(self._m)}

    def download_file(self, Bucket, Key, local):
        import shutil
        shutil.copy(self._f[Key], local)


def _distinct_cell(path, feats):
    import h5py
    with h5py.File(path, "w") as f:
        f.create_dataset("lat", data=np.array([1.0, 2.0]))
        f.create_dataset("lon", data=np.array([3.0, 4.0]))
        f.attrs["levels_m"] = [1000.0, 500.0]
        f.attrs["n_positions"] = 2
        for i in range(2):
            for L in (1000, 500):
                f.create_dataset(f"p{i}/l{int(L)}", data=feats.reshape(4, 4, 4).astype(np.float16))


def _query_h5(path, feats):
    import h5py
    with h5py.File(path, "w") as f:
        g = f.create_group("q0")
        g.create_dataset("ift_dino", data=feats.T.astype(np.float32))   # (D=4, N=16)
        g.attrs["patch_grid_h"] = 4; g.attrs["patch_grid_w"] = 4
        g.attrs["lat"] = 1.5; g.attrs["lon"] = 3.5
        g.attrs["filename"] = "0_1.5_3.5.jpg"


def test_main_diag_end_to_end(tmp_path, fake_cv2, monkeypatch):
    feats = F.normalize(_feats(3), dim=1).numpy().astype(np.float32)    # query == grids → matches
    qh5 = tmp_path / "q.h5"; _query_h5(qh5, feats)
    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    _distinct_cell(a, feats); _distinct_cell(b, feats)
    manifest = {"city": "kram", "config": {"tile_size_m": 1000.0, "patch": 14}, "n_cells": 2,
                "cells": {"kram:0_lvl0": {"key": "cells/a.h5", "lat": 1.0, "lon": 3.0},
                          "kram:1_lvl0": {"key": "cells/b.h5", "lat": 2.0, "lon": 4.0}}}
    s3 = _FakeS3(json.dumps(manifest).encode(), {"cells/a.h5": str(a), "cells/b.h5": str(b)})
    monkeypatch.setattr("boto3.client", lambda svc, *a, **k: s3)
    sl = tmp_path / "sl.json"
    sl.write_text(json.dumps({"shortlist": {"kram:0_1.5_3.5": {"cells": ["kram:0_lvl0", "kram:1_lvl0"]}}}))
    dd = tmp_path / "diag"

    def _run():
        return sps.main(["--queries", f"kram={qh5}", "--shortlist", str(sl), "--index-uri",
                         "s3://bkt/kram/_index.json", "--cache-dir", str(tmp_path / "cache"),
                         "--k-coarse", "100", "--topk", "5", "--cell-agg", "mean", "--level-agg", "sum",
                         "--device", "cpu", "--out", str(tmp_path / "out.json"), "--diag-dir", str(dd)])

    assert _run() == 0
    man = json.loads((dd / "manifest.json").read_text())
    assert man["cv2"]["method"] == "USAC_MAGSAC" and man["config"]["verify_backend"] == "cpu_magsac"
    assert man["fingerprint"] and man["schema_version"] == 1
    recs = [json.loads(l) for l in (dd / "records.jsonl").read_text().splitlines()]
    assert len(recs) == 2 * 2 * 2                            # 2 cells × 2 positions × 2 levels
    assert all(r["query_id"] == "kram:0_1.5_3.5" for r in recs)
    assert {r["cell_id"] for r in recs} == {"kram:0_lvl0", "kram:1_lvl0"}
    summ = json.loads((dd / "summary.jsonl").read_text().splitlines()[0])
    assert len(summ["coarse_top"]) == 2 and len(summ["reranked_top"]) == 2
    assert summ["gt"] == [1.5, 3.5] and len(summ["final_topk"]) == 2
    done = json.loads((dd / "completed.json").read_text())
    assert "kram:0_1.5_3.5" in done["queries"]

    n_recs = len(recs)
    assert _run() == 0                                       # resume: query already done → skipped
    recs2 = (dd / "records.jsonl").read_text().splitlines()
    assert len(recs2) == n_recs                              # no new records appended


def _local_store(root, feats):
    """Write a <root>/_index.json + cells/*.h5 store like build_cell_store_s3 (no S3)."""
    (root / "cells").mkdir(parents=True, exist_ok=True)
    cells = {}
    for cid, lat, lon in [("kram:0_lvl0", 1.0, 3.0), ("kram:1_lvl0", 2.0, 4.0)]:
        safe = cid.replace(":", "_")
        _distinct_cell(root / "cells" / f"{safe}.h5", feats)
        cells[cid] = {"key": f"cells/{safe}.h5", "lat": lat, "lon": lon}
    (root / "_index.json").write_text(json.dumps(
        {"city": "kram", "config": {"tile_size_m": 1000.0, "patch": 14}, "cells": cells}))


def test_main_diag_store_dir(tmp_path, fake_cv2):
    """--store-dir reads a local pyramid store: zero downloads, no boto3, full archive still written."""
    feats = F.normalize(_feats(3), dim=1).numpy().astype(np.float32)
    qh5 = tmp_path / "q.h5"; _query_h5(qh5, feats)
    root = tmp_path / "store"; _local_store(root, feats)
    sl = tmp_path / "sl.json"
    sl.write_text(json.dumps({"shortlist": {"kram:0_1.5_3.5": {"cells": ["kram:0_lvl0", "kram:1_lvl0"]}}}))
    dd = tmp_path / "diag"
    rc = sps.main(["--queries", f"kram={qh5}", "--shortlist", str(sl), "--store-dir", str(root),
                   "--k-coarse", "100", "--topk", "5", "--cell-agg", "mean", "--level-agg", "sum",
                   "--device", "cpu", "--out", str(tmp_path / "out.json"), "--diag-dir", str(dd)])
    assert rc == 0
    man = json.loads((dd / "manifest.json").read_text())
    assert man["config"]["store_source"] == f"local:{root}"
    recs = (dd / "records.jsonl").read_text().splitlines()
    assert len(recs) == 2 * 2 * 2                            # 2 cells × 2 positions × 2 levels
    assert "kram:0_1.5_3.5" in json.loads((dd / "completed.json").read_text())["queries"]


def test_store_dir_xor_index_uri(tmp_path, fake_cv2):
    sl = tmp_path / "sl.json"; sl.write_text(json.dumps({"shortlist": {}}))
    qh5 = tmp_path / "q.h5"; _query_h5(qh5, F.normalize(_feats(3), dim=1).numpy().astype(np.float32))
    base = ["--queries", f"kram={qh5}", "--shortlist", str(sl), "--device", "cpu",
            "--out", str(tmp_path / "o.json")]
    with pytest.raises(SystemExit):                          # neither given
        sps.main(base)
    with pytest.raises(SystemExit):                          # both given
        sps.main(base + ["--store-dir", str(tmp_path), "--index-uri", "s3://b/k/_index.json"])
