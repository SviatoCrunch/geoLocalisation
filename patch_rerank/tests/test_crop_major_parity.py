"""Parity: --execution-order crop-major must produce byte-identical rankings to query-major.

Builds a tiny synthetic hybrid store + dense index + query H5 + shortlist (two queries that SHARE
cells, so crop-major actually reuses reads), runs map_rerank both ways on CPU, and asserts the
per-query top-k cells, scores, coordinates and distances match. cv2 MAGSAC is deterministic per
(qm,rm) pair regardless of call order, so equality is exact up to float noise.
"""
import json
import math

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("torch")
pytest.importorskip("h5py")

_R = 6378137.0


def _merc(lat, lon):
    return _R * math.radians(lon), _R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def _t2c(m, lat):
    return m / math.cos(math.radians(lat))


def _snap(v, s):
    return round(v / s) * s


def _build(tmp_path):
    import h5py
    rng = np.random.default_rng(0)
    D, N = 32, 100                                    # small feature dim; 10x10 token grid
    cells = [("kup:A", 48.90, 37.60), ("kup:B", 48.9025, 37.6025)]   # overlap -> shared positions
    levels = [1000, 800, 600, 400]
    step, rad = 250.0, 500.0

    allpos, cellpos = {}, {}
    for (cid, clat, clon) in cells:
        cx, cy = _merc(clat, clon)
        sc, rc = _t2c(step, clat), _t2c(rad, clat)
        offs = np.arange(-rc, rc + 1e-6, sc)
        ps = []
        for oy in offs:
            for ox in offs:
                p = (_snap(cx + ox, sc), _snap(cy + oy, sc))
                ps.append(p); allpos.setdefault(p, clat)
        cellpos[cid] = ps
    poslist = sorted(allpos)
    idx = {p: i for i, p in enumerate(poslist)}

    store = tmp_path / "store.h5"
    grids = {}
    with h5py.File(store, "w") as f:
        f.create_dataset("px", data=np.array([p[0] for p in poslist]))
        f.create_dataset("py", data=np.array([p[1] for p in poslist]))
        f.create_dataset("lat", data=np.zeros(len(poslist)))
        f.create_dataset("lon", data=np.zeros(len(poslist)))
        f.attrs["step_m"] = step; f.attrs["levels_m"] = [float(x) for x in levels]
        f.attrs["output_px"] = 140; f.attrs["tile_size_m"] = 1000.0; f.attrs["backbone"] = "test"
        for i in range(len(poslist)):
            for L in levels:
                g = rng.standard_normal((10, 10, D)).astype(np.float16)
                grids[(i, L)] = g
                f.create_dataset(f"p{i}/l{int(L)}", data=g, chunks=True)

    dense = tmp_path / "dense.h5"
    with h5py.File(dense, "w") as f:
        f.create_dataset("tile_id", data=np.array([c[0].encode() for c in cells]))
        f.create_dataset("lat", data=np.array([c[1] for c in cells]))
        f.create_dataset("lon", data=np.array([c[2] for c in cells]))
        f.create_dataset("city", data=np.array([b"kup", b"kup"]))

    # query H5: give each query strong matches to ONE real grid so MAGSAC actually runs
    qh5 = tmp_path / "query.h5"
    with h5py.File(qh5, "w") as f:
        for qname, seedgrid in [("q1", grids[(idx[cellpos["kup:A"][0]], 1000)]),
                                ("q2", grids[(idx[cellpos["kup:B"][0]], 800)])]:
            feat = seedgrid.reshape(100, D).astype(np.float32)          # copy tokens -> guaranteed MNN
            g = f.create_group(qname)
            g.create_dataset("ift_dino", data=feat.T)                  # (D, N)
            g.attrs["patch_grid_h"] = 10; g.attrs["patch_grid_w"] = 10
            g.attrs["lat"] = 48.9013; g.attrs["lon"] = 37.6013
            g.attrs["filename"] = f"{qname}_48.9013_37.6013.jpg"

    shortlist = tmp_path / "sl.json"
    json.dump({"shortlist": {"kup:q1_48.9013_37.6013": {"cells": ["kup:A", "kup:B"]},
                             "kup:q2_48.9013_37.6013b": {"cells": ["kup:B", "kup:A"]}}},
              open(shortlist, "w"))
    return str(shortlist), str(dense), str(qh5), str(store)


def test_crop_major_equals_query_major(tmp_path):
    from patch_rerank.map_rerank import main
    sl, di, qh5, st = _build(tmp_path)
    common = ["--shortlist", sl, "--dense-index", di, "--queries", f"kup={qh5}", "--store", st,
              "--k", "70", "--topk", "5", "--only-city", "kup", "--model", "homography",
              "--estimator", "magsac", "--device", "cpu", "--levels-m", "1000", "800", "600", "400"]
    outq, outc = tmp_path / "q.json", tmp_path / "c.json"
    assert main(common + ["--execution-order", "query-major", "--out", str(outq)]) == 0
    assert main(common + ["--execution-order", "crop-major", "--out", str(outc)]) == 0
    a = json.loads(outq.read_text()); b = json.loads(outc.read_text())

    assert set(a["per_query"]) == set(b["per_query"]) and a["per_query"]
    for q in a["per_query"]:
        pa, pb = a["per_query"][q], b["per_query"][q]
        assert [t["cell_id"] for t in pa["topk"]] == [t["cell_id"] for t in pb["topk"]]   # ranking
        for x, y in zip(pa["topk"], pb["topk"]):
            assert x["cell_id"] == y["cell_id"]
            assert abs(x["cell_score"] - y["cell_score"]) < 1e-6
            assert abs(x["lat"] - y["lat"]) < 1e-9 and abs(x["lon"] - y["lon"]) < 1e-9
        assert abs(pa["fine_dist_m"] - pb["fine_dist_m"]) < 1e-6
        assert abs(pa["fine_dist_topk_m"] - pb["fine_dist_topk_m"]) < 1e-6
    assert b.get("rerank_diagnostics", {}).get("execution_order") == "crop-major"
