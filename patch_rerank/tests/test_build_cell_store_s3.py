"""Per-cell DINOv3 S3 store builder — CPU tests with the model, map reads and S3 all mocked.

Covers: the DINOv3 prefix-drop/reshape/normalise in extract_grids_v3 (fake model output), and the
build pipeline end-to-end (cell enumeration, per-cell H5 layout, shortlist restriction, incremental
"upload", manifest, and resume-skip) without torch models, GeoTIFFs or real S3.
"""
import json

import numpy as np
import pytest
import torch

from patch_rerank import build_cell_store_s3 as bcs
from patch_rerank.map_dino3 import extract_grids_v3


# --- extractor unit: prefix drop + reshape + L2 norm --------------------------------------------

class _FakeOut:
    def __init__(self, t):
        self.last_hidden_state = t


class _FakeModel:
    def __init__(self, n_prefix, hw, D):
        self.n_prefix, self.hw, self.D = n_prefix, hw, D

    def __call__(self, pixel_values):
        B = pixel_values.shape[0]
        return _FakeOut(torch.randn(B, self.n_prefix + self.hw, self.D))


def test_extract_grids_v3_drops_prefix_and_normalises():
    output_px, patch, D, n_prefix = 32, 16, 8, 5           # 32/16 = 2 → 2x2 = 4 patch tokens
    ext = {"model": _FakeModel(n_prefix, 4, D), "patch": patch, "D": D, "n_prefix": n_prefix,
           "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "model_id": "fake"}
    imgs = [np.random.randint(0, 255, (output_px, output_px, 3), np.uint8) for _ in range(3)]
    grids = extract_grids_v3(imgs, ext, output_px, device="cpu", amp=False)
    assert len(grids) == 3
    for g in grids:
        assert tuple(g.shape) == (2, 2, D)                 # (h,w,D), prefix removed
        norms = g.reshape(-1, D).norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)   # per-token L2-normed


def test_extract_grids_v3_rejects_bad_output_px():
    ext = {"model": _FakeModel(5, 4, 8), "patch": 16, "D": 8, "n_prefix": 5,
           "mean": [0, 0, 0], "std": [1, 1, 1], "model_id": "fake"}
    with pytest.raises(ValueError):
        extract_grids_v3([np.zeros((30, 30, 3), np.uint8)], ext, 30, device="cpu", amp=False)


# --- build pipeline (mocked model / map / S3) ----------------------------------------------------

def test_read_cells_from_mapextract_h5(tmp_path):
    """kup_prodaction-style map_extract H5 (per-tile groups with lat/lon/tile_index attrs)."""
    import h5py
    p = tmp_path / "prod.h5"
    with h5py.File(p, "w") as f:
        for ti, (la, lo) in [(0, (49.61, 37.56)), (100, (49.70, 37.66))]:
            g = f.create_group(f"{ti}_lvl0")
            g.create_dataset("ift_dino", data=np.zeros((8, 4), np.float32))
            g.attrs["lat"] = la; g.attrs["lon"] = lo; g.attrs["tile_index"] = float(ti)
    cells = bcs._read_cells(str(p), "kup")
    assert sorted(c[0] for c in cells) == ["kup:0_lvl0", "kup:100_lvl0"]
    d = {c[0]: (c[1], c[2]) for c in cells}
    assert d["kup:100_lvl0"] == (49.70, 37.66)


class _FakeS3:
    def __init__(self):
        self.objects, self.uploads, self.puts = {}, [], {}

    def head_object(self, Bucket, Key):
        if Key in self.objects:
            return {"ContentLength": self.objects[Key]}
        raise KeyError("404")

    def upload_file(self, local, Bucket, Key):
        import os
        self.objects[Key] = os.path.getsize(local); self.uploads.append(Key)

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.puts[Key] = Body


def _cell_index(tmp_path):
    import h5py
    p = tmp_path / "cells.h5"
    ids = ["kup:1_lvl0", "kup:2_lvl0", "kramatorsc:9_lvl0"]
    city = ["kup", "kup", "kramatorsc"]
    lat = [49.70, 49.71, 48.57]; lon = [37.66, 37.67, 37.63]
    with h5py.File(p, "w") as f:
        f.create_dataset("tile_id", data=np.array([s.encode() for s in ids]))
        f.create_dataset("city", data=np.array([s.encode() for s in city]))
        f.create_dataset("lat", data=np.array(lat)); f.create_dataset("lon", data=np.array(lon))
    return str(p)


def _patch_all(monkeypatch, s3):
    fake_ext = {"model": None, "patch": 16, "D": 8, "n_prefix": 5, "mean": [0, 0, 0], "std": [1, 1, 1],
                "model_id": "facebook/dinov3-vitl16-pretrain-sat493m",
                "backbone": "facebook/dinov3-vitl16-pretrain-sat493m"}
    monkeypatch.setattr(bcs, "build_dinov3_extractor", lambda model, device: fake_ext)
    monkeypatch.setattr(bcs, "extract_grids_v3",
                        lambda imgs, ext, opx, device, amp=True: [torch.randn(opx // 16, opx // 16, 8)
                                                                  for _ in imgs])
    monkeypatch.setattr(bcs, "open_src", lambda p: type("_Src", (), {"close": lambda self: None})())
    monkeypatch.setattr(bcs, "read_pyramid_from_one",
                        lambda src, px, py, levels, lat, opx: {int(L): np.zeros((opx, opx, 3), np.uint8)
                                                               for L in levels})
    monkeypatch.setattr("s3_gt_sync.core.make_s3_client", lambda: s3)


def _argv(idx, tmp_path, extra=None):
    a = ["--cell-index", idx, "--city", "kup", "--map", "dummy.tif",
         "--step-m", "500", "--levels-m", "1000", "500", "--output-px", "32", "--batch", "4",
         "--device", "cpu", "--s3-uri", "s3://bkt/emb/kup/rerank_cells",
         "--stage-dir", str(tmp_path / "stage")]
    return a + (extra or [])


def test_build_all_kup_cells(tmp_path, monkeypatch):
    s3 = _FakeS3(); _patch_all(monkeypatch, s3)
    idx = _cell_index(tmp_path)
    assert bcs.main(_argv(idx, tmp_path)) == 0

    stage = tmp_path / "stage"
    files = sorted(p.name for p in stage.glob("*.h5"))
    assert files == ["kup_1_lvl0.h5", "kup_2_lvl0.h5"]            # both kup cells, no other city

    import h5py
    with h5py.File(stage / "kup_1_lvl0.h5", "r") as f:
        assert f.attrs["cell_id"] == "kup:1_lvl0"
        assert "sat493m" in str(f.attrs["backbone"])
        assert set(f.keys()) >= {"px", "py", "lat", "lon"}
        p0 = f["p0"]
        assert "l1000" in p0 and "l500" in p0
        assert f["p0"]["l1000"].shape == (2, 2, 8)               # output_px/patch = 2

    keys = set(s3.uploads)
    assert "emb/kup/rerank_cells/kup/cells/kup_1_lvl0.h5" in keys
    assert "emb/kup/rerank_cells/kup/cells/kup_2_lvl0.h5" in keys
    assert "emb/kup/rerank_cells/kup/_index.json" in s3.puts     # manifest uploaded
    man = json.loads(s3.puts["emb/kup/rerank_cells/kup/_index.json"])
    assert man["n_cells"] == 2 and man["config"]["step_m"] == 500.0
    assert set(man["cells"]) == {"kup:1_lvl0", "kup:2_lvl0"}
    assert man["config"]["model"].endswith("sat493m")


def test_shortlist_restriction(tmp_path, monkeypatch):
    s3 = _FakeS3(); _patch_all(monkeypatch, s3)
    idx = _cell_index(tmp_path)
    sl = tmp_path / "sl.json"
    sl.write_text(json.dumps({"shortlist": {"kup:qA": {"cells": ["kup:1_lvl0"]}}}))
    assert bcs.main(_argv(idx, tmp_path, ["--cells-from-shortlist", str(sl)])) == 0
    files = sorted(p.name for p in (tmp_path / "stage").glob("*.h5"))
    assert files == ["kup_1_lvl0.h5"]                            # only the referenced cell


def test_resume_skips_existing(tmp_path, monkeypatch):
    s3 = _FakeS3(); _patch_all(monkeypatch, s3)
    idx = _cell_index(tmp_path)
    for cid in ("kup_1_lvl0", "kup_2_lvl0"):                     # pretend both already on S3
        s3.objects[f"emb/kup/rerank_cells/kup/cells/{cid}.h5"] = 123
    assert bcs.main(_argv(idx, tmp_path)) == 0
    assert not list((tmp_path / "stage").glob("*.h5"))           # nothing rebuilt
    assert s3.uploads == []                                      # nothing uploaded
    man = json.loads(s3.puts["emb/kup/rerank_cells/kup/_index.json"])
    assert man["n_cells"] == 2                                   # manifest still lists them
