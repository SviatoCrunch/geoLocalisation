"""DINOv3-sat query extractor — CPU test with the model mocked; verifies the output H5 is readable
by QueryGridStore with correct ids / lat-lon / token shapes."""
import numpy as np
import pytest
import torch

pytest.importorskip("h5py")
pytest.importorskip("cv2")

from patch_rerank import query_extract_dino3 as qx
from patch_rerank.query_io import QueryGridStore


def test_latlon_parse():
    assert qx._latlon_from_stem("1254_49.7016913268_37.6613484799") == (49.7016913268, 37.6613484799)
    la, lo = qx._latlon_from_stem("noname")
    assert la != la and lo != lo                                   # NaN


def test_extract_writes_querygridstore_format(tmp_path, monkeypatch):
    import cv2
    fr = tmp_path / "frames"; fr.mkdir()
    for nm in ["0_49.70_37.66.jpg", "1_49.71_37.67.jpg"]:
        cv2.imwrite(str(fr / nm), np.zeros((48, 64, 3), np.uint8))

    fake_ext = {"model": None, "patch": 16, "D": 8, "n_prefix": 5, "mean": [0, 0, 0], "std": [1, 1, 1],
                "model_id": "facebook/dinov3-vitl16-pretrain-sat493m"}
    monkeypatch.setattr(qx, "build_dinov3_extractor", lambda model, device: fake_ext)
    monkeypatch.setattr(qx, "extract_grids_v3",
                        lambda imgs, ext, opx, device, amp=True: [torch.randn(opx // 16, opx // 16, 8)
                                                                  for _ in imgs])
    out = tmp_path / "q.h5"
    rc = qx.main(["--frames", str(fr), "--city", "kup", "--output-px", "32", "--device", "cpu",
                  "--out", str(out)])
    assert rc == 0

    qs = QueryGridStore({"kup": str(out)})
    assert qs.has("kup:0_49.70_37.66") and qs.has("kup:1_49.71_37.67")
    feat, xy, lat, lon = qs.get("kup:0_49.70_37.66")
    assert feat.shape == (4, 8) and xy.shape == (4, 2)             # 32/16=2 → 2x2=4 tokens, D=8
    assert abs(lat - 49.70) < 1e-6 and abs(lon - 37.66) < 1e-6
    qs.close()
