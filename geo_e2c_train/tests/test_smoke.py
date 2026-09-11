"""End-to-end smoke: synthetic galleries + queries + split → 2 training epochs + eval.

Validates the whole glue (data loaders + build_split_relevance + build_e2c_model + DSS loop +
evaluate) for residual/cell and supervlad/concentric on CPU. Skips if h5py/shapely absent.
"""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("shapely")
import h5py  # noqa: E402


D = 32
CITIES = {"ca": (48.50, 37.60), "cb": (49.60, 37.10)}


def _tile_grid(H=6, W=6, seed=0):
    r = np.random.RandomState(seed)
    return r.rand(1, D, H, W).astype(np.float16)              # (1,D,H,W)


def _write_gallery(path, city, lat0, lon0, n=9):
    with h5py.File(path, "w") as f:
        idx = 0
        for i in range(3):
            for j in range(3):
                g = f.create_group(f"{idx}_lvl0")
                g.create_dataset("ift_dino", data=_tile_grid(seed=idx))
                g.attrs["tile_index"] = idx
                g.attrs["lat"] = lat0 + i * 0.003
                g.attrs["lon"] = lon0 + j * 0.004
                g.attrs["window_size_m"] = 1000.0
                idx += 1


def _frames_for(city, lat0, lon0, keep=6):
    """Return list of (stem, lat, lon) placed exactly on tile centres (guaranteed positives)."""
    out, idx = [], 0
    for i in range(3):
        for j in range(3):
            if idx < keep:
                lat, lon = lat0 + i * 0.003, lon0 + j * 0.004
                out.append((f"{idx}_{lat:.5f}_{lon:.5f}", lat, lon))
            idx += 1
    return out


def _write_queries(path, city, frames):
    r = np.random.RandomState(1)
    with h5py.File(path, "w") as f:
        for n, (stem, lat, lon) in enumerate(frames):
            g = f.create_group(f"img{n}")
            g.create_dataset("ift_dino", data=r.rand(D, 30).astype(np.float32))   # (D,N)
            g.attrs["filename"] = f"{stem}.jpg"
            g.attrs["lat"] = lat
            g.attrs["lon"] = lon


def _setup(tmp):
    import torch
    from geo_split_no_overlap.build_index import build_index

    gt_dirs, galleries, queries, all_pids = {}, {}, {}, []
    for city, (lat0, lon0) in CITIES.items():
        gal = tmp / f"map_dinov2_{city}_s250m_d1024.h5"
        _write_gallery(gal, city, lat0, lon0)
        galleries[city] = str(gal)
        frames = _frames_for(city, lat0, lon0)
        gtd = tmp / f"gt_{city}" / "GT_flat"
        gtd.mkdir(parents=True)
        for stem, _, _ in frames:
            (gtd / f"{stem}.jpg").write_bytes(b"")               # stem-only; loader never opens it
        gt_dirs[city] = str(gtd)
        q = tmp / f"query_{city}_d1024.h5"
        _write_queries(q, city, frames)
        queries[city] = str(q)
        all_pids += [f"{city}:{stem}" for stem, _, _ in frames]

    index = tmp / "tiles_index.h5"
    build_index([(c, galleries[c]) for c in CITIES], index)

    cfg = tmp / "split.yaml"
    cfg.write_text(
        "gt:\n" + "".join(f"  - {c}={gt_dirs[c]}\n" for c in CITIES) +
        f"tiles_h5: {index}\n"
        "positive_selection:\n  strategy: pyramid_top_iou_250\n  params: {threshold: 0.25}\n"
        "train_ratio: 0.7\nval_ratio: 0.15\ntest_ratio: 0.15\ntile_size_m: 1000.0\n"
        f"seed: 0\nout_dir: {tmp / 'split'}\n", encoding="utf-8")

    # hand-written split.json (avoids needing an optimizer backend for the smoke)
    tr = all_pids[:8]; va = all_pids[8:10]; te = all_pids[10:12]
    sj = tmp / "split.json"
    sj.write_text(json.dumps({"train": tr, "val": va, "test": te, "excluded": []}), encoding="utf-8")

    aw = torch.randn(6, D); cent = torch.randn(6, D)
    k6 = tmp / "k6.pt"
    torch.save({"assign_weight": aw, "centroids": cent}, k6)
    return dict(cfg=cfg, sj=sj, galleries=galleries, queries=queries, k6=k6, out=tmp / "run")


def _argv(s, agg, mode, extra):
    a = ["--split-config", str(s["cfg"]), "--split-json", str(s["sj"]),
         "--galleries", *[f"{c}={p}" for c, p in s["galleries"].items()],
         "--queries", *[f"{c}={p}" for c, p in s["queries"].items()],
         "--assign", str(s["k6"]), "--agg", agg, "--pyramid-mode", mode,
         "--k", "6", "--d-token", str(D), "--d-group", "8", "--d-out", "16", "--d-hidden", "32",
         "--epochs", "2", "--b-log", "4", "--top-k", "4", "--eval-every", "1", "--patience", "5",
         "--eval-chunk", "8", "--out", str(s["out"]), "--device", "cpu", "--seed", "0"]
    return a + extra


@pytest.mark.parametrize("agg,mode,extra", [
    ("residual", "cell", ["--scales", "2", "1"]),
    ("supervlad", "concentric", ["--concentric-sizes", "1000", "500", "250"]),
])
def test_train_smoke(tmp_path, agg, mode, extra):
    from geo_e2c_train.train import main
    s = _setup(tmp_path)
    assert main(_argv(s, agg, mode, extra)) == 0
    best = json.loads((s["out"] / "best.json").read_text())
    assert "epoch" in best
    lines = (s["out"] / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2                                       # 2 epochs logged
    last = json.loads(lines[-1])
    assert "train_loss" in last and "val" in last and "R@10" in last["val"]
