"""CLI smoke: build tiny galleries/queries/split + a checkpoint, run ALL checks, assert outputs."""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("shapely")
import h5py  # noqa: E402

D = 32
CITIES = {"ca": (48.50, 37.60), "cb": (49.60, 37.10)}


def _write_gallery(path, lat0, lon0):
    r = np.random.RandomState(0)
    with h5py.File(path, "w") as f:
        idx = 0
        for i in range(3):
            for j in range(3):
                g = f.create_group(f"{idx}_lvl0")
                g.create_dataset("ift_dino", data=r.rand(1, D, 6, 6).astype(np.float16))
                g.attrs["tile_index"] = idx
                g.attrs["lat"] = lat0 + i * 0.003
                g.attrs["lon"] = lon0 + j * 0.004
                g.attrs["window_size_m"] = 1000.0
                idx += 1


def _frames(lat0, lon0, keep=6):
    out, idx = [], 0
    for i in range(3):
        for j in range(3):
            if idx < keep:
                out.append((f"{idx}_{lat0 + i*0.003:.5f}_{lon0 + j*0.004:.5f}",
                            lat0 + i * 0.003, lon0 + j * 0.004))
            idx += 1
    return out


def _write_queries(path, frames):
    r = np.random.RandomState(1)
    with h5py.File(path, "w") as f:
        for n, (stem, lat, lon) in enumerate(frames):
            g = f.create_group(f"img{n}")
            g.create_dataset("ift_dino", data=r.rand(D, 30).astype(np.float32))
            g.attrs["filename"] = f"{stem}.jpg"
            g.attrs["lat"] = lat
            g.attrs["lon"] = lon


def _setup(tmp):
    import torch
    from geo_split_no_overlap.build_index import build_index
    from siam_e2c_model.config import E2cModelConfig
    from siam_e2c_model.model import build_e2c_model

    gt, galleries, queries, pids = {}, {}, {}, []
    for city, (lat0, lon0) in CITIES.items():
        gal = tmp / f"map_dinov2_{city}_s250m_d1024.h5"
        _write_gallery(gal, lat0, lon0); galleries[city] = str(gal)
        fr = _frames(lat0, lon0)
        d = tmp / f"gt_{city}" / "GT_flat"; d.mkdir(parents=True)
        for stem, _, _ in fr:
            (d / f"{stem}.jpg").write_bytes(b"")
        gt[city] = str(d)
        q = tmp / f"query_{city}.h5"; _write_queries(q, fr); queries[city] = str(q)
        pids += [f"{city}:{stem}" for stem, _, _ in fr]

    index = tmp / "tiles_index.h5"
    build_index([(c, galleries[c]) for c in CITIES], index)
    cfg = tmp / "split.yaml"
    cfg.write_text("gt:\n" + "".join(f"  - {c}={gt[c]}\n" for c in CITIES) +
                   f"tiles_h5: {index}\n"
                   "positive_selection:\n  strategy: pyramid_top_iou_250\n  params: {threshold: 0.25}\n"
                   "train_ratio: 0.7\nval_ratio: 0.15\ntest_ratio: 0.15\ntile_size_m: 1000.0\n"
                   f"seed: 0\nout_dir: {tmp / 'split'}\n", encoding="utf-8")
    sj = tmp / "split.json"
    sj.write_text(json.dumps({"train": pids[:8], "val": pids[8:10], "test": pids[10:12],
                              "excluded": []}), encoding="utf-8")

    k6 = tmp / "k6.pt"
    torch.save({"assign_weight": torch.randn(6, D), "centroids": torch.randn(6, D)}, k6)
    mcfg = E2cModelConfig(agg="residual", k=6, d_token=D, scales_cells=(2, 1), d_group=8,
                          d_out=16, d_hidden=32, assign_path=str(k6))
    model = build_e2c_model(mcfg)
    ckpt = tmp / "best.pt"
    torch.save({"model": model.state_dict(), "epoch": 5, "resolved_config": model.resolved_config()}, ckpt)
    return dict(cfg=cfg, sj=sj, galleries=galleries, queries=queries, k6=k6, ckpt=ckpt)


def test_cli_all_checks_produces_all_outputs(tmp_path):
    from retrieval_overfit_diagnostics.cli import main
    s = _setup(tmp_path)
    out = tmp_path / "outputs" / "run0"
    argv = ["--checkpoint", str(s["ckpt"]), "--split-config", str(s["cfg"]),
            "--split-json", str(s["sj"]),
            "--galleries", *[f"{c}={p}" for c, p in s["galleries"].items()],
            "--queries", *[f"{c}={p}" for c, p in s["queries"].items()],
            "--assign", str(s["k6"]), "--checks", "all", "--eval-chunk", "4",
            "--device", "cpu", "--output-dir", str(out)]
    assert main(argv) == 0
    expected = ["resolved_config.json", "train_objective_metrics.json",
                "train_full_gallery_metrics.json", "val_full_gallery_metrics.json",
                "decisive_summary.json", "cache_audit.json", "checkpoint_audit.json",
                "labels_audit.json", "scale_gate_summary.json", "ablation_metrics.json",
                "negative_mining_audit.json", "split_shift_audit.json", "per_query.csv",
                "hypotheses.json", "diagnostic_report.md", "run.log"]
    for fn in expected:
        assert (out / fn).exists(), f"missing {fn}"
    # decisive summary has the three blocks + interpretation
    dec = json.loads((out / "decisive_summary.json").read_text())
    for k in ("train_objective", "train_full_gallery", "val_full_gallery", "interpretation"):
        assert k in dec
    # checkpoint audit: strict load, no missing/unexpected keys, eval determinism ~0
    ck = json.loads((out / "checkpoint_audit.json").read_text())
    assert ck["missing_keys"] == [] and ck["unexpected_keys"] == []
    assert ck["eval_determinism_max_abs_diff"] < 1e-4
    # freshness: V changes when a map param is perturbed
    ca = json.loads((out / "cache_audit.json").read_text())
    assert ca["eval_uses_current_checkpoint"] is True
