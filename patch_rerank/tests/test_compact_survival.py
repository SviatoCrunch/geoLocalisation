"""compact_survival: mapping index<->baseline + selection + survival metrics run and are sane."""
import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("h5py")


def _make(tmp_path):
    import h5py
    from patch_rerank.build_compact_index import main as build
    rng = np.random.default_rng(0)
    D, npos, levels = 16, 5, (1000, 500)
    px = np.arange(npos, dtype=float) * 10.0
    py = np.arange(npos, dtype=float) * 100.0
    store = tmp_path / "store.h5"
    with h5py.File(store, "w") as f:
        f.create_dataset("px", data=px); f.create_dataset("py", data=py)
        f.create_dataset("lat", data=np.zeros(npos)); f.create_dataset("lon", data=np.zeros(npos))
        f.attrs["step_m"] = 250.0; f.attrs["levels_m"] = [float(x) for x in levels]
        f.attrs["output_px"] = 140; f.attrs["tile_size_m"] = 1000.0
        for i in range(npos):
            for L in levels:
                hw = 10 if L >= 1000 else 7
                f.create_dataset(f"p{i}/l{int(L)}", data=rng.standard_normal((hw, hw, D)).astype(np.float16),
                                 chunks=True)
    index = tmp_path / "compact.h5"
    assert build(["--store", str(store), "--out", str(index), "--pools", "mean", "grid4", "grid8",
                  "--device", "cpu"]) == 0

    # baseline: every (px,py,level) with a random score; one clear winner
    crops = []
    for i in range(npos):
        for L in levels:
            crops.append([float(px[i]), float(py[i]), int(L), float(rng.random())])
    crops[3][3] = 10.0                                     # unambiguous winner
    baseline = tmp_path / "baseline.json"
    json.dump({"kup:q1": crops}, open(baseline, "w"))

    qh5 = tmp_path / "query.h5"
    with h5py.File(qh5, "w") as f:
        g = f.create_group("q1")
        g.create_dataset("ift_dino", data=rng.standard_normal((D, 100)).astype(np.float32))
        g.attrs["patch_grid_h"] = 10; g.attrs["patch_grid_w"] = 10
        g.attrs["lat"] = 0.0; g.attrs["lon"] = 0.0; g.attrs["filename"] = "q1.jpg"
    return str(index), str(baseline), str(qh5)


def test_survival_runs_and_full_budget_keeps_winner(tmp_path):
    from patch_rerank.compact_survival import main
    index, baseline, qh5 = _make(tmp_path)
    out = tmp_path / "surv.json"
    assert main(["--compact-index", index, "--baseline", baseline, "--queries", f"kup={qh5}",
                 "--representations", "mean", "grid4", "grid8", "--grid-score", "chamfer",
                 "--budgets", "4", "20", "--device", "cpu", "--out", str(out)]) == 0
    tbl = json.loads(out.read_text())["table"]
    by = {(r["rep"], r["budget"]): r for r in tbl}
    assert len(tbl) == 6                                   # 3 reps x 2 budgets
    for rep in ("mean", "grid4", "grid8"):
        # budget 20 >= 10 crops -> everything selected -> winner + all top5 survive
        assert by[(rep, 20)]["winner_survival"] == 1.0
        assert by[(rep, 20)]["top5_survival"] == 1.0
        assert by[(rep, 20)]["winpos_survival"] == 1.0
        assert 0.0 <= by[(rep, 4)]["winner_survival"] <= 1.0
        assert by[(rep, 4)]["mean_distinct_pos"] <= by[(rep, 20)]["mean_distinct_pos"] + 1e-9


def test_survival_levels_per_position_cap(tmp_path):
    from patch_rerank.compact_survival import main
    index, baseline, qh5 = _make(tmp_path)
    out = tmp_path / "surv2.json"
    # cap 1 level/position with budget 20: at most npos(5) crops (one per position) selected
    assert main(["--compact-index", index, "--baseline", baseline, "--queries", f"kup={qh5}",
                 "--representations", "mean", "--budgets", "20", "--top-levels-per-position", "1",
                 "--device", "cpu", "--out", str(out)]) == 0
    r = json.loads(out.read_text())["table"][0]
    assert r["mean_distinct_pos"] <= 5 and r["mean_distinct_lev"] >= 1
