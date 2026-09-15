"""build_compact_index: offline pooled index from a store (no DINO), correct shapes + mean parity."""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("h5py")


def _store(tmp_path, n_pos=5, D=16, levels=(1000, 500)):
    import h5py
    rng = np.random.default_rng(0)
    grids = {}
    p = tmp_path / "store.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("px", data=np.arange(n_pos, dtype=float) * 10)
        f.create_dataset("py", data=np.arange(n_pos, dtype=float) * 100)
        f.create_dataset("lat", data=np.zeros(n_pos)); f.create_dataset("lon", data=np.zeros(n_pos))
        f.attrs["step_m"] = 250.0; f.attrs["levels_m"] = [float(x) for x in levels]
        f.attrs["output_px"] = 140; f.attrs["tile_size_m"] = 1000.0
        for i in range(n_pos):
            for L in levels:
                hw = 10 if L >= 1000 else 7
                g = rng.standard_normal((hw, hw, D)).astype(np.float16)
                grids[(i, int(L))] = g
                f.create_dataset(f"p{i}/l{int(L)}", data=g, chunks=True)
    return str(p), grids, D, levels, n_pos


def test_build_shapes_and_mean_parity(tmp_path):
    import h5py
    from patch_rerank.build_compact_index import main, SCHEMA_VERSION
    store, grids, D, levels, n_pos = _store(tmp_path)
    out = tmp_path / "compact.h5"
    assert main(["--store", store, "--out", str(out), "--pools", "mean", "grid4", "grid8",
                 "--device", "cpu"]) == 0
    with h5py.File(out, "r") as f:
        M = n_pos * len(levels)
        assert f["mean"].shape == (M, D)
        assert f["grid4"].shape == (M, 4, 4, D)
        assert f["grid8"].shape == (M, 8, 8, D)
        assert f.attrs["schema_version"] == SCHEMA_VERSION and f.attrs["source_fingerprint"]
        ci = f["crop_i"][:]; cl = f["crop_level"][:]
        # mean parity: index mean == grid.reshape(-1,D).mean(0)
        for r in range(M):
            g = grids[(int(ci[r]), int(cl[r]))].astype(np.float32)
            ref = g.reshape(-1, D).mean(0)
            got = f["mean"][r].astype(np.float32)
            assert np.allclose(got, ref, atol=3e-2), f"mean mismatch at {r}"


def test_pool_subset(tmp_path):
    import h5py
    from patch_rerank.build_compact_index import main
    store, _, D, levels, n_pos = _store(tmp_path)
    out = tmp_path / "c2.h5"
    assert main(["--store", store, "--out", str(out), "--pools", "mean", "--device", "cpu"]) == 0
    with h5py.File(out, "r") as f:
        assert "mean" in f and "grid4" not in f and "grid8" not in f
        assert list(f.attrs["pools"]) == ["mean"]
