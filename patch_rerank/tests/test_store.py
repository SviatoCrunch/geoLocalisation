"""RerankStore read round-trip against a hand-written H5 in the precompute layout."""
import numpy as np
import pytest

pytest.importorskip("h5py")


def test_store_roundtrip(tmp_path):
    import h5py
    from patch_rerank.store import RerankStore
    p = tmp_path / "store.h5"
    px = np.array([100.0, 200.0]); py = np.array([300.0, 400.0])
    with h5py.File(p, "w") as f:
        f.create_dataset("px", data=px); f.create_dataset("py", data=py)
        f.create_dataset("lat", data=np.array([48.9, 48.91]))
        f.create_dataset("lon", data=np.array([37.6, 37.61]))
        f.attrs["step_m"] = 250.0
        f.attrs["levels_m"] = [1000.0, 500.0]
        f.attrs["output_px"] = 518
        f.attrs["tile_size_m"] = 1000.0
        f.attrs["backbone"] = "dinov2_vitg14"
        for i in range(2):
            for L in (1000, 500):
                f.create_dataset(f"p{i}/l{L}", data=np.ones((3, 3, 4), np.float16) * (i + 1))

    s = RerankStore(str(p))
    assert s.step_m == 250.0 and s.levels_m == [1000.0, 500.0] and s.output_px == 518
    assert s.has(100.0, 300.0) and not s.has(999.0, 999.0)
    g = s.grid(200.0, 400.0, 1000)                    # position 1 → filled with 2.0
    assert g.shape == (3, 3, 4) and float(g[0, 0, 0]) == 2.0
    s.close()
