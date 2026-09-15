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


def test_parallel_reads_match_serial(tmp_path):
    """map_rerank --read-jobs reads with one h5py handle per thread; grids must equal serial reads."""
    import h5py
    from concurrent.futures import ThreadPoolExecutor
    from patch_rerank.store import RerankStore
    p = tmp_path / "store.h5"
    n = 12
    px = np.arange(n, dtype=float) * 10.0; py = np.arange(n, dtype=float) * 100.0
    with h5py.File(p, "w") as f:
        f.create_dataset("px", data=px); f.create_dataset("py", data=py)
        f.create_dataset("lat", data=np.zeros(n)); f.create_dataset("lon", data=np.zeros(n))
        f.attrs["step_m"] = 250.0; f.attrs["levels_m"] = [1000.0, 500.0]
        f.attrs["output_px"] = 518; f.attrs["tile_size_m"] = 1000.0
        for i in range(n):
            for L in (1000, 500):
                f.create_dataset(f"p{i}/l{L}", data=np.full((3, 3, 4), i + 0.5 * (L == 500), np.float16))

    s = RerankStore(str(p))
    keys = [(px[i], py[i], L) for i in range(n) for L in (1000, 500)]
    serial = {k: s.grid(*k) for k in keys}
    tls = __import__("threading").local()

    def _read(k):
        h = getattr(tls, "h", None)
        if h is None:
            h = tls.h = s.open_handle()
        return k, s.grid_from(h, *k)

    with ThreadPoolExecutor(max_workers=4) as ex:
        par = dict(ex.map(_read, keys))
    for k in keys:
        assert np.array_equal(par[k].numpy(), serial[k].numpy())
    s.close()
