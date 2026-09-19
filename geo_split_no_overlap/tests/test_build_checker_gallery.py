"""build_checker_gallery: dense gallery + checker selection → standalone checker gallery that is
BOTH a group-per-tile feature store (TileGridLoader) AND a baked geometry index (build_gallery_index).
"""
import numpy as np
import pytest


def _write_dense(path, city, n, lat0=48.56, lon0=37.62):
    """Raw group-per-tile gallery; each tile's ift_dino is filled with its index so a faithful
    copy is verifiable byte-for-byte."""
    import h5py
    with h5py.File(path, "w") as f:
        f.attrs["backbone"] = "dinov2_vitg14"          # a root attr that must survive the copy
        for i in range(n):
            g = f.create_group(f"{i}_lvl0")
            g.create_dataset("ift_dino", data=np.full((1, 4, 2, 2), i, np.float16))
            g.attrs["tile_index"] = i
            g.attrs["lat"] = lat0 + 0.001 * i
            g.attrs["lon"] = lon0 + 0.001 * i
            g.attrs["window_size_m"] = 1000.0


def _write_checker_index(path, entries):
    """entries: list of (city, groupkey). Writes the top-level city/tile_id schema the tool selects on."""
    import h5py
    str_dt = h5py.string_dtype("utf-8")
    tid = [f"{c}:{k}" for c, k in entries]
    city = [c for c, _ in entries]
    with h5py.File(path, "w") as f:
        f.create_dataset("city", data=np.asarray(city, object), dtype=str_dt)
        f.create_dataset("tile_id", data=np.asarray(tid, object), dtype=str_dt)


def test_standalone_checker_gallery_roundtrip(tmp_path):
    pytest.importorskip("h5py")
    import h5py
    from geo_split_no_overlap.build_checker_gallery import build
    from geo_split_no_overlap.positive_selection import build_gallery_index

    dense = tmp_path / "map_dinov2_kramatorsc_s250m_d1024.h5"
    _write_dense(dense, "kramatorsc", 5)
    checker = tmp_path / "checker1000_noverlap_kramatorsc.h5"
    _write_checker_index(checker, [("kramatorsc", "0_lvl0"), ("kramatorsc", "2_lvl0"),
                                   ("kramatorsc", "4_lvl0")])                    # keep 3 of 5
    out = tmp_path / "checker_gallery_kramatorsc.h5"

    infos = build({"kramatorsc": dense}, [checker], {"kramatorsc": out})
    assert infos[0]["n_tiles"] == 3

    with h5py.File(out, "r") as f:
        # only the kept tile groups were copied (no overlapping tiles left)
        groups = {k for k in f.keys() if hasattr(f[k], "keys") and "ift_dino" in f[k]}
        assert groups == {"0_lvl0", "2_lvl0", "4_lvl0"}
        # baked top-level geometry index in the SAME file
        assert list(f["tile_id"][:].astype(str)) == ["kramatorsc:0_lvl0", "kramatorsc:2_lvl0",
                                                      "kramatorsc:4_lvl0"]
        assert f.attrs["kind"] == "checker_gallery" and f.attrs["backbone"] == "dinov2_vitg14"
        # features copied faithfully (tile 2's grid is all 2s)
        assert np.allclose(np.asarray(f["2_lvl0"]["ift_dino"]), 2.0)

    # the split loader reads the checker gallery directly — no separate index needed
    g = build_gallery_index(str(out), "EPSG:3857", 1000.0)
    assert len(g) == 3 and set(g.cities()) == {"kramatorsc"}

    # TileGridLoader reads embeddings from the same file, keyed by tile_id
    from geo_e2c_train.data import TileGridLoader
    loader = TileGridLoader({"kramatorsc": str(out)})
    grid = loader.grid("kramatorsc:4_lvl0")                 # (H,W,D)
    assert grid.shape == (2, 2, 4) and float(grid.mean()) == 4.0
    loader.close()


def test_missing_tile_raises(tmp_path):
    pytest.importorskip("h5py")
    from geo_split_no_overlap.build_checker_gallery import build

    dense = tmp_path / "map_dinov2_kup_s250m_d1024.h5"
    _write_dense(dense, "kup", 3)
    checker = tmp_path / "checker.h5"
    _write_checker_index(checker, [("kup", "0_lvl0"), ("kup", "99_lvl0")])       # 99 absent
    with pytest.raises(SystemExit):
        build({"kup": dense}, [checker], {"kup": tmp_path / "o.h5"})


def test_combined_index_dispatched_per_city(tmp_path):
    pytest.importorskip("h5py")
    import h5py
    from geo_split_no_overlap.build_checker_gallery import build

    da = tmp_path / "kram.h5"; db = tmp_path / "kup.h5"
    _write_dense(da, "kramatorsc", 4, 48.56, 37.62)
    _write_dense(db, "kup", 4, 49.68, 37.10)
    combined = tmp_path / "tiles_index_checker1000.h5"
    _write_checker_index(combined, [("kramatorsc", "1_lvl0"), ("kup", "0_lvl0"),
                                    ("kup", "3_lvl0")])
    oa = tmp_path / "cg_kram.h5"; ob = tmp_path / "cg_kup.h5"
    infos = build({"kramatorsc": da, "kup": db}, [combined], {"kramatorsc": oa, "kup": ob})
    by_city = {i["city"]: i["n_tiles"] for i in infos}
    assert by_city == {"kramatorsc": 1, "kup": 2}
    with h5py.File(ob, "r") as f:
        assert set(f["tile_id"][:].astype(str)) == {"kup:0_lvl0", "kup:3_lvl0"}
