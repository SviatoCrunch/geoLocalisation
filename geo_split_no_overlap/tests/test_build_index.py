"""build_index: raw group-per-tile galleries → tiles-index H5 readable by build_gallery_index."""
import numpy as np
import pytest


def _write_raw(path, n, lat0, lon0):
    import h5py
    with h5py.File(path, "w") as f:
        for i in range(n):
            g = f.create_group(f"{i}_lvl0")
            g.create_dataset("ift_dino", data=np.zeros((1, 4, 2, 2), np.float16))
            g.attrs["tile_index"] = i
            g.attrs["lat"] = lat0 + 0.001 * i
            g.attrs["lon"] = lon0 + 0.001 * i
            g.attrs["window_size_m"] = 1000.0


def test_build_index_and_read_back(tmp_path):
    pytest.importorskip("h5py")
    from geo_split_no_overlap.build_index import build_index
    from geo_split_no_overlap.positive_selection import build_gallery_index

    ga = tmp_path / "map_dinov2_kramatorsc_s250m_d1024.h5"
    gb = tmp_path / "map_dinov2_kup_s250m_d1024.h5"
    _write_raw(ga, 5, 48.56, 37.62)
    _write_raw(gb, 3, 49.68, 37.10)
    out = tmp_path / "tiles_index.h5"
    info = build_index([("kramatorsc", ga), ("kup", gb)], out)
    assert info["n_tiles"] == 8 and info["per_city"] == {"kramatorsc": 5, "kup": 3}

    # the split loader reads it: correct cities, count, window_size_m
    g = build_gallery_index(str(out), "EPSG:3857", 1000.0)
    assert len(g) == 8
    assert set(g.cities()) == {"kramatorsc", "kup"}
    t0 = g.tiles_for_city("kramatorsc")[0]
    assert t0.size_m == 1000.0 and t0.tile_id.startswith("kramatorsc:")


def _write_metric_grid(path, city, n, step_m, lat0=48.9, lon0=37.5):
    """A regular ``n x n`` tile grid spaced ``step_m`` TRUE metres apart (equirectangular)."""
    import math
    import h5py
    cl = math.cos(math.radians(lat0))
    with h5py.File(path, "w") as f:
        idx = 0
        for i in range(n):
            for j in range(n):
                g = f.create_group(f"{idx}_lvl0")
                g.create_dataset("ift_dino", data=np.zeros((1, 4, 2, 2), np.float16))
                g.attrs["tile_index"] = idx
                g.attrs["lat"] = lat0 + (j * step_m) / 110540.0
                g.attrs["lon"] = lon0 + (i * step_m) / (cl * 111320.0)
                g.attrs["window_size_m"] = 1000.0
                idx += 1


def test_grid_snap_checkerboard(tmp_path):
    pytest.importorskip("h5py")
    from geo_split_no_overlap.build_index import build_index

    # 6x6 tiles, 100 m apart → span 0..500 m. snap to 300 m → cells {0,1} per axis → 2x2 = 4 kept.
    ga = tmp_path / "map_dinov2_kramatorsc_s250m_d1024.h5"
    _write_metric_grid(ga, "kramatorsc", 6, 100.0)
    out = tmp_path / "checker.h5"
    info = build_index([("kramatorsc", ga)], out, grid_snap_m=300.0)
    assert info["n_dense"] == 36 and info["n_tiles"] == 4 and info["per_city"] == {"kramatorsc": 4}

    # the 4 kept tiles are in distinct 300 m cells (non-overlap): each pair is >= ~one cell apart.
    from geo_split_no_overlap.positive_selection import build_gallery_index
    g = build_gallery_index(str(out), "EPSG:3857", 1000.0)
    ts = g.tiles_for_city("kramatorsc")
    assert len(ts) == 4
    import itertools
    for a, b in itertools.combinations(ts, 2):
        assert abs(a.center_x - b.center_x) > 1.0 or abs(a.center_y - b.center_y) > 1.0
