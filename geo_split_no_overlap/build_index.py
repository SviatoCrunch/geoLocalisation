"""Build a lightweight tiles-index H5 (geometry only) from raw group-per-tile DINO galleries.

The split's ``build_gallery_index`` reads a multicity-schema H5 with top-level datasets
``lat``/``lon``/``city`` (+ optional ``tile_id``/``window_size_m``). The map_extract galleries are
group-per-tile (``<idx>_lvl0/ift_dino`` with ``lat``/``lon``/``window_size_m`` in ``group.attrs``),
so this tool reads just those attrs (no feature data → fast, MBs) and writes the datasets the split
needs. Use it to make a tiles_h5 for the CURRENT galleries so the split runs on the real geometry.

Run::

    uv run --with h5py --with numpy python -m geo_split_no_overlap.build_index \
      --gallery kramatorsc=/…/map_dinov2_kramatorsc_s250m_d1024.h5 \
                kup=/…/map_dinov2_kup_s250m_d1024.h5 \
                liman_day=/…/map_dinov2_liman_day_s250m_d1024.h5 \
      --out /…/gallery_h5/dict/tiles_index_s250m.h5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _city_and_path(arg: str):
    if "=" not in arg:
        raise SystemExit(f"--gallery expects city=path, got {arg!r}")
    city, p = arg.split("=", 1)
    return city, Path(p).expanduser()


def _tile_groups(f):
    keys = [k for k in f.keys() if hasattr(f[k], "keys") and "ift_dino" in f[k]]

    def _order(k):
        ti = f[k].attrs.get("tile_index")
        return (0, int(ti)) if ti is not None else (1, k)
    return sorted(keys, key=_order)


def build_index(galleries, out_path) -> dict:
    """galleries: list of (city, path). Writes lat/lon/city/tile_id/window_size_m datasets."""
    import h5py

    lat, lon, win, city, tid = [], [], [], [], []
    per_city = {}
    for c, p in galleries:
        p = Path(p).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"gallery not found: {p}")
        n0 = len(tid)
        with h5py.File(p, "r") as f:
            for k in _tile_groups(f):
                a = f[k].attrs
                lat.append(float(a["lat"]))
                lon.append(float(a["lon"]))
                win.append(float(a.get("window_size_m", np.nan)))
                city.append(c)
                tid.append(f"{c}:{k}")
        per_city[c] = len(tid) - n0
    if not tid:
        raise SystemExit("no tiles found in the input galleries")

    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    str_dt = h5py.string_dtype("utf-8")
    with h5py.File(out, "w") as f:
        f.create_dataset("lat", data=np.asarray(lat, "f8"))
        f.create_dataset("lon", data=np.asarray(lon, "f8"))
        f.create_dataset("window_size_m", data=np.asarray(win, "f8"))
        f.create_dataset("city", data=np.asarray(city, dtype=object), dtype=str_dt)
        f.create_dataset("tile_id", data=np.asarray(tid, dtype=object), dtype=str_dt)
        f.attrs["kind"] = "tiles_index_geometry"
        f.attrs["n_tiles"] = len(tid)
        f.attrs["cities"] = list(per_city)
        f.attrs["city_counts"] = [per_city[c] for c in per_city]
    return {"out": str(out), "n_tiles": len(tid), "per_city": per_city}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gallery", nargs="+", required=True, help="city=path (raw group-per-tile H5)")
    ap.add_argument("--out", required=True, help="output tiles-index H5")
    args = ap.parse_args(argv)
    galleries = [_city_and_path(a) for a in args.gallery]
    info = build_index(galleries, args.out)
    print(f"[ok] {info['n_tiles']} tiles -> {info['out']}  per-city={info['per_city']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
