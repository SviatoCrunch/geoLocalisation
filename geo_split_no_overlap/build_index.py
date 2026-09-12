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


def _checkerboard_keep(lat, lon, city, grid_snap_m) -> np.ndarray:
    """Boolean keep-mask sub-sampling the dense tiles to a NON-OVERLAPPING ``grid_snap_m``
    (TRUE-metre) grid: per city, snap each tile to a cell and keep the one tile nearest the
    cell centre → one tile per occupied cell (a "checkerboard"). A local equirectangular
    metric (per-city origin) is enough for a single city's small extent; the exact CRS is
    irrelevant since we only need a consistent grid to decimate on."""
    import math
    lat = np.asarray(lat, float); lon = np.asarray(lon, float); city = np.asarray(city, object)
    keep = np.zeros(len(lat), bool)
    for c in np.unique(city):
        idx = np.nonzero(city == c)[0]
        lat0, lon0 = float(lat[idx].mean()), float(lon[idx].mean())
        cl = math.cos(math.radians(lat0))
        x = (lon[idx] - lon0) * cl * 111320.0          # local TRUE metres (east)
        y = (lat[idx] - lat0) * 110540.0               # local TRUE metres (north)
        xmin, ymin = x.min(), y.min()
        cx = np.floor((x - xmin) / grid_snap_m).astype(int)
        cy = np.floor((y - ymin) / grid_snap_m).astype(int)
        ccx = xmin + (cx + 0.5) * grid_snap_m          # cell centre
        ccy = ymin + (cy + 0.5) * grid_snap_m
        d2 = (x - ccx) ** 2 + (y - ccy) ** 2
        best = {}
        for j in range(len(idx)):
            k = (int(cx[j]), int(cy[j]))
            if k not in best or d2[j] < d2[best[k]]:
                best[k] = j
        for j in best.values():
            keep[idx[j]] = True
    return keep


def build_index(galleries, out_path, grid_snap_m=None) -> dict:
    """galleries: list of (city, path). Writes lat/lon/city/tile_id/window_size_m datasets.

    ``grid_snap_m`` (TRUE metres) optionally sub-samples the dense tiles to a
    non-overlapping checkerboard (one tile per grid cell) — see :func:`_checkerboard_keep`.
    """
    import h5py

    lat, lon, win, city, tid = [], [], [], [], []
    for c, p in galleries:
        p = Path(p).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"gallery not found: {p}")
        with h5py.File(p, "r") as f:
            for k in _tile_groups(f):
                a = f[k].attrs
                lat.append(float(a["lat"]))
                lon.append(float(a["lon"]))
                win.append(float(a.get("window_size_m", np.nan)))
                city.append(c)
                tid.append(f"{c}:{k}")
    if not tid:
        raise SystemExit("no tiles found in the input galleries")

    n_dense = len(tid)
    if grid_snap_m is not None:
        keep = _checkerboard_keep(lat, lon, city, float(grid_snap_m))
        lat = list(np.asarray(lat)[keep]); lon = list(np.asarray(lon)[keep])
        win = list(np.asarray(win)[keep]); city = list(np.asarray(city, object)[keep])
        tid = list(np.asarray(tid, object)[keep])

    per_city = {}
    for c in city:
        per_city[c] = per_city.get(c, 0) + 1

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
        if grid_snap_m is not None:
            f.attrs["grid_snap_m"] = float(grid_snap_m)     # non-overlapping checkerboard step
            f.attrs["n_dense"] = int(n_dense)
    return {"out": str(out), "n_tiles": len(tid), "per_city": per_city,
            "n_dense": n_dense, "grid_snap_m": grid_snap_m}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gallery", nargs="+", required=True, help="city=path (raw group-per-tile H5)")
    ap.add_argument("--out", required=True, help="output tiles-index H5")
    ap.add_argument("--grid-snap-m", type=float, default=None,
                    help="sub-sample tiles to a NON-OVERLAPPING checkerboard of this TRUE-metre "
                         "step (one tile per cell); omit to keep the dense (overlapping) gallery")
    args = ap.parse_args(argv)
    galleries = [_city_and_path(a) for a in args.gallery]
    info = build_index(galleries, args.out, grid_snap_m=args.grid_snap_m)
    tail = (f"  (checkerboard {info['grid_snap_m']:g} m: {info['n_dense']}->{info['n_tiles']})"
            if info["grid_snap_m"] is not None else "")
    print(f"[ok] {info['n_tiles']} tiles -> {info['out']}  per-city={info['per_city']}{tail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
