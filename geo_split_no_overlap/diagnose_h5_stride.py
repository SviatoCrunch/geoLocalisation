"""Tile-stride check for ANY map H5 — read-only, no split dependency.

Reports the nearest-neighbour spacing of tile centres for one or more H5 galleries,
handling BOTH schemas:

  * multicity gallery  — top-level ``lat``/``lon`` datasets (built by our new code);
  * raw DINO map H5    — one group per tile keyed ``<idx>_lvl<L>`` with ``lat``/``lon``
    (and ``window_size_m``) in ``group.attrs`` (tif_dino_extract output).

Spacing is in EPSG:3857 grid-metres (what the extractor's ``stride_m`` steps in) plus the
true-metre value (grid * cos(lat)). Use it to compare the SOURCE per-city galleries
against the combined multicity gallery and see where a stride diverges.

Run::

    uv run --with numpy --with h5py python -m geo_split_no_overlap.diagnose_h5_stride \
      --h5 /home/ubuntu/work/out/gallery_h5/map_dinov2_1000_kramatorsc_fp16.h5 \
           /home/ubuntu/work/out/gallery_h5/map_dinov2_1000_kup.h5 \
           /home/ubuntu/work/out/gallery_h5/map_dinov2_1000_liman_day.h5 \
           /home/ubuntu/work/out/gallery_h5/multicity_vlad_k32.h5
"""
from __future__ import annotations

import argparse
import math

import h5py
import numpy as np

_R = 6378137.0


def _merc(lat, lon):
    return _R * np.radians(lon), _R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def _centers(path):
    """(lat, lon) of unique tile centres, from either schema."""
    with h5py.File(path, "r") as f:
        if "lat" in f and isinstance(f["lat"], h5py.Dataset):
            return np.asarray(f["lat"], float), np.asarray(f["lon"], float)
        lat, lon, seen = [], [], set()

        def visit(name, obj):
            if isinstance(obj, h5py.Group) and "lat" in obj.attrs and "lon" in obj.attrs:
                key = obj.attrs.get("tile_index", name)          # levels share a centre
                if key in seen:
                    return
                seen.add(key)
                lat.append(float(obj.attrs["lat"]))
                lon.append(float(obj.attrs["lon"]))

        f.visititems(visit)
        return np.asarray(lat, float), np.asarray(lon, float)


def _nn(cx, cy, sample=500):
    n = len(cx)
    idx = np.arange(n) if n <= sample else np.linspace(0, n - 1, sample).astype(int)
    d = []
    for i in idx:
        dd = np.sqrt((cx - cx[i]) ** 2 + (cy - cy[i]) ** 2)
        dd[i] = np.inf
        d.append(dd.min())
    return np.asarray(d)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", nargs="+", required=True)
    args = ap.parse_args(argv)

    for p in args.h5:
        try:
            lat, lon = _centers(p)
        except Exception as e:                       # noqa: BLE001 — report, keep going
            print(f"{p}\n  [error] {e}")
            continue
        if len(lat) < 2:
            print(f"{p}\n  n={len(lat)} (need >=2 centres)")
            continue
        cx, cy = _merc(lat, lon)
        d = _nn(cx, cy)
        cl = math.cos(math.radians(float(np.mean(lat))))
        print(f"{p}\n  n={len(lat)} lat~{np.mean(lat):.2f} | NN grid: median={np.median(d):.0f} "
              f"min={d.min():.0f} p90={np.percentile(d, 90):.0f} | "
              f"true median={np.median(d) * cl:.0f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
