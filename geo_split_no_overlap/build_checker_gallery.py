"""Materialize a STANDALONE non-overlapping checkerboard gallery from a dense gallery.

The dense galleries (``map_dinov2_<city>_s250m_d1024.h5``) are a group-per-tile grid of
*overlapping* 1000 m windows at 250 m stride. The regular non-overlap checkerboard is normally
a separate geometry index (``checker1000_noverlap_<city>.h5``: ``tile_id/lat/lon/city/…``) whose
``tile_id`` values point *into* the dense gallery — so the checkerboard cannot be used without
keeping every overlapping tile around (that is where its embeddings live).

This tool breaks that coupling. Given a dense gallery + the checkerboard's ``tile_id`` selection,
it copies ONLY the kept tiles' groups (``ift_dino`` + attrs) into a new file AND bakes the geometry
index (top-level ``lat/lon/city/tile_id/window_size_m`` datasets) into the SAME file. The result is
self-contained:

* :class:`geo_e2c_train.data.TileGridLoader` reads embeddings from the tile groups (unchanged), and
* :func:`geo_split_no_overlap.build_gallery_index` / ``patch_rerank --checker-index`` /
  ``map_rerank --dense-index`` read the geometry from the top-level datasets.

So one file replaces ``(dense gallery + checker index)`` for the coarse train/eval AND the fine
rerank (which slides fresh DINO over the GeoTIFF, not the dense grid). After building + verifying,
the dense overlapping galleries can be deleted locally (S3 keeps the source of truth).

Geometry is read from each kept tile's OWN ``group.attrs`` (the source of truth for the embeddings),
not from the checker index — the index is used only to SELECT which tiles to keep, so the baked
geometry can never drift from the copied features.

Run (per city, or pass several cities at once — one output file each)::

    uv run --with h5py --with numpy python -m geo_split_no_overlap.build_checker_gallery \
      --dense   kramatorsc=/…/map_dinov2_kramatorsc_s250m_d1024.h5 \
      --checker kramatorsc=/…/checker1000_noverlap_kramatorsc.h5 \
      --out     kramatorsc=/…/checker_gallery_kramatorsc.h5

``--checker`` may also be a single combined index (e.g. ``tiles_index_checker1000.h5`` with a
``city`` dataset); tiles are dispatched to the matching ``--dense``/``--out`` city.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _kv(arg: str):
    if "=" not in arg:
        raise SystemExit(f"expected city=path, got {arg!r}")
    c, p = arg.split("=", 1)
    return c, Path(p).expanduser()


def _checker_path(arg: str) -> Path:
    """Accept a bare ``path`` OR ``city=path`` (the city label is ignored — it is read from the
    file's ``city`` dataset). Lets ``--checker`` mirror the ``city=path`` form of ``--dense``/``--out``
    without the ``city=`` prefix leaking into the path."""
    if "=" in arg:
        head, tail = arg.split("=", 1)
        if "/" not in head and "\\" not in head:            # a city token, not a drive/path with '='
            return Path(tail).expanduser()
    return Path(arg).expanduser()


def _decode(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def load_checker_ids(checker_paths) -> dict:
    """Read one or more checker-index H5s → ``{city: [tile_id, ...]}`` (file order preserved).

    Each index must have top-level ``city`` and ``tile_id`` datasets (the schema written by
    :mod:`geo_split_no_overlap.build_index` and the regular non-overlap checkerboard builder).
    """
    import h5py

    per: dict = {}
    for p in checker_paths:
        p = Path(p).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"checker index not found: {p}")
        with h5py.File(p, "r") as f:
            for req in ("city", "tile_id"):
                if req not in f:
                    raise KeyError(f"{p} lacks dataset '{req}' (not a tiles-index H5?)")
            city = [_decode(c) for c in f["city"][:]]
            tid = [_decode(t) for t in f["tile_id"][:]]
        for c, t in zip(city, tid):
            per.setdefault(c, []).append(t)
    return per


def build_one(dense_path, city: str, keep_ids, out_path) -> dict:
    """Copy the kept tiles' groups out of ``dense_path`` into a standalone checker gallery.

    ``keep_ids`` are ``"<city>:<groupkey>"`` strings (ids of the checkerboard tiles). Only ids of
    ``city`` are used. Writes tile groups (verbatim) + baked top-level geometry datasets.
    """
    import h5py

    dense_path = Path(dense_path).expanduser()
    out_path = Path(out_path).expanduser()
    if not dense_path.exists():
        raise FileNotFoundError(f"dense gallery not found: {dense_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lat, lon, win, carr, tarr = [], [], [], [], []
    missing = []
    str_dt = h5py.string_dtype("utf-8")
    with h5py.File(dense_path, "r") as fin, h5py.File(out_path, "w") as fout:
        for k, v in fin.attrs.items():                     # keep backbone/desc_dim/proj_seed/etc.
            fout.attrs[k] = v
        for tid in keep_ids:
            c, key = tid.split(":", 1)
            if c != city:
                continue
            if key not in fin or "ift_dino" not in fin[key]:
                missing.append(tid)
                continue
            fin.copy(key, fout, name=key)                  # group = ift_dino + attrs, verbatim
            a = fin[key].attrs
            lat.append(float(a["lat"]))
            lon.append(float(a["lon"]))
            win.append(float(a.get("window_size_m", np.nan)))
            carr.append(c)
            tarr.append(tid)
        if missing:
            raise SystemExit(f"{city}: {len(missing)} checker tile(s) absent from {dense_path.name} "
                             f"(e.g. {missing[:3]}) — dense/checker mismatch")
        if not tarr:
            raise SystemExit(f"{city}: no checker tiles selected (check --checker city / tile_id city)")
        fout.create_dataset("lat", data=np.asarray(lat, "f8"))
        fout.create_dataset("lon", data=np.asarray(lon, "f8"))
        fout.create_dataset("window_size_m", data=np.asarray(win, "f8"))
        fout.create_dataset("city", data=np.asarray(carr, dtype=object), dtype=str_dt)
        fout.create_dataset("tile_id", data=np.asarray(tarr, dtype=object), dtype=str_dt)
        fout.attrs["kind"] = "checker_gallery"             # gallery + baked geometry index in one file
        fout.attrs["n_tiles"] = len(tarr)
        fout.attrs["cities"] = [city]
        fout.attrs["source_dense"] = str(dense_path)
    return {"city": city, "out": str(out_path), "n_tiles": len(tarr)}


def build(dense_by_city, checker_paths, out_by_city) -> list:
    per = load_checker_ids(checker_paths)
    infos = []
    for city, dpath in dense_by_city.items():
        if city not in out_by_city:
            raise SystemExit(f"no --out for city {city!r}")
        keep = per.get(city)
        if not keep:
            raise SystemExit(f"no checker tiles for city {city!r} in the given --checker index(es); "
                             f"have {sorted(per)}")
        infos.append(build_one(dpath, city, keep, out_by_city[city]))
    return infos


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dense", nargs="+", required=True, help="city=dense gallery H5 (group-per-tile)")
    ap.add_argument("--checker", nargs="+", required=True,
                    help="checker-index H5(s): per-city checker1000_noverlap_<city>.h5 or a combined "
                         "tiles_index_checker1000.h5 (selection = its tile_id list). Bare path or "
                         "city=path (the city label is ignored — read from the file).")
    ap.add_argument("--out", nargs="+", required=True, help="city=output checker gallery H5")
    args = ap.parse_args(argv)

    dense_by_city = dict(_kv(a) for a in args.dense)
    out_by_city = dict(_kv(a) for a in args.out)
    checker_paths = [_checker_path(p) for p in args.checker]
    infos = build(dense_by_city, checker_paths, out_by_city)
    for info in infos:
        print(f"[ok] {info['city']}: {info['n_tiles']} checker tiles -> {info['out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
