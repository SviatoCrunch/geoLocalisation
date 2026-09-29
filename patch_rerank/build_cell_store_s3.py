"""Build a per-CELL DINOv3-sat pyramid embedding store for one city and stream it to S3.

For every 1000 m checkerboard cell of the city (id/lat/lon read from a tiles index), slide a window
WITHIN the cell (``--step-m``), cut the concentric pyramid (``--levels-m``) at each position from the
real GeoTIFF, embed each crop with DINOv3-sat (``map_dino3``), and write ONE self-contained H5 per cell
(``p{i}/l{L}`` token grids fp16 + per-position px/py/lat/lon + config attrs). Each cell H5 is uploaded
to S3 immediately and (``--delete-after``) removed locally, so peak local disk ≈ one cell — a whole
city builds with a few GB free. A JSON manifest ``_index.json`` (cell_id → S3 key + geometry + config
fingerprint) is uploaded last so the search side resolves cells by index without listing.

At query time the reranker fetches only the shortlist's top-K cell objects by id (companion reader,
separate). This replaces the monolithic local store (``precompute_store``) with an addressable,
disk-unbounded, S3-resident one.

Run (Kupiansk, shortlist-union cells only — the minimal set the current search needs)::

    HF_TOKEN=hf_… uv run --python 3.11 --with "torch==2.5.1" --with torchvision --with transformers \
      --with opencv-python-headless --with rasterio --with h5py --with numpy --with tqdm --with boto3 \
      python -m patch_rerank.build_cell_store_s3 \
      --cell-index /…/dict/tiles_index_dense.h5 --city kup \
      --map /home/ubuntu/work/tif/Kup_esri_z18.tif \
      --cells-from-shortlist /…/kup/shortlist_kup_k70.json \
      --step-m 250 --levels-m 1000 700 500 350 --output-px 256 \
      --s3-uri s3://geo-reference/embeddings/kup/rerank_cells_dinov3sat \
      --stage-dir /home/ubuntu/work/out/_cellstage --delete-after --device cuda --amp
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

from .map_source import latlon_to_merc, merc_to_latlon, open_src, read_pyramid_from_one
from .map_rerank import cell_window_centres
from .map_dino3 import DEFAULT_MODEL, build_dinov3_extractor, extract_grids_v3


def _safe(cell_id: str) -> str:
    """cell_id → filesystem/S3-safe stem (kup:996_lvl0 → kup_996_lvl0)."""
    return cell_id.replace(":", "_").replace("/", "_")


def _read_cells(cell_index: str, city: str):
    """→ list of (cell_id, lat, lon) for the city. Accepts EITHER a flat tiles index
    (tile_id/city/lat/lon datasets) OR a map_extract H5 (per-tile groups with lat/lon[/tile_index]
    attrs, e.g. kup_prodaction.h5) — auto-detected. For map_extract, cell_id = ``<city>:<tile_index>_lvl0``
    (falls back to the group key) so it matches the dense/checker id style."""
    import h5py
    with h5py.File(Path(cell_index).expanduser(), "r") as f:
        if "tile_id" in f:                                   # flat tiles index
            ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f["tile_id"][:]]
            cc = [x.decode() if isinstance(x, bytes) else str(x) for x in f["city"][:]]
            lat = np.asarray(f["lat"][:], float); lon = np.asarray(f["lon"][:], float)
            return [(ids[i], float(lat[i]), float(lon[i])) for i in range(len(ids)) if cc[i] == city]
        cells = []                                           # map_extract H5 (per-tile groups)
        for k in f.keys():
            g = f[k]
            if not (hasattr(g, "attrs") and "lat" in g.attrs and "lon" in g.attrs):
                continue
            ti = g.attrs.get("tile_index", None)
            cid = f"{city}:{int(ti)}_lvl0" if ti is not None else f"{city}:{k}"
            cells.append((cid, float(g.attrs["lat"]), float(g.attrs["lon"])))
        if not cells:
            raise SystemExit(f"{cell_index}: not a tiles-index (no 'tile_id') and no map_extract "
                             f"groups with lat/lon attrs (keys={list(f.keys())[:5]}…)")
        return cells


def _shortlist_union(path: str) -> set:
    sj = json.loads(Path(path).expanduser().read_text())
    out = set()
    for entry in sj["shortlist"].values():
        out.update(entry.get("cells", []))
    return out


def _build_cell_h5(out_path: Path, cell_id: str, clat: float, clon: float, ext, src, args) -> int:
    """Write one self-contained cell H5 (streaming DINO). Returns the number of window positions."""
    import h5py
    radius_m = args.tile_size_m / 2.0
    cx, cy = latlon_to_merc(clat, clon)
    positions = cell_window_centres(cx, cy, clat, radius_m, args.step_m)     # snapped merc centres
    lla = np.array([merc_to_latlon(px, py) for (px, py) in positions], float).reshape(-1, 2)

    fout = h5py.File(out_path, "w")
    fout.create_dataset("px", data=np.array([p[0] for p in positions], float))
    fout.create_dataset("py", data=np.array([p[1] for p in positions], float))
    fout.create_dataset("lat", data=lla[:, 0]); fout.create_dataset("lon", data=lla[:, 1])
    fout.attrs.update({"cell_id": cell_id, "city": args.city, "step_m": args.step_m,
                       "levels_m": [float(x) for x in args.levels_m], "tile_size_m": args.tile_size_m,
                       "output_px": int(args.output_px), "patch": int(ext["patch"]),
                       "D": int(ext["D"]), "n_prefix": int(ext["n_prefix"]),
                       "backbone": ext["model_id"], "cell_lat": clat, "cell_lon": clon,
                       "n_positions": len(positions)})

    buf_imgs, buf_meta = [], []                                             # (posidx, L)

    def _flush():
        if not buf_imgs:
            return
        grids = extract_grids_v3(buf_imgs, ext, args.output_px, args.device, amp=args.amp)
        for (pi, L), g in zip(buf_meta, grids):
            fout.create_dataset(f"p{pi}/l{int(L)}", data=g.numpy().astype(np.float16), chunks=True)
        buf_imgs.clear(); buf_meta.clear()

    for i, (px, py) in enumerate(positions):
        pyr = read_pyramid_from_one(src, px, py, args.levels_m, clat, args.output_px)
        for L in args.levels_m:
            buf_imgs.append(pyr[L]); buf_meta.append((i, int(L)))
            if len(buf_imgs) >= args.batch:
                _flush()
    _flush()
    fout.close()
    return len(positions)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell-index", required=True, help="tiles index with tile_id/city/lat/lon (cells)")
    ap.add_argument("--city", required=True)
    ap.add_argument("--map", required=True, help="city GeoTIFF read for the pyramid crops")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="DINOv3 HF repo (satellite variant)")
    ap.add_argument("--cells-from-shortlist", default=None,
                    help="restrict to the union of cells referenced by this shortlist json (minimal build)")
    ap.add_argument("--step-m", type=float, default=250.0)
    ap.add_argument("--levels-m", type=float, nargs="+", default=[1000, 700, 500, 350])
    ap.add_argument("--tile-size-m", type=float, default=1000.0)
    ap.add_argument("--output-px", type=int, default=256, help="crop px (multiple of patch 16)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--s3-uri", required=True, help="s3://bucket/prefix target (city/ appended)")
    ap.add_argument("--stage-dir", default="/tmp/cellstage", help="local staging dir for cell H5s")
    ap.add_argument("--delete-after", action="store_true", help="remove each cell H5 after upload")
    ap.add_argument("--overwrite", action="store_true", help="rebuild even if the S3 object exists")
    ap.add_argument("--limit", type=int, default=0, help="build only the first N cells (smoke)")
    args = ap.parse_args(argv)

    from tqdm import tqdm
    from s3_gt_sync.core import make_s3_client, parse_s3_uri

    cells = _read_cells(args.cell_index, args.city)
    if not cells:
        raise SystemExit(f"no cells for city {args.city!r} in {args.cell_index}")
    if args.cells_from_shortlist:
        keep = _shortlist_union(args.cells_from_shortlist)
        cells = [c for c in cells if c[0] in keep]
        print(f"[cells] restricted to shortlist union: {len(cells)} of {len(keep)} referenced", flush=True)
    if args.limit:
        cells = cells[:args.limit]
    print(f"[build] {args.city}: {len(cells)} cells × step {args.step_m}m × {len(args.levels_m)} levels "
          f"→ {args.s3_uri}/{args.city}", flush=True)

    bucket, prefix = parse_s3_uri(args.s3_uri)
    prefix = prefix.rstrip("/")
    s3 = make_s3_client()
    stage = Path(args.stage_dir).expanduser(); stage.mkdir(parents=True, exist_ok=True)

    ext = build_dinov3_extractor(args.model, args.device)
    print(f"[dino3] {ext['model_id']} patch={ext['patch']} D={ext['D']} n_prefix={ext['n_prefix']}",
          flush=True)
    src = open_src(args.map)

    manifest, built, skipped, bytes_up = {}, 0, 0, 0
    for (cell_id, clat, clon) in tqdm(cells, desc=f"cells:{args.city}", unit="cell"):
        key = f"{prefix}/{args.city}/cells/{_safe(cell_id)}.h5"
        if not args.overwrite:
            try:
                h = s3.head_object(Bucket=bucket, Key=key)                  # resume: skip existing
                manifest[cell_id] = {"key": key, "lat": clat, "lon": clon, "bytes": int(h["ContentLength"])}
                skipped += 1
                continue
            except Exception:
                pass
        local = stage / f"{_safe(cell_id)}.h5"
        n_pos = _build_cell_h5(local, cell_id, clat, clon, ext, src, args)
        sz = local.stat().st_size
        s3.upload_file(str(local), bucket, key)
        bytes_up += sz; built += 1
        manifest[cell_id] = {"key": key, "lat": clat, "lon": clon, "n_pos": n_pos, "bytes": sz}
        if args.delete_after:
            local.unlink(missing_ok=True)

    src.close()
    index = {"city": args.city, "s3_prefix": f"s3://{bucket}/{prefix}/{args.city}",
             "config": {"model": ext["model_id"], "patch": ext["patch"], "D": ext["D"],
                        "n_prefix": ext["n_prefix"], "step_m": args.step_m,
                        "levels_m": [float(x) for x in args.levels_m], "tile_size_m": args.tile_size_m,
                        "output_px": int(args.output_px)},
             "n_cells": len(manifest), "cells": manifest}
    idx_key = f"{prefix}/{args.city}/_index.json"
    s3.put_object(Bucket=bucket, Key=idx_key, Body=json.dumps(index).encode("utf-8"),
                  ContentType="application/json")
    print(f"[ok] built={built} skipped={skipped} uploaded={bytes_up / 1e9:.2f}GB | "
          f"manifest -> s3://{bucket}/{idx_key} ({len(manifest)} cells)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
