"""Precompute the fine-rerank token-grid store for ONE city (offline DINO over the real map).

Enumerates every non-overlapping 1000 m checkerboard cell of the city, slides a window WITHIN each
cell (step), builds the concentric pyramid [levels] per position, runs DINO once, and writes the token
grids to an H5 (streaming — RAM stays bounded). Positions are snapped + de-duplicated across cells, so
each map location is stored once. At query time map_rerank --store loads these grids instead of
re-running DINO (minutes/frame → seconds/frame).

Run::

    uv run --python 3.11 --with "torch==2.5.1" --with torchvision --with opencv-python-headless \
           --with rasterio --with h5py --with numpy --with tqdm python -m patch_rerank.precompute_store \
      --city kup --checker-index /…/dict/tiles_index_checker1000.h5 \
      --map /home/ubuntu/work/tif/Kup_esri_z18.tif --ref-h5 /…/query_kup_d1024.h5 \
      --step-m 250 --levels-m 1000 700 500 350 --output-px 518 --tile-size-m 1000 \
      --batch 8 --device cuda --out /…/rerank_store_kup.h5
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

from .map_source import latlon_to_merc, merc_to_latlon, open_src, read_pyramid_from_one, true_m_to_crs
from .map_dino import build_matched_extractor, extract_grids
from .map_rerank import cell_window_centres


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", required=True)
    ap.add_argument("--checker-index", required=True, help="tiles_index_checker1000.h5 (1000 m cells)")
    ap.add_argument("--map", required=True, help="city GeoTIFF")
    ap.add_argument("--ref-h5", required=True, help="a query/gallery H5 of this city (backbone/proj match)")
    ap.add_argument("--step-m", type=float, default=250.0)
    ap.add_argument("--levels-m", type=float, nargs="+", default=[1000, 700, 500, 350])
    ap.add_argument("--tile-size-m", type=float, default=1000.0)
    ap.add_argument("--output-px", type=int, default=518)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--proj-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import h5py
    from tqdm import tqdm

    # 1) city checkerboard cells
    with h5py.File(Path(args.checker_index).expanduser(), "r") as f:
        city = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f["city"][:]])
        lat = np.asarray(f["lat"][:], float); lon = np.asarray(f["lon"][:], float)
    m = city == args.city
    clat, clon = lat[m], lon[m]
    if clat.size == 0:
        raise SystemExit(f"no cells for city {args.city!r} in {args.checker_index}")

    # 2) within-cell positions, de-duplicated to a global snapped grid
    radius_m = args.tile_size_m / 2.0
    pos = {}                                              # (px,py) -> lat (for size conversion)
    for la, lo in zip(clat, clon):
        cx, cy = latlon_to_merc(la, lo)
        for (px, py) in cell_window_centres(cx, cy, la, radius_m, args.step_m):
            pos.setdefault((px, py), la)
    keys = sorted(pos)                                    # deterministic order
    print(f"[precompute] {args.city}: {clat.size} cells → {len(keys)} unique positions × "
          f"{len(args.levels_m)} levels = {len(keys) * len(args.levels_m)} crops", flush=True)

    ext, proj, patch, backbone, D = build_matched_extractor(args.ref_h5, args.device, args.proj_seed)
    print(f"[dino] backbone={backbone} D={D} projector={'yes' if proj else 'NO'}", flush=True)
    src = open_src(args.map)
    dev = args.device

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fout = h5py.File(out, "w")
    pxa = np.array([k[0] for k in keys], float)
    pya = np.array([k[1] for k in keys], float)
    lla = np.array([merc_to_latlon(k[0], k[1]) for k in keys], float)   # (n,2) lat,lon
    fout.create_dataset("px", data=pxa); fout.create_dataset("py", data=pya)
    fout.create_dataset("lat", data=lla[:, 0]); fout.create_dataset("lon", data=lla[:, 1])
    fout.attrs["city"] = args.city
    fout.attrs["step_m"] = args.step_m
    fout.attrs["levels_m"] = [float(x) for x in args.levels_m]
    fout.attrs["output_px"] = int(args.output_px)
    fout.attrs["tile_size_m"] = args.tile_size_m
    fout.attrs["backbone"] = backbone
    fout.attrs["projector"] = bool(proj)
    idx_of = {k: i for i, k in enumerate(keys)}

    buf_imgs, buf_meta = [], []                           # meta = (posidx, L)

    def _flush():
        if not buf_imgs:
            return
        grids = extract_grids(buf_imgs, ext, proj, patch, dev, amp=args.amp)
        for (pi, L), g in zip(buf_meta, grids):
            fout.create_dataset(f"p{pi}/l{int(L)}", data=g.half().numpy(), chunks=True)
        buf_imgs.clear(); buf_meta.clear()

    for (px, py) in tqdm(keys, desc=f"store:{args.city}", unit="pos"):
        pyr = read_pyramid_from_one(src, px, py, args.levels_m, pos[(px, py)], args.output_px)
        pi = idx_of[(px, py)]
        for L in args.levels_m:
            buf_imgs.append(pyr[L]); buf_meta.append((pi, L))
            if len(buf_imgs) >= args.batch:
                _flush()
    _flush()
    fout.attrs["n_positions"] = len(keys)
    fout.close(); src.close()
    print(f"[ok] store -> {out}  ({len(keys)} positions, {len(args.levels_m)} levels)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
