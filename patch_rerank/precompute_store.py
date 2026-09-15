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


def _center_crop(g, frac):
    """Central fraction of a (H,W,D) token grid → the concentric level crop (D unchanged)."""
    H, W = g.shape[0], g.shape[1]
    h = max(1, int(round(frac * H))); w = max(1, int(round(frac * W)))
    y0, x0 = (H - h) // 2, (W - w) // 2
    return g[y0:y0 + h, x0:x0 + w, :].contiguous()


def _build_from_dense(args) -> int:
    """ZERO-DINO store: level-1000 = the dense tile's ift_dino (positions align on the 250 grid);
    smaller levels = centre-crops of that grid. No GeoTIFF, no DINO — just read dense H5 + slice."""
    import h5py
    import numpy as np
    from tqdm import tqdm
    from geo_e2c_train.data import TileGridLoader

    if not (args.dense_h5 and args.dense_index):
        raise SystemExit("--from-dense needs --dense-h5 city=path and --dense-index")
    dcity, dpath = args.dense_h5.split("=", 1)

    with h5py.File(Path(args.checker_index).expanduser(), "r") as f:
        cc = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f["city"][:]])
        clat = np.asarray(f["lat"][:], float)[cc == args.city]
        clon = np.asarray(f["lon"][:], float)[cc == args.city]
    if clat.size == 0:
        raise SystemExit(f"no cells for {args.city!r} in {args.checker_index}")
    with h5py.File(Path(args.dense_index).expanduser(), "r") as f:
        ids = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f["tile_id"][:]])
        dcin = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f["city"][:]])
        dlat = np.asarray(f["lat"][:], float); dlon = np.asarray(f["lon"][:], float)
    ds = dcin == args.city
    dids = ids[ds]
    dmx = np.array([latlon_to_merc(a, o) for a, o in zip(dlat[ds], dlon[ds])], float).reshape(-1, 2)

    radius_m = args.tile_size_m / 2.0
    posinfo = {}
    for la, lo in zip(clat, clon):
        cx, cy = latlon_to_merc(la, lo)
        for (px, py) in cell_window_centres(cx, cy, la, radius_m, args.step_m):
            posinfo.setdefault((px, py), la)
    keys, tile_for = [], {}
    for (px, py), la in posinfo.items():                       # align each position to its dense tile
        d2 = (dmx[:, 0] - px) ** 2 + (dmx[:, 1] - py) ** 2
        j = int(np.argmin(d2))
        if float(d2[j]) ** 0.5 <= true_m_to_crs(140.0, la):    # within ~half a dense stride
            keys.append((px, py)); tile_for[(px, py)] = str(dids[j])
    keys.sort()
    print(f"[from-dense] {args.city}: {clat.size} cells, {len(posinfo)} positions, "
          f"{len(keys)} matched to dense × {len(args.levels_m)} levels (0 DINO)", flush=True)
    if not keys:
        raise SystemExit("no positions matched a dense tile (check step/index/city)")

    tiles = TileGridLoader({dcity: dpath}, cache=False)
    with h5py.File(dpath, "r") as f:
        backbone = str(f.attrs.get("backbone", "?"))
    out = Path(args.out).expanduser(); out.parent.mkdir(parents=True, exist_ok=True)
    fout = h5py.File(out, "w")
    fout.create_dataset("px", data=np.array([k[0] for k in keys], float))
    fout.create_dataset("py", data=np.array([k[1] for k in keys], float))
    lla = np.array([merc_to_latlon(k[0], k[1]) for k in keys], float).reshape(-1, 2)
    fout.create_dataset("lat", data=lla[:, 0]); fout.create_dataset("lon", data=lla[:, 1])
    g0 = tiles.grid(tile_for[keys[0]])
    fout.attrs.update({"city": args.city, "step_m": args.step_m,
                       "levels_m": [float(x) for x in args.levels_m], "tile_size_m": args.tile_size_m,
                       "output_px": int(g0.shape[1]), "backbone": backbone,
                       "projector": True, "from_dense": True, "n_positions": len(keys)})
    for i, (px, py) in enumerate(tqdm(keys, desc=f"from-dense:{args.city}", unit="pos")):
        g = tiles.grid(tile_for[(px, py)])                     # native dense grid = level 1000
        for L in args.levels_m:
            gc = g if float(L) >= args.tile_size_m else _center_crop(g, float(L) / args.tile_size_m)
            fout.create_dataset(f"p{i}/l{int(L)}", data=gc.half().numpy(), chunks=True)
    fout.close(); tiles.close()
    print(f"[ok] from-dense store -> {out} ({len(keys)} positions)", flush=True)
    return 0


def _cells_and_dense(args):
    """Shared setup: within-cell positions (dedup) + nearest dense tile per position."""
    import h5py
    import numpy as np
    with h5py.File(Path(args.checker_index).expanduser(), "r") as f:
        cc = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f["city"][:]])
        clat = np.asarray(f["lat"][:], float)[cc == args.city]
        clon = np.asarray(f["lon"][:], float)[cc == args.city]
    if clat.size == 0:
        raise SystemExit(f"no cells for {args.city!r} in {args.checker_index}")
    with h5py.File(Path(args.dense_index).expanduser(), "r") as f:
        ids = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f["tile_id"][:]])
        dcin = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f["city"][:]])
        dlat = np.asarray(f["lat"][:], float); dlon = np.asarray(f["lon"][:], float)
    ds = dcin == args.city
    dids = ids[ds]
    dmx = np.array([latlon_to_merc(a, o) for a, o in zip(dlat[ds], dlon[ds])], float).reshape(-1, 2)
    radius_m = args.tile_size_m / 2.0
    posinfo = {}
    for la, lo in zip(clat, clon):
        cx, cy = latlon_to_merc(la, lo)
        for (px, py) in cell_window_centres(cx, cy, la, radius_m, args.step_m):
            posinfo.setdefault((px, py), la)
    keys, tile_for = [], {}
    for (px, py), la in posinfo.items():
        d2 = (dmx[:, 0] - px) ** 2 + (dmx[:, 1] - py) ** 2
        j = int(np.argmin(d2))
        if float(d2[j]) ** 0.5 <= true_m_to_crs(140.0, la):
            keys.append((px, py)); tile_for[(px, py)] = str(dids[j])
    keys.sort()
    return clat.size, posinfo, keys, tile_for


def _build_hybrid(args) -> int:
    """level-1000 = dense tile (reuse, no DINO); smaller levels 900..300 = FRESH DINO zoom from map."""
    import h5py
    import numpy as np
    from tqdm import tqdm
    from geo_e2c_train.data import TileGridLoader
    if not (args.dense_h5 and args.dense_index and args.map and args.ref_h5):
        raise SystemExit("--hybrid needs --dense-h5, --dense-index, --map, --ref-h5")
    dcity, dpath = args.dense_h5.split("=", 1)

    ncells, posinfo, keys, tile_for = _cells_and_dense(args)
    if not keys:
        raise SystemExit("no positions matched a dense tile (check step/index/city)")
    small = [L for L in args.levels_m if float(L) < args.tile_size_m]     # fresh (900..300)
    has_1000 = any(float(L) >= args.tile_size_m for L in args.levels_m)   # reuse from dense
    print(f"[hybrid] {args.city}: {ncells} cells, {len(keys)} positions | level-1000 reuse=dense, "
          f"fresh levels={small}", flush=True)

    ext, proj, patch, backbone, D = build_matched_extractor(args.ref_h5, args.device, args.proj_seed)
    print(f"[dino] backbone={backbone} D={D} projector={'yes' if proj else 'NO'}", flush=True)
    src = open_src(args.map)
    tiles = TileGridLoader({dcity: dpath}, cache=False)

    out = Path(args.out).expanduser(); out.parent.mkdir(parents=True, exist_ok=True)
    fout = h5py.File(out, "w")
    fout.create_dataset("px", data=np.array([k[0] for k in keys], float))
    fout.create_dataset("py", data=np.array([k[1] for k in keys], float))
    lla = np.array([merc_to_latlon(k[0], k[1]) for k in keys], float).reshape(-1, 2)
    fout.create_dataset("lat", data=lla[:, 0]); fout.create_dataset("lon", data=lla[:, 1])
    fout.attrs.update({"city": args.city, "step_m": args.step_m,
                       "levels_m": [float(x) for x in args.levels_m], "tile_size_m": args.tile_size_m,
                       "output_px": int(args.output_px), "backbone": backbone,
                       "projector": bool(proj), "hybrid": True, "n_positions": len(keys)})

    buf_imgs, buf_meta = [], []                          # (posidx, L) for fresh small levels

    def _flush():
        if not buf_imgs:
            return
        grids = extract_grids(buf_imgs, ext, proj, patch, args.device, amp=args.amp)
        for (pi, L), g in zip(buf_meta, grids):
            fout.create_dataset(f"p{pi}/l{int(L)}", data=g.half().numpy(), chunks=True)
        buf_imgs.clear(); buf_meta.clear()

    for i, (px, py) in enumerate(tqdm(keys, desc=f"hybrid:{args.city}", unit="pos")):
        if has_1000:                                     # level-1000 straight from dense (no DINO)
            g1000 = tiles.grid(tile_for[(px, py)])
            fout.create_dataset(f"p{i}/l1000", data=g1000.half().numpy(), chunks=True)
        if small:                                        # smaller levels = fresh zoom from the map
            pyr = read_pyramid_from_one(src, px, py, small, posinfo[(px, py)], args.output_px)
            for L in small:
                buf_imgs.append(pyr[L]); buf_meta.append((i, L))
                if len(buf_imgs) >= args.batch:
                    _flush()
    _flush()
    fout.close(); src.close(); tiles.close()
    print(f"[ok] hybrid store -> {out} ({len(keys)} positions)", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", required=True)
    ap.add_argument("--checker-index", required=True, help="tiles_index_checker1000.h5 (1000 m cells)")
    ap.add_argument("--map", default=None, help="city GeoTIFF (fresh-DINO mode)")
    ap.add_argument("--ref-h5", default=None, help="a query/gallery H5 of this city (fresh-DINO backbone/proj)")
    ap.add_argument("--from-dense", action="store_true",
                    help="ZERO-DINO approx: level-1000 = dense tile; smaller levels = centre-crops (coarse)")
    ap.add_argument("--hybrid", action="store_true",
                    help="level-1000 = dense tile (reuse); smaller levels = FRESH DINO zoom from the map")
    ap.add_argument("--dense-h5", default=None, help="city=dense d1024 H5 (with --from-dense)")
    ap.add_argument("--dense-index", default=None, help="tiles_index_dense.h5 geometry (with --from-dense)")
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

    if args.hybrid:
        return _build_hybrid(args)
    if args.from_dense:
        return _build_from_dense(args)
    if not (args.map and args.ref_h5):
        raise SystemExit("fresh-DINO mode needs --map and --ref-h5 (or use --from-dense/--hybrid)")

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
