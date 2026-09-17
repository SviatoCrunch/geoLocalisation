"""Build the native ``S`` cache: read 1000/500/250 m crops straight from the COG, run the SAME
production DINOv2 + RandomProjector as the galleries, and store the frozen per-cell per-group sums.

Per base tile (21 DINO views, shared backbone):
    1 × DINO(1000 m)  → n=1 cell
    4 × DINO(500 m)   → n=2 cells (row-major, i=north/j=west)
   16 × DINO(250 m)   → n=4 cells; each 250 m grid split into 2×2 quadrants → 64 n=8 cells
No separate 125 m DINO forward — n=8 reuses the 250 m grids' quadrants.

The DINO forward and COG read are injected (``dino_batch_fn`` / ``read_crop_fn``) so the ordering /
source logic is unit-tested with a mock; the CLI wires the real ``map_extract`` machinery.

Run (server, GPU + COG)::

    uv run --python 3.11 --with "torch==2.5.1" --with torchvision --with h5py --with numpy \
           --with rasterio --with pyproj --with pillow --with tqdm \
      python -m native_map_pyramid.build_cache \
        --index   /home/ubuntu/work/out/gallery_h5/geo_iso_noverlap/tiles/tiles_index_dense.h5 \
        --cog     kramatorsc=/home/ubuntu/work/tif/Krum_esri_48.5683N_37.6272E_z18_cog.tif \
                  kup=/home/ubuntu/work/tif/Kup_esri_z18.tif \
                  liman_day=/home/ubuntu/work/tif/Liman_day_esri_z18.tif \
        --assign  /home/ubuntu/work/out/gallery_h5/dict/k32.pt \
        --backbone dinov2_vitg14 --dino-facet value --output-px 840 \
        --desc-dim 1024 --proj-seed 0 --tile-size-m 1000 \
        --out /home/ubuntu/work/out/gallery_h5/geo_iso_noverlap/native_cache/S_native.h5 --device cuda --amp
"""
from __future__ import annotations

import argparse

import numpy as np

from .geometry import (native_view_plan, quadrant_token_slices, n8_flat_index,
                       VIEW_SCALES_M, N_VIEWS_PER_TILE)
from . import cache as C


# ── injectable I/O ────────────────────────────────────────────────────────────────────────
def read_crop(src, minx, miny, maxx, maxy, output_px):
    """Read one EPSG:3857 bbox from an open rasterio dataset → (output_px,output_px,3) uint8 RGB.

    Same window/resample convention as ``map_extract.pyramid.PyramidContextSampler._read_from``
    (window_from_bounds + bilinear + boundless), so a 500/250 m child crop is byte-consistent with
    how the 1000 m gallery tiles were read."""
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds as window_from_bounds
    window = window_from_bounds(minx, miny, maxx, maxy, src.transform)
    arr = src.read([1, 2, 3], window=window, out_shape=(3, output_px, output_px),
                   resampling=Resampling.bilinear, boundless=True, fill_value=0)
    return arr.transpose(1, 2, 0).astype(np.uint8)


def make_dino_batch_fn(dino, projector, device, patch_size, *, amp=False, batch=16):
    """Return ``fn(list[img HWC uint8]) -> list[grid (Hd,Wd,Dv) torch]`` using the PRODUCTION
    preprocessing + RandomProjector (reuses ``map_extract`` verbatim). Chunked by ``batch`` for
    VRAM; chunking does not change the per-crop result."""
    import torch
    from map_extract.extract import _tile_to_batch, _extract_level_features

    def fn(imgs):
        grids = []
        for s in range(0, len(imgs), batch):
            chunk = imgs[s:s + batch]
            b, h_r, w_r = _tile_to_batch(chunk, device, patch_size, preprocess=None)
            feats = _extract_level_features(dino, b, h_r, w_r, projector, amp)   # list (1,Dv,h_r,w_r)
            for f in feats:
                grids.append(torch.from_numpy(f[0].transpose(1, 2, 0)).contiguous())  # (h_r,w_r,Dv)
        return grids

    return fn


# ── core: per-tile frozen cell sums (S) from native crops ───────────────────────────────────
def compute_tile_cell_sums(center_x, center_y, tile_true_m, true_scale, output_px,
                           read_crop_fn, dino_batch_fn, agg, eps=1e-8):
    """One base tile → (S (85,K,Dv), n_dino_views). ``n_dino_views`` MUST equal 21 (1+4+16)."""
    import torch
    from siam_e2c_model.vendored.stage2_core import tokens_to_group_sum

    plan = native_view_plan(center_x, center_y, tile_true_m, true_scale)
    imgs, keys = [], []
    for s in VIEW_SCALES_M:                                   # (1000, 500, 250)
        for c in plan[s]:
            imgs.append(read_crop_fn(c.minx, c.miny, c.maxx, c.maxy, output_px))
            keys.append((s, c.i, c.j))
    grids = dino_batch_fn(imgs)                               # 21 grids, shared backbone
    if len(grids) != len(imgs):
        raise RuntimeError(f"dino_batch_fn returned {len(grids)} grids for {len(imgs)} crops")
    dev = agg.assign.weight.device                            # run aggregation on agg's device
    grids = [g.to(dev) for g in grids]
    by = {k: g for k, g in zip(keys, grids)}

    K = agg.n_groups
    Dv = grids[0].shape[-1]
    S8 = torch.zeros(64, K, Dv, device=dev)
    S4 = torch.zeros(16, K, Dv, device=dev)
    S2 = torch.zeros(4, K, Dv, device=dev)
    S1 = torch.zeros(1, K, Dv, device=dev)

    c1 = plan[1000.0][0]
    S1[0] = tokens_to_group_sum(by[(1000.0, c1.i, c1.j)], agg, eps)          # n=1
    for c in plan[500.0]:                                                    # n=2
        S2[c.flat] = tokens_to_group_sum(by[(500.0, c.i, c.j)], agg, eps)
    for c in plan[250.0]:                                                    # n=4 + n=8
        g = by[(250.0, c.i, c.j)]
        S4[c.flat] = tokens_to_group_sum(g, agg, eps)
        Hd, Wd = g.shape[0], g.shape[1]
        for dy, dx, rs, cs in quadrant_token_slices(Hd, Wd):                 # 2×2 quadrants → n=8
            S8[n8_flat_index(c.i, c.j, dy, dx)] = tokens_to_group_sum(g[rs, cs], agg, eps)

    S = torch.cat([S8, S4, S2, S1], dim=0)                                   # (85,K,Dv) canonical
    return S, len(imgs)


# ── CLI (server: GPU + COG) ─────────────────────────────────────────────────────────────────
def _kv(a):
    c, p = a.split("=", 1)
    return c, p


def _build_frozen_agg(assign_path, k, d_token, device):
    """Rebuild the SAME frozen CellSuperVLAD (identity ψ/φ, frozen assign) the model uses."""
    from siam_e2c_model.adapters import stage3 as S3
    from siam_e2c_model.vendored.stage2_core import CellSuperVLAD, apply_assignment_init
    aw = S3.load_assign_weight(assign_path, k=k, d=d_token)
    agg = CellSuperVLAD(d_token=d_token, n_groups=k, value_proj="identity",
                        phi_proj="identity", n_ghost=0, assign_bias=False)
    apply_assignment_init(agg, aw, {"method": "file"})
    agg.set_assign_frozen(True)
    agg.to(device).eval()
    return agg, aw


def main(argv=None) -> int:
    import torch
    import rasterio
    from pyproj import Transformer
    from tqdm import tqdm
    import h5py
    from map_extract.dino import build_dino_extractor, RandomProjector
    from map_extract.geometry import mercator_true_scale

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", nargs="+", required=True,
                    help="geometry index H5(s) with lat/lon/city/tile_id (one combined file, or the "
                         "per-city checker files the split uses — pass all of them)")
    ap.add_argument("--cog", nargs="+", required=True, help="city=path to the source COG")
    ap.add_argument("--assign", required=True, help="dict/k32.pt (frozen VLAD assignment)")
    ap.add_argument("--backbone", default="dinov2_vitg14")
    ap.add_argument("--dino-layer", type=int, default=None)
    ap.add_argument("--dino-facet", default="value")
    ap.add_argument("--output-px", type=int, default=840)
    ap.add_argument("--desc-dim", type=int, default=1024)
    ap.add_argument("--proj-seed", type=int, default=0)
    ap.add_argument("--tile-size-m", type=float, default=1000.0)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--batch-crops", type=int, default=16)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap #tiles (debug)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    cogs = dict(_kv(a) for a in args.cog)
    dino = build_dino_extractor(args.backbone, layer=args.dino_layer, facet=args.dino_facet,
                                device=args.device, norm_descs=False)
    patch_size = int(getattr(dino, "patch_size", 14))
    projector = RandomProjector(args.desc_dim, args.proj_seed)
    agg, aw = _build_frozen_agg(args.assign, args.k, args.desc_dim, args.device)
    dino_batch_fn = make_dino_batch_fn(dino, projector, args.device, patch_size,
                                       amp=args.amp, batch=args.batch_crops)
    to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform

    lat_l, lon_l, city, tid = [], [], [], []
    for ipath in args.index:                                   # one combined index OR per-city checkers
        with h5py.File(ipath, "r") as f:
            la = np.asarray(f["lat"], float); lo = np.asarray(f["lon"], float)
            cy = [c.decode() if isinstance(c, bytes) else str(c) for c in f["city"][:]]
            ti = [t.decode() if isinstance(t, bytes) else str(t) for t in f["tile_id"][:]] \
                if "tile_id" in f else [f"{cy[i]}:row{i}" for i in range(len(la))]
        lat_l.append(la); lon_l.append(lo); city += cy; tid += ti
    lat = np.concatenate(lat_l); lon = np.concatenate(lon_l)

    keep = [i for i in range(len(tid)) if city[i] in cogs]
    if args.limit:
        keep = keep[:args.limit]
    print(f"[native-cache] {len(keep)} tiles across {sorted(set(city[i] for i in keep))}", flush=True)

    srcs, true_scale = {}, {}
    for c, p in cogs.items():
        srcs[c] = rasterio.open(p)
        b = srcs[c].bounds
        cx, cy = (b.left + b.right) / 2, (b.bottom + b.top) / 2
        _, clat = Transformer.from_crs(srcs[c].crs, "EPSG:4326", always_xy=True).transform(cx, cy)
        true_scale[c] = mercator_true_scale(float(clat))        # SAME single map-centre scale as extraction

    S_all = np.zeros((len(keep), C.N_CELLS_TOTAL, args.k, args.desc_dim), dtype=np.float16)
    out_ids = []
    for row, i in enumerate(tqdm(keep, desc="native S", unit="tile")):
        c = city[i]
        cx, cy = to_merc(lon[i], lat[i])
        Sfn = lambda a, b_, d, e, px, src=srcs[c]: read_crop(src, a, b_, d, e, px)
        with torch.no_grad():
            S, nv = compute_tile_cell_sums(cx, cy, args.tile_size_m, true_scale[c], args.output_px,
                                           Sfn, dino_batch_fn, agg)
        if nv != N_VIEWS_PER_TILE:
            raise RuntimeError(f"tile {tid[i]} produced {nv} DINO views, expected {N_VIEWS_PER_TILE}")
        S_all[row] = S.cpu().numpy().astype(np.float16)
        out_ids.append(tid[i])
    for s in srcs.values():
        s.close()

    proj_attrs = projector.h5_attrs()
    ident = C.build_fingerprint(
        backbone=args.backbone, dino_layer=args.dino_layer, dino_facet=args.dino_facet,
        output_px=args.output_px, tile_size_m=args.tile_size_m, patch_size=patch_size,
        projection=proj_attrs.get("projection", "none"), projection_seed=args.proj_seed,
        projection_in_dim=int(proj_attrs.get("projection_in_dim", args.desc_dim)),
        projection_out_dim=int(proj_attrs.get("projection_out_dim", args.desc_dim)),
        n_groups=args.k, d_value=args.desc_dim, vlad_dict_id=C.vlad_dict_id(aw))
    C.write_cache(args.out, S_all, out_ids, ident,
                  extra_meta={"n_tiles": len(out_ids), "cities": sorted(set(city[i] for i in keep)),
                              "true_scale": {c: true_scale[c] for c in srcs}})
    print(f"[ok] native cache → {args.out}  ({len(out_ids)} tiles, {S_all.nbytes/1e9:.2f} GB fp16)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
