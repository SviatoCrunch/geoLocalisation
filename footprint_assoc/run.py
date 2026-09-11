"""Real footprint run + KMZ (SERVER-ONLY: torch DINOv2 + rasterio COG).

Integration entry point — wires the isolated pieces for an actual run:
  sky_filter masks  +  CogCropSource (COG)  +  DinoV2Extractor  +  footprint_assoc pipeline  +  KMZ.

Per UAV frame (centre parsed from the filename ``<lat>_<lon>``):
  1. DINOv2 tokens of the UAV frame; DROP sky tokens using the precomputed keep-mask
     (frames without a mask are skipped unless --include-unmasked).
  2. Concentric satellite crops at each scale → DINOv2 tokens (levels).
  3. estimate_frame (cascade) → optional association with gallery tiles → KMZ.

Example
-------
    uv run --with torch --with rasterio --with pillow --with numpy --with h5py \
      python -m footprint_assoc.run \
        --cog   /home/ubuntu/work/tif/Krum_esri_48.5683N_37.6272E_z18_cog.tif \
        --frames-dir /home/ubuntu/work/gt_cramatorsc \
        --city cramatorsc --sky-store /home/ubuntu/work/sky_masks \
        --tiles-h5 /home/ubuntu/work/out/gallery_h5/multicity_vlad_k32.h5 \
        --out /home/ubuntu/work/out/footprint_kram.kmz --limit 20 --device cuda
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from .config import FootprintConfig
from .schemas import PyramidFeatures, QueryFeatures, LevelFeatures
from .pipeline import estimate_frame
from .association import associate
from .mask_query import sky_filtered_query          # local helper (below)
from . import kmz

_IMG_EXTS = {".jpg", ".jpeg"}                         # UAV frames only (never mask PNGs)
_LAT = (44.0, 53.0)                                   # Ukraine band — reject corrupt stems


def _strip_leading_id(stem: str) -> str:
    """``100_48.59_37.59`` -> ``48.59_37.59`` (COCO named frames without the id prefix)."""
    toks = stem.split("_")
    return "_".join(toks[1:]) if len(toks) > 1 and toks[0].isdigit() else stem


def _mask_key(city: str, stem: str, store_keys: set):
    """Resolve the sky-store key for a jpg stem (try full, then id-stripped)."""
    for cand in (stem, _strip_leading_id(stem)):
        k = f"{city}:{cand}"
        if k in store_keys:
            return k
    return None


def _parse_latlon(stem: str):
    nums = []
    for t in re.split(r"[_,]", stem):
        try:
            nums.append(float(t))
        except ValueError:
            pass
    # first pair that looks like (lat, lon) in range
    for i in range(len(nums) - 1):
        lat, lon = nums[i], nums[i + 1]
        if _LAT[0] <= lat <= _LAT[1] and 20.0 <= lon <= 45.0:
            return lat, lon
    return None


def _load_tiles(tiles_h5, city):
    import h5py
    with h5py.File(tiles_h5, "r") as f:
        lat = np.asarray(f["lat"], float)
        lon = np.asarray(f["lon"], float)
        cty = [c.decode() if isinstance(c, bytes) else str(c) for c in f["city"][:]]
        tid = [t.decode() if isinstance(t, bytes) else str(t) for t in f["tile_id"][:]]
        win = np.asarray(f["window_size_m"], float) if "window_size_m" in f else np.full(len(lat), 1000.0)
    tiles, geom = [], {}
    for i in range(len(lat)):
        if cty[i] != city:
            continue
        size = float(win[i]) if np.isfinite(win[i]) and win[i] > 0 else 1000.0
        tiles.append((tid[i], float(lat[i]), float(lon[i]), size))
        geom[tid[i]] = (float(lat[i]), float(lon[i]), size)
    return tiles, geom


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cog", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--city", required=True)
    ap.add_argument("--sky-store", required=True)
    ap.add_argument("--out", required=True, help="output KMZ path")
    ap.add_argument("--tiles-h5", default=None, help="gallery H5 for footprint→tile association")
    ap.add_argument("--limit", type=int, default=0, help="process at most N frames (0 = all)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-px", type=int, default=224, help="satellite crop resolution")
    ap.add_argument("--cell-sky-max", type=float, default=0.5)
    ap.add_argument("--include-unmasked", action="store_true",
                    help="also process frames without a sky mask (no sky drop) instead of skipping")
    args = ap.parse_args(argv)

    from .extractors.dinov2 import DinoV2Extractor
    from .extractors.cog_crop import CogCropSource
    from sky_filter import MaskStore

    cfg = FootprintConfig()
    cfg.validate()
    scales = cfg.scales()
    extractor = DinoV2Extractor(device=args.device)
    crops = CogCropSource(args.cog, out_px=args.out_px)
    store = MaskStore(args.sky_store)

    frames = sorted(p for p in Path(args.frames_dir).expanduser().rglob("*")
                    if p.suffix.lower() in _IMG_EXTS)
    tiles, geom = ([], {})
    if args.tiles_h5:
        tiles, geom = _load_tiles(args.tiles_h5, args.city)
    store_keys = set(store.frame_ids())

    from PIL import Image
    estimates, assoc = [], {}
    n_done = n_skip_nocoord = n_skip_nomask = 0
    for p in frames:
        ll = _parse_latlon(p.stem)
        if ll is None:
            n_skip_nocoord += 1
            continue
        lat, lon = ll
        mkey = _mask_key(args.city, p.stem, store_keys)
        keep_px = store.load(mkey) if mkey else None
        frame_id = mkey or f"{args.city}:{p.stem}"
        if keep_px is None and not args.include_unmasked:
            n_skip_nomask += 1
            continue

        uav = np.asarray(Image.open(p).convert("RGB"))
        query = sky_filtered_query(extractor, uav, keep_px, args.cell_sky_max)
        levels = []
        for s in scales:
            lg, lp = extractor.extract(crops.crop(lat, lon, s))
            levels.append(LevelFeatures(scale_m=float(s), global_vec=lg, patch_grid=lp))
        pyr = PyramidFeatures(frame_id, lat, lon, query, tuple(levels))
        est = estimate_frame(pyr, cfg)
        estimates.append(est)
        if tiles:
            assoc[frame_id] = associate(est, tiles, cfg)
        n_done += 1
        print(f"[{n_done}] {frame_id} best={est.best_scale} status={est.status} conf={est.confidence:.2f}",
              flush=True)
        if args.limit and n_done >= args.limit:
            break

    out = kmz.write_kmz(args.out, estimates, tiles_geom=geom or None,
                        assoc=assoc or None, name=f"footprint_{args.city}")
    from collections import Counter
    st = Counter(e.status for e in estimates)
    print(f"\n[ok] {n_done} frames -> {out}")
    print(f"     status: {dict(st)} | skipped no-coord={n_skip_nocoord} no-mask={n_skip_nomask}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
