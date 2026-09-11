"""Real footprint run + KMZ (SERVER-ONLY: torch DINOv2 + rasterio COG).

Integration entry point — wires the isolated pieces for an actual run:
  sky_filter masks  +  CogCropSource (COG)  +  DinoV2Extractor  +  footprint_assoc pipeline  +  KMZ.

Per UAV frame (centre parsed from the filename ``<lat>_<lon>``):
  1. DINOv2 tokens of the UAV frame; DROP sky tokens using the precomputed keep-mask
     (or a neural SegFormer mask if --neural and no precomputed mask; else skip).
  2. Concentric satellite crops at each scale → DINOv2 tokens (levels).
  3. estimate_frame (cascade) → optional association with gallery tiles → KMZ.

Single-city CLI here; :mod:`footprint_assoc.run_multicity` loops several cities into one KMZ.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from .config import FootprintConfig
from .schemas import PyramidFeatures, LevelFeatures
from .pipeline import estimate_frame
from .association import associate
from .mask_query import sky_filtered_query
from . import kmz

_IMG_EXTS = {".jpg", ".jpeg"}
_LAT = (44.0, 53.0)


def _strip_leading_id(stem: str) -> str:
    toks = stem.split("_")
    return "_".join(toks[1:]) if len(toks) > 1 and toks[0].isdigit() else stem


def _mask_key(city: str, stem: str, store_keys: set):
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
    for i in range(len(nums) - 1):
        lat, lon = nums[i], nums[i + 1]
        if _LAT[0] <= lat <= _LAT[1] and 20.0 <= lon <= 45.0:
            return lat, lon
    return None


def load_tiles(tiles_h5, city):
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


def process_city(cfg, extractor, store, *, city, cog, frames_dir, tiles=None,
                 out_px=224, cell_sky_max=0.5, neural_masker=None,
                 include_unmasked=False, limit=0, verbose=True):
    """Run one city; return (estimates, assoc, stats). Appends nothing to disk."""
    from PIL import Image
    from .extractors.cog_crop import CogCropSource

    crops = CogCropSource(cog, out_px=out_px)
    scales = cfg.scales()
    store_keys = set(store.frame_ids())
    tiles = tiles or []

    frames = sorted(p for p in Path(frames_dir).expanduser().rglob("*")
                    if p.suffix.lower() in _IMG_EXTS)
    estimates, assoc = [], {}
    n_done = n_nocoord = n_nomask = 0
    for p in frames:
        ll = _parse_latlon(p.stem)
        if ll is None:
            n_nocoord += 1
            continue
        lat, lon = ll
        mkey = _mask_key(city, p.stem, store_keys)
        keep_px = store.load(mkey) if mkey else None
        uav = np.asarray(Image.open(p).convert("RGB"))
        if keep_px is None:
            if neural_masker is not None:
                keep_px = neural_masker.mask(p.stem, uav)          # SegFormer sky
            elif not include_unmasked:
                n_nomask += 1
                continue
        frame_id = f"{city}/{p.stem}"                              # KMZ label (city + jpg name)
        query = sky_filtered_query(extractor, uav, keep_px, cell_sky_max)
        levels = [LevelFeatures(float(s), *extractor.extract(crops.crop(lat, lon, s)))
                  for s in scales]
        pyr = PyramidFeatures(frame_id, lat, lon, query, tuple(levels))
        est = estimate_frame(pyr, cfg)
        estimates.append(est)
        if tiles:
            assoc[frame_id] = associate(est, tiles, cfg)
        n_done += 1
        if verbose:
            print(f"[{city} {n_done}] {p.stem} best={est.best_scale} status={est.status} "
                  f"conf={est.confidence:.2f}", flush=True)
        if limit and n_done >= limit:
            break
    return estimates, assoc, {"done": n_done, "no_coord": n_nocoord, "no_mask": n_nomask}


def build_neural_masker(model_id, device, sky_class):
    from .config import FootprintConfig  # noqa
    from sky_filter.config import SkyFilterConfig
    from sky_filter.maskers.neural import NeuralSkyMasker
    return NeuralSkyMasker(SkyFilterConfig(neural_model=model_id, device=device),
                           sky_class_id=int(sky_class))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cog", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--city", required=True)
    ap.add_argument("--sky-store", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiles-h5", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-px", type=int, default=224)
    ap.add_argument("--cell-sky-max", type=float, default=0.5)
    ap.add_argument("--include-unmasked", action="store_true")
    ap.add_argument("--neural", action="store_true", help="SegFormer sky for frames without a GT mask")
    ap.add_argument("--neural-model", default="nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    ap.add_argument("--sky-class", type=int, default=10, help="sky class id (Cityscapes=10)")
    args = ap.parse_args(argv)

    from .extractors.dinov2 import DinoV2Extractor
    from sky_filter import MaskStore

    cfg = FootprintConfig(); cfg.validate()
    extractor = DinoV2Extractor(device=args.device)
    store = MaskStore(args.sky_store)
    tiles, geom = ([], {})
    if args.tiles_h5:
        tiles, geom = load_tiles(args.tiles_h5, args.city)
    nm = build_neural_masker(args.neural_model, args.device, args.sky_class) if args.neural else None

    estimates, assoc, st = process_city(
        cfg, extractor, store, city=args.city, cog=args.cog, frames_dir=args.frames_dir,
        tiles=tiles, out_px=args.out_px, cell_sky_max=args.cell_sky_max,
        neural_masker=nm, include_unmasked=args.include_unmasked, limit=args.limit)

    out = kmz.write_kmz(args.out, estimates, tiles_geom=geom or None, assoc=assoc or None,
                        name=f"footprint_{args.city}")
    from collections import Counter
    print(f"\n[ok] {st['done']} frames -> {out}")
    print(f"     status: {dict(Counter(e.status for e in estimates))} | "
          f"no-coord={st['no_coord']} no-mask={st['no_mask']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
