"""Multi-city footprint run → ONE combined KMZ (SERVER-ONLY).

Loops several cities (each with its own COG + frames) through the same per-city pipeline
as :mod:`footprint_assoc.run`, sharing one sky-store, one gallery H5, one DINOv2 model,
and (optionally) one neural sky masker. Writes a single KMZ with a folder per frame,
grouped by ``city/<jpg-stem>``.

Config (YAML)::

    cities:
      - {city: cramatorsc, cog: /…/Krum_esri_48.5683N_37.6272E_z18_cog.tif, frames_dir: /…/gt_cramatorsc/GT_flat}
      - {city: kup,        cog: /…/Kup_esri_z18.tif,        frames_dir: /…/gt_kup/GT_flat}
      - {city: liman_day,  cog: /…/Liman_day_esri_z18.tif,  frames_dir: /…/gt_liman/GT_flat}
    sky_store: /home/ubuntu/work/sky_masks
    tiles_h5:  /home/ubuntu/work/out/gallery_h5/multicity_vlad_k32.h5
    out:       /home/ubuntu/work/out/gallery_kmz/footprint_multicity.kmz
    device: cuda
    neural: true          # SegFormer sky for cities/frames without a GT mask
    neural_model: nvidia/segformer-b0-finetuned-cityscapes-1024-1024
    sky_class: 10
    out_px: 224
    cell_sky_max: 0.5
    limit_per_city: 0

Run::

    uv run --with "torch==2.5.1" --with rasterio --with pillow --with numpy --with h5py \
           --with pyyaml --with transformers \
      python -m footprint_assoc.run_multicity --config /home/ubuntu/work/footprint_multicity.yaml
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .config import FootprintConfig
from .run import process_city, load_tiles, build_neural_masker
from . import kmz


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)

    import yaml
    conf = yaml.safe_load(Path(args.config).expanduser().read_text(encoding="utf-8"))
    device = conf.get("device", "cuda")
    out_px = int(conf.get("out_px", 224))
    cell_sky_max = float(conf.get("cell_sky_max", 0.5))
    limit = int(conf.get("limit_per_city", 0))
    tiles_h5 = conf.get("tiles_h5")

    from .extractors.dinov2 import DinoV2Extractor
    from sky_filter import MaskStore

    cfg = FootprintConfig(); cfg.validate()
    extractor = DinoV2Extractor(device=device)
    store = MaskStore(conf["sky_store"])
    nm = None
    if conf.get("neural"):
        nm = build_neural_masker(conf.get("neural_model", "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"),
                                 device, conf.get("sky_class", 10))

    all_est, all_assoc, all_geom, per_city = [], {}, {}, {}
    for c in conf["cities"]:
        city = c["city"]
        tiles, geom = load_tiles(tiles_h5, city) if tiles_h5 else ([], {})
        est, assoc, st = process_city(
            cfg, extractor, store, city=city, cog=c["cog"], frames_dir=c["frames_dir"],
            tiles=tiles, out_px=out_px, cell_sky_max=cell_sky_max, neural_masker=nm,
            include_unmasked=bool(c.get("include_unmasked", False)), limit=limit)
        all_est.extend(est); all_assoc.update(assoc); all_geom.update(geom)
        per_city[city] = {"stats": st, "status": dict(Counter(e.status for e in est))}
        print(f"[{city}] {st} status={per_city[city]['status']}", flush=True)

    out = kmz.write_kmz(conf["out"], all_est, tiles_geom=all_geom or None,
                        assoc=all_assoc or None, name="footprint_multicity")
    print(f"\n[ok] {len(all_est)} frames total -> {out}")
    for city, v in per_city.items():
        print(f"   {city}: {v['stats']} {v['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
