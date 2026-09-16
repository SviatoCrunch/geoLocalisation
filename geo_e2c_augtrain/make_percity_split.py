"""Per-city stratified split that keeps EVERY frame with a positive and puts EVERY city in
train/val/test - the split for weighted (IoU) multi-city training.

The strict leakage-free splitter (geo_split_no_overlap) cannot split a DENSE city (kramatorsc) under
1000 m IoU positives, because all its frames share checker cells => one giant leakage component that
must go whole to one split (=> either all-in-train, or mass-excluded). Meaningful IoU weights REQUIRE
that 1000 m footprint, so leakage-free + per-city + weights + all-frames is unsatisfiable together.

This splitter drops the leakage-free component constraint (kept: all frames, all cities, min excluded,
IoU weights): each city's frames that have >=1 overlap_weighted positive are shuffled (seeded) and cut
70/15/15. Positives + IoU weights are generated at train time by build_split_relevance with an
overlap_weighted config (this file only decides membership). Output: {"train":[point_id], "val":[...],
"test":[...]} keyed "<city>:<stem>". NOTE: not leakage-free (a checker cell can be positive for a
train AND a test frame of the same dense city) - fine for a seen-area deployment.

Run::

    uv run --python 3.11 --with h5py --with numpy --with shapely python -m geo_e2c_augtrain.make_percity_split \
      --gt kramatorsc=~/work/gt_cramatorsc/GT_flat kup=~/work/gt_kup/GT_flat liman_day=~/work/gt_liman/GT_flat \
      --tiles ~/work/out/gallery_h5/dict/tiles_index_checker1000.h5 \
      --min-overlap 0.05 --query-size-m 1000 --ratios 0.70 0.15 0.15 --seed 0 \
      --out ~/work/out/gallery_h5/split_iou_percity/split.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _kv(a):
    c, p = a.split("=", 1)
    return c, p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", nargs="+", required=True, help="city=dir (raw drone frames)")
    ap.add_argument("--tiles", required=True, help="checker geometry H5 (lat/lon/city/tile_id/window_size_m)")
    ap.add_argument("--min-overlap", type=float, default=0.05)
    ap.add_argument("--query-size-m", type=float, default=1000.0)
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.70, 0.15, 0.15])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid-crs", default="EPSG:3857")
    ap.add_argument("--tile-size-m", type=float, default=1000.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import numpy as np
    from geo_split_no_overlap.positive_selection import (build_gallery_index, build_geo_points,
                                                        create_positive_selector,
                                                        materialize_positive_sets)

    gallery = build_gallery_index(args.tiles, args.grid_crs, args.tile_size_m)
    points = build_geo_points([f"{c}={d}" for c, d in (_kv(a) for a in args.gt)], args.grid_crs)
    selector = create_positive_selector({"strategy": "overlap_weighted",
                                         "params": {"min_overlap": args.min_overlap,
                                                    "query_size_m": args.query_size_m,
                                                    "same_city_only": True}})
    mps = materialize_positive_sets(points, gallery, selector)

    by_city = defaultdict(list)                                 # city -> point_ids WITH a positive
    dropped = defaultdict(int)
    for p in points:
        if p.point_id in mps.point_to_tile_ids:
            by_city[p.city].append(p.point_id)
        else:
            dropped[p.city] += 1

    rtr, rva, rte = args.ratios
    out = {"train": [], "val": [], "test": []}
    rng = np.random.RandomState(args.seed)
    print(f"[percity] tiles={args.tiles} min_overlap={args.min_overlap} query={args.query_size_m} "
          f"ratios={args.ratios} seed={args.seed}", flush=True)
    for city in sorted(by_city):
        ids = sorted(by_city[city])                            # deterministic before shuffle
        rng.shuffle(ids)
        n = len(ids)
        ntr = int(round(n * rtr)); nva = int(round(n * rva))
        nva = min(nva, n - ntr - 1) if n - ntr >= 2 else nva   # ensure test >=1 when possible
        tr, va, te = ids[:ntr], ids[ntr:ntr + nva], ids[ntr + nva:]
        out["train"] += tr; out["val"] += va; out["test"] += te
        print(f"    {city:<12} kept={n} (dropped_no_pos={dropped.get(city,0)}) "
              f"-> train {len(tr)} / val {len(va)} / test {len(te)}", flush=True)

    p = Path(args.out).expanduser(); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({k: sorted(v) for k, v in out.items()}), encoding="utf-8")
    print(f"[ok] train {len(out['train'])} / val {len(out['val'])} / test {len(out['test'])} -> {p}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
