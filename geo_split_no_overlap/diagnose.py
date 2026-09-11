"""Coverage diagnostic — WHY do so few query points get positives?

Reads the SAME config as ``cli build`` (single-gallery or per-city) and, using the exact
loaders the split uses, reports per city:

  * gallery vs point city labels — catches ``same_city_only`` label mismatches (e.g.
    gallery ``cramatorsc`` vs config key ``kramatorsc``) that silently zero a whole city;
  * tile stride — nearest-neighbour spacing of tile centres in grid-metres AND true-metres
    (true = grid * cos(lat)); tells you the real footprint the positive rules must span;
  * nearest same-city tile distance per point — are frames inside gallery coverage at all,
    or did the drone fly beyond the mapped area (no strategy can help then)?

This module is NOT imported by the split pipeline; it is a read-only report.

Run::

    uv run --with numpy --with h5py --with shapely --with pyyaml \
      python -m geo_split_no_overlap.diagnose --config /home/ubuntu/work/split_pyr250.yaml
"""
from __future__ import annotations

import argparse
import math
from collections import Counter

import numpy as np

from .config import load_config
from .positive_selection import build_gallery_index, build_geo_points


def _load(cfg):
    """(gallery, points) exactly as the split sees them (single-gallery or per-city)."""
    if cfg.is_per_city():
        from . import multicity
        g, pts, _ = multicity.load_merged(cfg)   # same label logic as the real run
        return g, pts
    g = build_gallery_index(cfg.tiles_h5, cfg.grid_crs, cfg.tile_size_m)
    pts = build_geo_points(cfg.gt, cfg.grid_crs)
    return g, pts


def _stride_grid(cx, cy, sample=400):
    """Median/min/p90 nearest-neighbour spacing (grid units), sampled for speed."""
    n = len(cx)
    if n < 2:
        return (float("nan"),) * 3
    idx = np.arange(n) if n <= sample else np.linspace(0, n - 1, sample).astype(int)
    d = []
    for i in idx:
        dd = np.sqrt((cx - cx[i]) ** 2 + (cy - cy[i]) ** 2)
        dd[i] = np.inf
        d.append(dd.min())
    d = np.asarray(d)
    return float(np.median(d)), float(d.min()), float(np.percentile(d, 90))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    g, pts = _load(cfg)

    gal_cities = dict(Counter(t.city for t in g.all_tiles()))
    pt_cities = dict(Counter(p.city for p in pts))
    print(f"gallery cities : {gal_cities}")
    print(f"point   cities : {pt_cities}")
    only_pts = sorted(set(pt_cities) - set(gal_cities))
    if only_pts:
        n = sum(pt_cities[c] for c in only_pts)
        print(f"[!] point labels with NO gallery tiles: {only_pts} -> {n} pts get 0 positives "
              f"(same_city_only mismatch)")

    xy = {c: (np.array([t.center_x for t in g.tiles_for_city(c)], float),
              np.array([t.center_y for t in g.tiles_for_city(c)], float),
              float(np.mean([t.lat for t in g.tiles_for_city(c)])))
          for c in g.cities()}

    print("--- tile stride (grid-m | true-m = grid*cos(lat)) ---")
    for c, (cx, cy, lat) in xy.items():
        med, mn, p90 = _stride_grid(cx, cy)
        cl = math.cos(math.radians(lat))
        print(f"  [{c}] {len(cx)} tiles lat~{lat:.2f} | NN median={med:.0f} (true {med*cl:.0f}) "
              f"min={mn:.0f} (true {mn*cl:.0f}) p90={p90:.0f} (true {p90*cl:.0f})")

    print("--- nearest same-city tile distance per point (grid-m | true-m) ---")
    nomatch = 0
    for c in sorted(pt_cities):
        e = xy.get(c)
        pc = [p for p in pts if p.city == c]
        if e is None or len(e[0]) == 0:
            nomatch += len(pc)
            print(f"  [{c}] NO tiles for this label -> {len(pc)} pts get 0 positives (LABEL MISMATCH)")
            continue
        cx, cy, lat = e
        cl = math.cos(math.radians(lat))
        ds = np.array([np.sqrt((cx - p.x) ** 2 + (cy - p.y) ** 2).min() for p in pc])
        med = float(np.median(ds))
        print(f"  [{c}] {len(ds)} pts | nearest median={med:.0f} (true {med*cl:.0f}) | "
              f"<=190m:{(ds <= 190).mean()*100:.0f}% <=380m:{(ds <= 380).mean()*100:.0f}% "
              f"<=760m:{(ds <= 760).mean()*100:.0f}%")
    print(f"points whose city label has NO gallery tiles: {nomatch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
