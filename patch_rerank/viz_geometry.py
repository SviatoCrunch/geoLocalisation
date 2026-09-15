"""Debug KMZ of the geometry: the straight checkerboard (1000 m non-overlapping cells) + the within-
cell sliding positions and their concentric pyramid [1000..300] — to eyeball it in Google Earth.

All cells are drawn as 1000 m squares; for the first ``--sample-cells`` cells the within-cell sliding
positions (points) and the concentric pyramid squares at each position are drawn too. Pure geometry
(h5py + math), no GPU/torch.

Run::

    uv run --with h5py --with numpy python -m patch_rerank.viz_geometry \
      --checker-index /…/dict/tiles_index_checker1000.h5 --city kup \
      --step-m 250 --levels-m 1000 900 800 700 600 500 400 300 --tile-size-m 1000 \
      --sample-cells 2 --out /…/geometry_kup.kmz
"""
from __future__ import annotations

import argparse
import math
import zipfile
from pathlib import Path

from .map_source import latlon_to_merc, merc_to_latlon
from .map_rerank import cell_window_centres

_HEAD = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
<Style id="cell"><LineStyle><color>ff00ffff</color><width>2</width></LineStyle>
  <PolyStyle><fill>0</fill></PolyStyle></Style>
<Style id="pos"><IconStyle><color>ff00ff00</color><scale>0.5</scale></IconStyle></Style>
"""
_TAIL = "</Document></kml>\n"

# level → line colour (aabbggrr): 1000 red → 300 blue-ish, spread across the pyramid
_LVLCOL = ["ff0000ff", "ff0080ff", "ff00ffff", "ff00ff80", "ff00ff00", "ffffff00",
           "ffff8000", "ffff00ff"]


def _square(lat, lon, size_m):
    dlat = (size_m / 2.0) / 110540.0
    dlon = (size_m / 2.0) / (math.cos(math.radians(lat)) * 111320.0)
    pts = [(lon - dlon, lat + dlat), (lon + dlon, lat + dlat), (lon + dlon, lat - dlat),
           (lon - dlon, lat - dlat), (lon - dlon, lat + dlat)]
    return " ".join(f"{x:.7f},{y:.7f},0" for x, y in pts)


def _poly(coords, style):
    return (f"<Placemark><styleUrl>#{style}</styleUrl><Polygon><outerBoundaryIs><LinearRing>"
            f"<coordinates>{coords}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>")


def _lvl_poly(lat, lon, size_m, color):
    return (f'<Placemark><Style><LineStyle><color>{color}</color><width>1</width></LineStyle>'
            f'<PolyStyle><fill>0</fill></PolyStyle></Style><Polygon><outerBoundaryIs><LinearRing>'
            f'<coordinates>{_square(lat, lon, size_m)}</coordinates></LinearRing></outerBoundaryIs>'
            f'</Polygon></Placemark>')


def _pt(lat, lon):
    return f'<Placemark><styleUrl>#pos</styleUrl><Point><coordinates>{lon:.7f},{lat:.7f},0</coordinates></Point></Placemark>'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checker-index", required=True)
    ap.add_argument("--city", required=True)
    ap.add_argument("--step-m", type=float, default=250.0)
    ap.add_argument("--levels-m", type=float, nargs="+", default=[1000, 900, 800, 700, 600, 500, 400, 300])
    ap.add_argument("--tile-size-m", type=float, default=1000.0)
    ap.add_argument("--sample-cells", type=int, default=2, help="cells to draw full sliding+pyramid for")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import h5py
    import numpy as np
    with h5py.File(Path(args.checker_index).expanduser(), "r") as f:
        cc = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f["city"][:]])
        lat = np.asarray(f["lat"][:], float)[cc == args.city]
        lon = np.asarray(f["lon"][:], float)[cc == args.city]
    if lat.size == 0:
        raise SystemExit(f"no cells for {args.city!r} in {args.checker_index}")

    body = [_HEAD, f"<Folder><name>checkerboard {args.city} ({lat.size} cells, {args.tile_size_m:.0f} m)</name>"]
    for la, lo in zip(lat, lon):
        body.append(_poly(_square(la, lo, args.tile_size_m), "cell"))
    body.append("</Folder>")

    radius_m = args.tile_size_m / 2.0
    n = min(args.sample_cells, lat.size)
    body.append(f"<Folder><name>sliding + pyramid (first {n} cells, step {args.step_m:.0f} m)</name>")
    for ci in range(n):
        la, lo = float(lat[ci]), float(lon[ci])
        cx, cy = latlon_to_merc(la, lo)
        body.append(f"<Folder><name>cell {ci}</name>")
        for (px, py) in cell_window_centres(cx, cy, la, radius_m, args.step_m):
            plat, plon = merc_to_latlon(px, py)
            body.append(_pt(plat, plon))
            for k, L in enumerate(args.levels_m):
                body.append(_lvl_poly(plat, plon, float(L), _LVLCOL[k % len(_LVLCOL)]))
        body.append("</Folder>")
    body.append("</Folder>")
    body.append(_TAIL)

    out = Path(args.out).expanduser(); out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", "".join(body))
    print(f"[ok] geometry kmz -> {out}  ({lat.size} cells, {n} with pyramids)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
