"""KMZ views of the two-stage search (torch-free: h5py + math + zipfile).

Modes:
  --shortlist S.json            → COARSE kmz: per query the top-K cell squares + GT (all val+test).
  --rerank-json R.json          → THREE kmz from a map_rerank result:
       *_coarse.kmz  top-K cells + GT
       *_rerank.kmz  each cell's best position (dot, size≈score) + the WINNER's winning-level pyramid
                     box + label "L=<level>m"; GT
       *_final.kmz   predicted (red) + GT (green) + error line

GT is taken from the query id suffix ``..._<lat>_<lon>`` (or rec["gt"] for rerank-json).
"""
from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

_STYLES = """
<Style id="cell"><LineStyle><color>ff00ffff</color><width>2</width></LineStyle><PolyStyle><fill>0</fill></PolyStyle></Style>
<Style id="win"><LineStyle><color>ff00ffff</color><width>1</width></LineStyle><PolyStyle><fill>0</fill></PolyStyle></Style>
<Style id="gt"><IconStyle><color>ff00ff00</color><scale>1.1</scale></IconStyle></Style>
<Style id="pred"><IconStyle><color>ff0000ff</color><scale>1.1</scale></IconStyle></Style>
<Style id="cand"><IconStyle><color>ffff8000</color><scale>0.4</scale></IconStyle></Style>
<Style id="link"><LineStyle><color>ffffffff</color><width>2</width></LineStyle></Style>
"""


def _isfloat(t):
    try:
        float(t); return True
    except ValueError:
        return False


def _gt_from_id(qid):
    toks = qid.split(":", 1)[-1].split("_")
    nums = [t for t in toks if _isfloat(t)]
    return (float(nums[-2]), float(nums[-1])) if len(nums) >= 2 else (float("nan"), float("nan"))


def _square(lat, lon, size_m):
    dlat = (size_m / 2.0) / 110540.0
    dlon = (size_m / 2.0) / (math.cos(math.radians(lat)) * 111320.0)
    pts = [(lon - dlon, lat + dlat), (lon + dlon, lat + dlat), (lon + dlon, lat - dlat),
           (lon - dlon, lat - dlat), (lon - dlon, lat + dlat)]
    return " ".join(f"{x:.7f},{y:.7f},0" for x, y in pts)


def _poly(coords, style):
    return (f"<Placemark><styleUrl>#{style}</styleUrl><Polygon><outerBoundaryIs><LinearRing>"
            f"<coordinates>{coords}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>")


def _pt(lat, lon, style, name=""):
    nm = f"<name>{name}</name>" if name else ""
    return f"<Placemark>{nm}<styleUrl>#{style}</styleUrl><Point><coordinates>{lon:.7f},{lat:.7f},0</coordinates></Point></Placemark>"


def _line(a, b, name=""):
    nm = f"<name>{name}</name>" if name else ""
    return (f"<Placemark>{nm}<styleUrl>#link</styleUrl><LineString><coordinates>"
            f"{a[1]:.7f},{a[0]:.7f},0 {b[1]:.7f},{b[0]:.7f},0</coordinates></LineString></Placemark>")


def _write(path, folders):
    kml = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>', _STYLES]
    kml += folders
    kml.append("</Document></kml>")
    p = Path(path).expanduser(); p.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", "".join(kml))
    print(f"[ok] -> {p}", flush=True)


def _cell_xy(dense_index):
    import h5py
    import numpy as np
    with h5py.File(Path(dense_index).expanduser(), "r") as f:
        ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f["tile_id"][:]]
        lat = np.asarray(f["lat"][:], float); lon = np.asarray(f["lon"][:], float)
    return {t: (float(lat[i]), float(lon[i])) for i, t in enumerate(ids)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shortlist", default=None)
    ap.add_argument("--rerank-json", default=None)
    ap.add_argument("--dense-index", required=True)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--tile-size-m", type=float, default=1000.0)
    ap.add_argument("--day-only", action="store_true", help="skip *_night queries (day map only)")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args(argv)

    cellxy = _cell_xy(args.dense_index)

    if args.shortlist:
        sj = json.loads(Path(args.shortlist).expanduser().read_text())["shortlist"]
        folders = []
        for q, entry in sj.items():
            if args.day_only and "_night" in q:
                continue
            glat, glon = _gt_from_id(q)
            body = [f"<Folder><name>{q}</name>", _pt(glat, glon, "gt", "GT")]
            for cid in entry["cells"][:args.k]:
                if cid in cellxy:
                    la, lo = cellxy[cid]
                    body.append(_poly(_square(la, lo, args.tile_size_m), "cell"))
            body.append("</Folder>")
            folders.append("".join(body))
        _write(f"{args.out_prefix}_coarse.kmz", folders)
        return 0

    if not args.rerank_json:
        raise SystemExit("give --shortlist or --rerank-json")
    rj = json.loads(Path(args.rerank_json).expanduser().read_text())["per_query"]
    coarse, rerank, final = [], [], []
    for q, rec in rj.items():
        if args.day_only and "_night" in q:
            continue
        gt = (rec["gt"]["lat"], rec["gt"]["lon"])
        cells = rec.get("cells", [])
        # winner = cell with max cell_score
        win = max(cells, key=lambda c: c.get("cell_score", -1)) if cells else None
        # (1) coarse: top-K cell squares
        cb = [f"<Folder><name>{q}</name>", _pt(*gt, "gt", "GT")]
        for c in cells:
            if c["cell_id"] in cellxy:
                la, lo = cellxy[c["cell_id"]]
                cb.append(_poly(_square(la, lo, args.tile_size_m), "cell"))
        cb.append("</Folder>"); coarse.append("".join(cb))
        # (2) rerank: each cell best dot + winner level box + label
        rb = [f"<Folder><name>{q}</name>", _pt(*gt, "gt", "GT")]
        for c in cells:
            b = c.get("best", {})
            if b.get("lat") is not None:
                rb.append(_pt(b["lat"], b["lon"], "cand", f"{c.get('cell_score',0):.3f}"))
        if win and win.get("best", {}).get("lat") is not None:
            b = win["best"]
            rb.append(_pt(b["lat"], b["lon"], "pred", f"WIN L={b.get('level_m')}m s={win.get('cell_score',0):.3f}"))
            rb.append(_poly(_square(b["lat"], b["lon"], float(b.get("level_m", args.tile_size_m))), "win"))
        rb.append("</Folder>"); rerank.append("".join(rb))
        # (3) final: pred vs GT
        if win and win.get("best", {}).get("lat") is not None:
            pred = (win["best"]["lat"], win["best"]["lon"])
            fb = [f"<Folder><name>{q} ({rec.get('fine_dist_m', float('nan')):.0f} m)</name>",
                  _pt(*gt, "gt", "GT"), _pt(*pred, "pred", f"pred {rec.get('fine_dist_m',0):.0f}m"),
                  _line(gt, pred, f"{rec.get('fine_dist_m',0):.0f} m"), "</Folder>"]
            final.append("".join(fb))
    _write(f"{args.out_prefix}_coarse.kmz", coarse)
    _write(f"{args.out_prefix}_rerank.kmz", rerank)
    _write(f"{args.out_prefix}_final.kmz", final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
