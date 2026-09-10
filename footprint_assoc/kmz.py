"""KMZ visualization of footprint estimates — the poor-man's oracle (no GT footprint).

Per frame it draws: the concentric pyramid squares (fill opacity ∝ fused level
probability, so the chosen scale stands out), the frame centre coloured by status
(accept/soft/refuse), and — if gallery geometry + associations are given — the tiles
coloured by pair status. Open in Google Earth to eyeball whether the estimator picks a
sensible footprint and where methods disagree. Pure stdlib (xml + zipfile).
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from .geometry import merc, square_corners_lonlat
from .schemas import ACCEPT, SOFT, REFUSE, STRONG, PARTIAL, UNDETERMINED, NEGATIVE

# KML colours aabbggrr
_STATUS = {ACCEPT: "ff37b34a", SOFT: "ff0aa5ff", REFUSE: "ff2222dd"}
_PAIR = {STRONG: "ff37b34a", PARTIAL: "ff00d7ff", UNDETERMINED: "ff888888", NEGATIVE: None}


def _poly(ring, line_color, fill_alpha_hex, name):
    coords = " ".join(f"{lon:.7f},{lat:.7f},0" for lon, lat in ring)
    fill = fill_alpha_hex + line_color[2:]
    return (f'<Placemark><name>{escape(name)}</name>'
            f'<Style><LineStyle><color>{line_color}</color><width>1</width></LineStyle>'
            f'<PolyStyle><color>{fill}</color></PolyStyle></Style>'
            f'<Polygon><outerBoundaryIs><LinearRing><coordinates>{coords}'
            f'</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>')


def build_kml(estimates, tiles_geom: dict = None, assoc: dict = None, name: str = "footprint") -> str:
    folders = []
    for est in estimates:
        cx, cy = merc(est.center_lat, est.center_lon)
        parts = []
        # concentric level squares, opacity ∝ fused probability
        pmax = max(est.soft.values()) if est.soft else 0.0
        for s in est.scales:
            p = est.soft.get(float(s), 0.0)
            is_best = (est.best_scale is not None and abs(s - est.best_scale) < 1e-6)
            line = "ffffffff" if not is_best else _STATUS.get(est.status, "ffffffff")
            alpha = int(30 + 150 * (p / pmax if pmax > 0 else 0))
            parts.append(_poly(square_corners_lonlat(cx, cy, s),
                               line, f"{alpha:02x}", f"{s:.0f}m p={p:.2f}"))
        # centre point coloured by status
        parts.append(
            f'<Placemark><name>{escape(est.frame_id)} [{est.status}] '
            f'best={est.best_scale} conf={est.confidence:.2f}</name>'
            f'<Style><IconStyle><color>{_STATUS.get(est.status, "ffffffff")}</color>'
            f'<scale>0.8</scale></IconStyle></Style>'
            f'<Point><coordinates>{est.center_lon:.7f},{est.center_lat:.7f},0</coordinates></Point>'
            f'</Placemark>')
        # gallery tiles coloured by pair status (optional)
        if tiles_geom and assoc and est.frame_id in assoc:
            for pa in assoc[est.frame_id]:
                col = _PAIR.get(pa.status)
                if col is None or pa.tile_id not in tiles_geom:
                    continue
                tlat, tlon, tsize = tiles_geom[pa.tile_id]
                tx, ty = merc(tlat, tlon)
                parts.append(_poly(square_corners_lonlat(tx, ty, tsize), col, "30",
                                   f"{pa.tile_id} {pa.status} iou={pa.iou:.2f}"))
        folders.append(f'<Folder><name>{escape(est.frame_id)} [{est.status}]</name>'
                       f'{"".join(parts)}</Folder>')
    return ('<?xml version="1.0" encoding="UTF-8"?>'
            '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
            f'<name>{escape(name)}</name>{"".join(folders)}</Document></kml>')


def write_kmz(path, estimates, tiles_geom: dict = None, assoc: dict = None,
              name: str = "footprint") -> Path:
    kml = build_kml(estimates, tiles_geom=tiles_geom, assoc=assoc, name=name)
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", kml)
    return out
