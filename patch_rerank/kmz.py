"""Minimal KMZ (zipped KML) writer for the fine-localization results — view in Google Earth.

Per query: a green GT placemark, a red predicted placemark, and a line between them (labelled with
the error in metres). Pure stdlib (zipfile) — no project deps."""
from __future__ import annotations

import zipfile
from pathlib import Path

_HEAD = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
<Style id="gt"><IconStyle><color>ff00ff00</color><scale>1.0</scale></IconStyle></Style>
<Style id="pred"><IconStyle><color>ff0000ff</color><scale>1.0</scale></IconStyle></Style>
<Style id="predk"><IconStyle><color>ff00a5ff</color><scale>0.8</scale></IconStyle></Style>
<Style id="link"><LineStyle><color>ffffffff</color><width>2</width></LineStyle></Style>
"""
_TAIL = "</Document></kml>\n"


def _pt(name, lat, lon, style):
    return (f"<Placemark><name>{name}</name><styleUrl>#{style}</styleUrl>"
            f"<Point><coordinates>{lon:.7f},{lat:.7f}</coordinates></Point></Placemark>")


def _line(lat1, lon1, lat2, lon2, name):
    return (f"<Placemark><name>{name}</name><styleUrl>#link</styleUrl><LineString>"
            f"<coordinates>{lon1:.7f},{lat1:.7f} {lon2:.7f},{lat2:.7f}</coordinates>"
            f"</LineString></Placemark>")


def build_kml(entries) -> str:
    """entries: list of {name, gt:(lat,lon), pred:(lat,lon), dist_m, ranked?}.

    ``ranked`` (optional) = list of {rank, lat, lon, dist_m, ...}: rank 1 is drawn as the red ``pred``
    with a GT link; ranks 2..N as smaller orange pins (the reranker's top-N shortlist)."""
    body = [_HEAD]
    for e in entries:
        glat, glon = e["gt"]
        plat, plon = e["pred"]
        d = e.get("dist_m", float("nan"))
        body.append(f"<Folder><name>{e['name']} ({d:.0f} m)</name>")
        body.append(_pt("GT", glat, glon, "gt"))
        body.append(_pt(f"pred {d:.0f}m", plat, plon, "pred"))
        body.append(_line(glat, glon, plat, plon, f"{d:.0f} m"))
        for t in e.get("ranked", []):
            if t.get("rank", 1) == 1:
                continue                              # rank 1 already drawn as red pred
            body.append(_pt(f"#{t['rank']} {t.get('dist_m', float('nan')):.0f}m",
                            t["lat"], t["lon"], "predk"))
        body.append("</Folder>")
    body.append(_TAIL)
    return "".join(body)


def write_kmz(path, entries) -> None:
    kml = build_kml(entries)
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", kml)
