"""KMZ visualization of a split — points (and optional positive-tile footprints)
coloured by split (train / val / test / excluded).

Pure stdlib (xml text + zipfile); no shapely/pyproj. Consumes the same public models
as the rest of the split layer: ``GeoPoint`` (lat/lon) and ``GalleryIndex`` /
``GalleryTile`` (centre in EPSG:3857 grid + size) — tile corners are inverse-Mercator
projected to lon/lat for the polygon rings. Nothing here touches the split algorithm.
"""
from __future__ import annotations

import math
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from .positive_selection import GalleryIndex
from .schemas import TRAIN, VAL, TEST, EXCLUDED

_R = 6378137.0  # EPSG:3857 spherical radius (matches the tile grid)

# KML colours are aabbggrr (alpha, blue, green, red).
_COLORS = {
    TRAIN:    "ff37b34a",   # green
    VAL:      "ffe08214",   # blue-ish
    TEST:     "ff2222dd",   # red
    EXCLUDED: "ff888888",   # grey
}
_ORDER = (TRAIN, VAL, TEST, EXCLUDED)


def _merc_inv(x: float, y: float) -> tuple:
    """EPSG:3857 (x, y) -> (lon, lat) degrees."""
    lon = math.degrees(x / _R)
    lat = math.degrees(2.0 * math.atan(math.exp(y / _R)) - math.pi / 2.0)
    return lon, lat


def _tile_split_map(point_split: dict, point_to_tiles: dict) -> dict:
    """tile_id -> the (single) kept split whose points use it as a positive."""
    out = {}
    for pid, tiles in point_to_tiles.items():
        sp = point_split.get(pid)
        if sp in (TRAIN, VAL, TEST):
            for t in tiles:
                out[t] = sp                 # disjoint after a valid split
    return out


def build_kml(points, gallery: GalleryIndex, point_split: dict, *,
              point_to_tiles: dict = None, draw_tiles: bool = False,
              tile_size_fallback: float = 1000.0, name: str = "geo_split") -> str:
    styles = "".join(
        f'<Style id="pt_{sp}"><IconStyle><color>{c}</color><scale>0.7</scale>'
        f'<Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href>'
        f'</Icon></IconStyle></Style>'
        f'<Style id="tile_{sp}"><LineStyle><color>{c}</color><width>1</width></LineStyle>'
        f'<PolyStyle><color>{_fill(c)}</color></PolyStyle></Style>'
        for sp, c in _COLORS.items())

    by_split = {sp: [] for sp in _ORDER}
    for p in points:
        sp = point_split.get(p.point_id, EXCLUDED)
        if sp not in by_split:
            sp = EXCLUDED
        by_split[sp].append(p)

    folders = []
    for sp in _ORDER:
        placemarks = "".join(
            f'<Placemark><name>{escape(p.point_id)}</name>'
            f'<styleUrl>#pt_{sp}</styleUrl>'
            f'<Point><coordinates>{p.lon:.7f},{p.lat:.7f},0</coordinates></Point></Placemark>'
            for p in by_split[sp])
        folders.append(f'<Folder><name>points: {sp} ({len(by_split[sp])})</name>{placemarks}</Folder>')

    if draw_tiles and point_to_tiles is not None:
        tmap = _tile_split_map(point_split, point_to_tiles)
        tiles_by_split = {sp: [] for sp in (TRAIN, VAL, TEST)}
        for tid, sp in tmap.items():
            tiles_by_split[sp].append(tid)
        for sp in (TRAIN, VAL, TEST):
            polys = "".join(_tile_polygon(gallery.get(tid), sp, tile_size_fallback)
                            for tid in sorted(tiles_by_split[sp]))
            folders.append(f'<Folder><name>tiles: {sp} ({len(tiles_by_split[sp])})</name>{polys}</Folder>')

    return ('<?xml version="1.0" encoding="UTF-8"?>'
            '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
            f'<name>{escape(name)}</name>{styles}{"".join(folders)}'
            '</Document></kml>')


def _fill(line_color: str) -> str:
    """Semi-transparent fill from a line colour (drop alpha to ~25%)."""
    return "40" + line_color[2:]


def _tile_polygon(tile, sp: str, fallback: float) -> str:
    s = tile.size_m if (tile.size_m == tile.size_m and tile.size_m > 0) else fallback
    h = s / 2.0
    corners = [(tile.center_x - h, tile.center_y - h), (tile.center_x + h, tile.center_y - h),
               (tile.center_x + h, tile.center_y + h), (tile.center_x - h, tile.center_y + h),
               (tile.center_x - h, tile.center_y - h)]
    ring = " ".join(f"{lon:.7f},{lat:.7f},0" for lon, lat in (_merc_inv(x, y) for x, y in corners))
    return (f'<Placemark><name>{escape(tile.tile_id)}</name><styleUrl>#tile_{sp}</styleUrl>'
            f'<Polygon><outerBoundaryIs><LinearRing><coordinates>{ring}'
            f'</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>')


def write_kmz(path, points, gallery: GalleryIndex, point_split: dict, *,
              point_to_tiles: dict = None, draw_tiles: bool = False,
              tile_size_fallback: float = 1000.0, name: str = "geo_split") -> Path:
    kml = build_kml(points, gallery, point_split, point_to_tiles=point_to_tiles,
                    draw_tiles=draw_tiles, tile_size_fallback=tile_size_fallback, name=name)
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", kml)
    return out
