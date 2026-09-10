"""Local fixtures for the positive_selection subsystem tests."""
from __future__ import annotations

import math

from geo_split_no_overlap.positive_selection.models import (GalleryIndex, GalleryTile,
                                                            GeoPoint)

_R = 6378137.0


def xy_to_latlon(x, y):
    return (math.degrees(2.0 * math.atan(math.exp(y / _R)) - math.pi / 2.0),
            math.degrees(x / _R))


def latlon_to_xy(lat, lon):
    return (_R * math.radians(lon),
            _R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))


def tile(tid, x, y, size=1000.0, city="c"):
    lat, lon = xy_to_latlon(x, y)
    return GalleryTile(tile_id=str(tid), city=city, center_x=x, center_y=y,
                       lat=lat, lon=lon, size_m=size)


def gallery(tiles, crs="EPSG:3857", fallback=1000.0):
    return GalleryIndex(list(tiles), crs=crs, area_units="true_m2 (test)",
                        tile_size_fallback=fallback)


def point(pid, x, y, city="c"):
    lat, lon = xy_to_latlon(x, y)
    return GeoPoint(point_id=pid, city=city, lat=lat, lon=lon, x=x, y=y)
