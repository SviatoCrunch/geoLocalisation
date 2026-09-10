"""CogCropSource test — builds a tiny synthetic EPSG:3857 GeoTIFF (skips if no rasterio)."""
import numpy as np
import pytest

from footprint_assoc.geometry import merc


def _make_tif(path, center_lat, center_lon, half_m=2000.0, px=256):
    rio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds
    cx, cy = merc(center_lat, center_lon)
    minx, miny, maxx, maxy = cx - half_m, cy - half_m, cx + half_m, cy + half_m
    data = np.zeros((3, px, px), np.uint8)
    data[0, px // 4:3 * px // 4, px // 4:3 * px // 4] = 200      # a red square in the middle
    tr = from_bounds(minx, miny, maxx, maxy, px, px)
    with rio.open(path, "w", driver="GTiff", height=px, width=px, count=3,
                  dtype="uint8", crs="EPSG:3857", transform=tr) as ds:
        ds.write(data)


def test_crop_shape_and_center(tmp_path):
    pytest.importorskip("rasterio")
    from footprint_assoc.extractors.cog_crop import CogCropSource
    lat, lon = 48.5, 37.8
    tif = tmp_path / "syn.tif"
    _make_tif(tif, lat, lon)
    src = CogCropSource(str(tif), out_px=32)
    img = src.crop(lat, lon, 1000.0)
    assert img.shape == (32, 32, 3) and img.dtype == np.uint8
    # centre of a 1000 m crop lands inside the painted red square
    assert img[16, 16, 0] > 100
