"""CogCropSource — crop concentric RGB squares from a satellite COG (SERVER-ONLY, rasterio).

Implements :class:`CropSource`. Crops the EXACT EPSG:3857 square ``[center3857 ± size/2]``
(so the crop matches footprint_assoc's 3857 footprint geometry) regardless of the COG's
own CRS, resampled to a fixed ``out_px`` RGB image. Lazy rasterio import — importing the
package pulls nothing heavy.
"""
from __future__ import annotations

import numpy as np

from ..geometry import merc


class CogCropSource:
    def __init__(self, cog_path, out_px: int = 224, bands=(1, 2, 3)):
        self.path = str(cog_path)
        self.out_px = int(out_px)
        self.bands = tuple(bands)
        import rasterio
        with rasterio.open(self.path) as ds:
            self.crs = str(ds.crs)
        self._is_3857 = self.crs.upper() in ("EPSG:3857", "EPSG:900913")

    def crop(self, center_lat: float, center_lon: float, size_m: float) -> np.ndarray:
        import rasterio
        from rasterio.windows import from_bounds
        from rasterio.enums import Resampling
        x, y = merc(center_lat, center_lon)            # 3857 centre
        h = float(size_m) / 2.0
        with rasterio.open(self.path) as ds:
            if self._is_3857:
                minx, miny, maxx, maxy = x - h, y - h, x + h, y + h
            else:
                from rasterio.warp import transform as warp_transform
                xs, ys = warp_transform("EPSG:3857", ds.crs,
                                        [x - h, x + h, x - h, x + h],
                                        [y - h, y - h, y + h, y + h])
                minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
            win = from_bounds(minx, miny, maxx, maxy, ds.transform)
            arr = ds.read(indexes=list(self.bands), window=win,
                          out_shape=(len(self.bands), self.out_px, self.out_px),
                          boundless=True, fill_value=0, resampling=Resampling.bilinear)
        return np.transpose(arr, (1, 2, 0)).astype(np.uint8)   # (out_px, out_px, 3)
