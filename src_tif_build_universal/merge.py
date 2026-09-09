"""Window-by-window multi-layer TIF merge.

No dependencies on other library modules or src_tif_build.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from rasterio.windows import Window, bounds as window_bounds
from tqdm import tqdm


def merge_layers(layer_paths: List[Path], out_path: Path) -> None:
    """Merge TIF layers into a mosaic without loading the full raster into RAM.

    Layers are sorted by pixel resolution (smallest first = highest
    resolution).  The highest-resolution layer fills each pixel first;
    lower-resolution layers only fill pixels that are still empty.

    Args:
        layer_paths: GeoTIF paths to merge.
        out_path: Output mosaic path.
    """
    sources = [rasterio.open(p) for p in layer_paths]
    try:
        sources.sort(key=lambda src: src.res[0])

        base = sources[0]
        dst_crs = base.crs
        dst_res_x, dst_res_y = base.res
        dtype = base.dtypes[0]
        count = base.count
        nodata = base.nodata if base.nodata is not None else 0

        left = min(src.bounds.left for src in sources)
        bottom = min(src.bounds.bottom for src in sources)
        right = max(src.bounds.right for src in sources)
        top = max(src.bounds.top for src in sources)

        width = int(np.ceil((right - left) / dst_res_x))
        height = int(np.ceil((top - bottom) / dst_res_y))
        transform = from_origin(left, top, dst_res_x, dst_res_y)

        profile = base.profile.copy()
        profile.update(
            driver="GTiff",
            width=width,
            height=height,
            transform=transform,
            crs=dst_crs,
            dtype=dtype,
            count=count,
            nodata=nodata,
            BIGTIFF="YES",
            SPARSE_OK="TRUE",
            compress="deflate",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(out_path, "w", **profile) as dst:
            for _, win in tqdm(
                list(dst.block_windows(1)),
                desc="Merging layers",
                unit="block",
            ):
                dst_arr = np.full(
                    (count, int(win.height), int(win.width)),
                    nodata,
                    dtype=dtype,
                )
                dst_win_transform = dst.window_transform(win)
                dst_win_bounds = window_bounds(win, transform)

                for src in sources:
                    if (
                        src.bounds.right <= dst_win_bounds[0]
                        or src.bounds.left >= dst_win_bounds[2]
                        or src.bounds.top <= dst_win_bounds[1]
                        or src.bounds.bottom >= dst_win_bounds[3]
                    ):
                        continue

                    temp = np.full_like(dst_arr, nodata)
                    reproject(
                        source=rasterio.band(src, list(range(1, count + 1))),
                        destination=temp,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        src_nodata=src.nodata if src.nodata is not None else nodata,
                        dst_transform=dst_win_transform,
                        dst_crs=dst_crs,
                        dst_nodata=nodata,
                        resampling=Resampling.nearest,
                    )

                    dst_empty = np.all(dst_arr == nodata, axis=0)
                    temp_valid = ~np.all(temp == nodata, axis=0)
                    dst_arr[:, dst_empty & temp_valid] = temp[:, dst_empty & temp_valid]

                dst.write(dst_arr, window=win)
    finally:
        for src in sources:
            src.close()
