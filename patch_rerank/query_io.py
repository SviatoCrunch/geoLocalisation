"""Query token GRIDS with 2D patch positions (needed by patch-RANSAC).

The query H5 (map_extract ``--images``) stores ``img<idx>/ift_dino`` (D,N) plus ``patch_grid_h`` /
``patch_grid_w`` attrs and, when sky-filtered, a ``keep_indices`` dataset into the full h×w grid —
so the 2D coordinate of every kept token is recoverable (unlike the flat QueryTokenStore used for
training). Keyed by the split point id ``"<city>:<stem>"`` to match the shortlist json.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .matcher import grid_keypoints


class QueryGridStore:
    def __init__(self, query_by_city: dict):
        import h5py
        self._paths = {c: Path(p).expanduser() for c, p in query_by_city.items()}
        self._files: dict = {}
        self._index: dict = {}
        for c, p in self._paths.items():
            with h5py.File(p, "r") as f:
                for k in f.keys():
                    if not (hasattr(f[k], "keys") and "ift_dino" in f[k]):
                        continue
                    fn = f[k].attrs.get("filename", "")
                    stem = Path(str(fn)).stem if fn else k
                    self._index[f"{c}:{stem}"] = (c, k)

    def _file(self, city):
        f = self._files.get(city)
        if f is None:
            import h5py
            f = h5py.File(self._paths[city], "r")
            self._files[city] = f
        return f

    def has(self, point_id: str) -> bool:
        return point_id in self._index

    def get(self, point_id: str):
        """→ (feat (N,D) torch, xy (N,2) np, lat, lon). xy in patch-grid units, sky tokens dropped."""
        city, key = self._index[point_id]
        g = self._file(city)[key]
        feat = torch.from_numpy(np.asarray(g["ift_dino"]).T.astype(np.float32))     # (N, D)
        h = int(g.attrs["patch_grid_h"]); w = int(g.attrs["patch_grid_w"])
        xy = grid_keypoints(h, w)                                                    # (h*w, 2)
        if "keep_indices" in g:
            xy = xy[np.asarray(g["keep_indices"]).reshape(-1)]
        if xy.shape[0] != feat.shape[0]:                                            # grid/token mismatch
            raise ValueError(f"{point_id}: {xy.shape[0]} positions vs {feat.shape[0]} tokens")
        lat = float(g.attrs.get("lat", float("nan")))
        lon = float(g.attrs.get("lon", float("nan")))
        return feat, xy, lat, lon

    def close(self):
        for f in self._files.values():
            f.close()
        self._files.clear()
