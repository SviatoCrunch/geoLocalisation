"""Data loaders for training: map tile grids + drone query tokens from map_extract H5s.

Tile grids: raw galleries are group-per-tile ``<key>/ift_dino`` (1,D,H,W) → returned as (H,W,D).
Query tokens: query H5 (map_extract ``--images``) is ``img<idx>/ift_dino`` (D,N) with attr
``filename`` → returned as (N,D), keyed by the split point id ``"<city>:<stem>"`` so it matches
``geo_split`` / ``build_split_relevance`` query ids by construction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class TileGridLoader:
    """tile_id ``"<city>:<groupkey>"`` → token grid tensor (H,W,D), cached in RAM."""

    def __init__(self, gallery_by_city: dict, grid_size: int | None = None, cache: bool = True):
        self._paths = {c: Path(p).expanduser() for c, p in gallery_by_city.items()}
        self.grid_size = int(grid_size) if grid_size else None
        self._files: dict = {}
        self._cache: dict | None = {} if cache else None

    def _file(self, city):
        f = self._files.get(city)
        if f is None:
            import h5py
            f = h5py.File(self._paths[city], "r")
            self._files[city] = f
        return f

    def grid(self, tile_id: str) -> torch.Tensor:
        if self._cache is not None and tile_id in self._cache:
            return self._cache[tile_id]
        city, key = tile_id.split(":", 1)
        arr = np.asarray(self._file(city)[key]["ift_dino"])
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]                                       # (D,H,W)
        D, H, W = arr.shape
        g = torch.from_numpy(arr.reshape(D, H * W).T.reshape(H, W, D).astype(np.float32))
        if self.grid_size and (H != self.grid_size or W != self.grid_size):
            g = F.interpolate(g.permute(2, 0, 1).unsqueeze(0), size=(self.grid_size, self.grid_size),
                              mode="bilinear", align_corners=False)[0].permute(1, 2, 0).contiguous()
        if self._cache is not None:
            self._cache[tile_id] = g
        return g

    def stack(self, tile_ids) -> torch.Tensor:
        return torch.stack([self.grid(t) for t in tile_ids])

    def close(self):
        for f in self._files.values():
            f.close()
        self._files.clear()


class QueryTokenStore:
    """split point id ``"<city>:<stem>"`` → drone-frame tokens (N,D)."""

    def __init__(self, query_by_city: dict):
        import h5py
        self._paths = {c: Path(p).expanduser() for c, p in query_by_city.items()}
        self._files: dict = {}
        self._index: dict = {}                                # point_id -> (city, groupkey)
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

    def ids(self):
        return set(self._index)

    def tokens(self, point_id: str) -> torch.Tensor:
        city, key = self._index[point_id]
        arr = np.asarray(self._file(city)[key]["ift_dino"])   # (D, N)
        return torch.from_numpy(arr.T.astype(np.float32))     # (N, D)

    def close(self):
        for f in self._files.values():
            f.close()
        self._files.clear()
