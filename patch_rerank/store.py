"""Reader for a precomputed per-cell fine-rerank token-grid store (one H5 per city).

Layout (written by precompute_store.py):
  attrs: city, step_m, levels_m, output_px, backbone, projector, tile_size_m
  datasets: px, py, lat, lon          (n_positions,)  — snapped merc centres + geo
  group  p<i>/l<int(L)>               token grid (h,w,D) fp16  for position i, level L

Positions are snapped to the SAME global grid the reranker uses (snap(v,step)), so a query's
within-cell centres map to store keys exactly. DINO is done ONCE offline → query-time = load + match.
"""
from __future__ import annotations

import numpy as np
import torch


class RerankStore:
    def __init__(self, path: str):
        import h5py
        self.path = path
        self.f = h5py.File(path, "r")
        self.px = np.asarray(self.f["px"][:], float)
        self.py = np.asarray(self.f["py"][:], float)
        self.step_m = float(self.f.attrs["step_m"])
        self.levels_m = [float(x) for x in self.f.attrs["levels_m"]]
        self.output_px = int(self.f.attrs["output_px"])
        self.tile_size_m = float(self.f.attrs.get("tile_size_m", 1000.0))
        self._idx = {(round(float(x), 2), round(float(y), 2)): i
                     for i, (x, y) in enumerate(zip(self.px, self.py))}

    def _k(self, px, py):
        return (round(float(px), 2), round(float(py), 2))

    def has(self, px, py) -> bool:
        return self._k(px, py) in self._idx

    def open_handle(self):
        """A fresh read-only h5py handle over the SAME file — one per thread (h5py is not thread-safe
        on a single handle). The shared ``_idx`` (built once) is read-only, so threads reuse it."""
        import h5py
        return h5py.File(self.path, "r")

    def grid_from(self, f, px, py, L) -> torch.Tensor:
        """Token grid (h,w,D) float32 via a caller-supplied handle ``f`` (enables parallel reads)."""
        i = self._idx[self._k(px, py)]
        arr = np.asarray(f[f"p{i}/l{int(L)}"])
        return torch.from_numpy(arr.astype(np.float32))

    def grid(self, px, py, L) -> torch.Tensor:
        """Token grid (h,w,D) float32 for the snapped position + level (raises if absent)."""
        return self.grid_from(self.f, px, py, L)

    def close(self):
        self.f.close()
