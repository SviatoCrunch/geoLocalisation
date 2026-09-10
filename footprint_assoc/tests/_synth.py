"""Synthetic fixtures for footprint_assoc (no backbone / no satellite needed)."""
from __future__ import annotations

import numpy as np

from footprint_assoc.schemas import (PyramidFeatures, QueryFeatures, LevelFeatures,
                                     LevelScore)

_SCALES = [100.0, 200.0, 400.0, 800.0, 1000.0]


def _unit(v):
    return v / max(np.linalg.norm(v), 1e-12)


def make_pyramid(true_idx=2, scales=None, D=16, g=6, center=(48.5, 37.8), seed=0):
    """Pyramid where level ``true_idx`` matches the UAV query (global + patch)."""
    scales = list(scales or _SCALES)
    rng = np.random.default_rng(seed)
    levels = []
    grids, globs = [], []
    for s in scales:
        gv = _unit(rng.standard_normal(D))
        pg = rng.standard_normal((g, g, D))
        globs.append(gv); grids.append(pg)
        levels.append(LevelFeatures(scale_m=float(s), global_vec=gv, patch_grid=pg))
    query = QueryFeatures(global_vec=globs[true_idx].copy(), patch_grid=grids[true_idx].copy())
    return PyramidFeatures(frame_id="f0", center_lat=center[0], center_lon=center[1],
                           query=query, levels=tuple(levels))


def per_method_peaked(scales, g_peak, p_peak, g_val=0.9, p_val=0.8, base=0.1):
    """Build a per_method dict with global/patch curves peaked at given indices."""
    gm = [LevelScore(s, g_val if i == g_peak else base) for i, s in enumerate(scales)]
    pm = [LevelScore(s, p_val if i == p_peak else base) for i, s in enumerate(scales)]
    return {"global_vlad": gm, "patch_overlap": pm}


class FakeExtractor:
    """Deterministic extractor: image (H,W,3) uint8 -> (global, patch grid) from its bytes."""

    def __init__(self, D=16, g=6):
        self.D, self.g = D, g

    def extract(self, image):
        h = int(np.asarray(image).astype(np.int64).sum())
        rng = np.random.default_rng(h % (2 ** 32))
        gv = rng.standard_normal(self.D)
        gv = gv / max(np.linalg.norm(gv), 1e-12)
        pg = rng.standard_normal((self.g, self.g, self.D))
        return gv, pg


class FakeCropSource:
    """Returns a deterministic image per (rounded) scale so the same scale -> same features."""

    def crop(self, center_lat, center_lon, size_m):
        val = int(round(size_m)) % 255
        return np.full((8, 8, 3), val, dtype=np.uint8)
