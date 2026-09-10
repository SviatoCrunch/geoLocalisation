"""Sky-aware query construction — the bridge ``sky_filter`` → ``footprint_assoc``.

Builds ``QueryFeatures`` from a UAV image + a pixel keep-mask by DROPPING sky tokens
(``reduce_to_grid`` + ``apply_drop``), then recomputing the global descriptor from the
kept ground tokens only. Integration helper — NOT imported by ``footprint_assoc.__init__``
(the isolated core stays free of the sky_filter dependency); used by ``run.py``/tests.
"""
from __future__ import annotations

import numpy as np

from .schemas import QueryFeatures


def sky_filtered_query(extractor, uav_image, keep_px, cell_sky_max: float = 0.5) -> QueryFeatures:
    g, grid = extractor.extract(uav_image)                 # (D,), (h, w, D)
    if keep_px is None:
        return QueryFeatures(global_vec=g, patch_grid=grid)

    from sky_filter import reduce_to_grid, apply_drop
    gh, gw = grid.shape[0], grid.shape[1]
    token_keep = reduce_to_grid(np.asarray(keep_px, bool), gh, gw, cell_sky_max)
    kept = apply_drop(grid, token_keep)                    # (N_kept, D)
    if kept.shape[0] == 0:                                  # all sky -> fall back to full grid
        return QueryFeatures(global_vec=g, patch_grid=grid)

    gv = kept.mean(axis=0)
    gv = gv / max(float(np.linalg.norm(gv)), 1e-12)
    return QueryFeatures(global_vec=gv.astype(grid.dtype, copy=False),
                         patch_grid=kept[:, None, :])       # (N_kept, 1, D)
