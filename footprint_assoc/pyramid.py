"""Build the concentric-pyramid features for a frame (multi-scale features don't exist
yet, so we build them here from a satellite crop source + a frozen extractor).

Both the extractor and the crop source are injected (protocols) — so this is testable
with fakes and carries no heavy dependency itself.
"""
from __future__ import annotations

from typing import Sequence

from .schemas import PyramidFeatures, QueryFeatures, LevelFeatures


def build(frame_id: str, center_lat: float, center_lon: float, uav_image,
          scales: Sequence[float], crop_source, extractor) -> PyramidFeatures:
    """UAV image + concentric satellite crops at ``scales`` → :class:`PyramidFeatures`."""
    qg, qp = extractor.extract(uav_image)
    query = QueryFeatures(global_vec=qg, patch_grid=qp)
    levels = []
    for s in sorted(float(x) for x in scales):
        img = crop_source.crop(center_lat, center_lon, s)
        lg, lp = extractor.extract(img)
        levels.append(LevelFeatures(scale_m=float(s), global_vec=lg, patch_grid=lp))
    return PyramidFeatures(frame_id=frame_id, center_lat=center_lat, center_lon=center_lon,
                           query=query, levels=tuple(levels))
