"""Orchestration: pyramid features → per-method level scores → fusion → (association).

Estimator/fusion choices come from the config + Registries (no if-name branching here).
"""
from __future__ import annotations

from .association import associate
from .estimators import create_estimator
from .fusion import create_fusion
from .schemas import FrameEstimate


def estimate_frame(pyramid, cfg) -> FrameEstimate:
    cfg.validate()
    estimators = [create_estimator(n) for n in cfg.estimators]
    scales = pyramid.scales()
    per_method = {est.name: [est.score_level(pyramid.query, lv, cfg) for lv in pyramid.levels]
                  for est in estimators}
    fusion = create_fusion(cfg.fusion)
    return fusion.fuse(pyramid.frame_id, pyramid.center_lat, pyramid.center_lon,
                       scales, per_method, cfg)


def estimate_and_associate(pyramid, tiles, cfg):
    """Return (FrameEstimate, list[PairAssoc]). ``tiles`` = (tile_id, lat, lon, size_m) iterable."""
    est = estimate_frame(pyramid, cfg)
    return est, associate(est, tiles, cfg)
