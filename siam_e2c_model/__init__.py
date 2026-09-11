"""siam_e2c_model — the e2c Stage-2 query-conditioned model, model-only.

Switchable per-cell aggregation (``supervlad`` soft / ``vlad``==``residual`` classic) via a
Registry, configurable vocabulary size ``k``, and a switchable output pyramid via ``pyramid_mode``
(run separately for A/B, never merged):

  * ``cell`` (default) — n×n spatial cell splits ``scales_cells`` (e2c ``(8,4,2,1)``);
  * ``concentric``     — nested central crops of physical size ``concentric_sizes_m`` (apex 250 m),
                         one fractional-masked region per level.

Both modes share the same query encoder + cross-scale scorer (``score = Σ_ℓ β_ℓ(q)·S_ℓ``, a soft
mixture of scales). Self-contained: the Stage-2 core is VENDORED in ``siam_e2c_model.vendored``
(byte-faithful copy of siam_model_stage3's stage2_core, never edited); the concentric core subclasses
it; the residual arm is ported from the e2c script — no external RevisitAnything dependency.

Contains NO loss / batch (see ``geo_train_batching``), no split, no data IO — only the model
contract: ``build_V`` / ``encode_query`` / ``score`` / ``score_with_details`` / ``scales_cells`` /
params. ``score_with_details`` level signals are PSEUDO-footprints (no scale supervision), never
metric-verified footprints.
"""
from __future__ import annotations

from .aggregation import (AggregationStrategy, available_aggregations, create_aggregation,
                          register_aggregation)
from .config import E2cModelConfig
from .model import E2cModel, build_e2c_model

__all__ = [
    "E2cModel", "E2cModelConfig", "build_e2c_model",
    "AggregationStrategy", "available_aggregations", "create_aggregation",
    "register_aggregation",
]

__version__ = "0.1.0"
