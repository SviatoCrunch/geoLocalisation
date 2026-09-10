"""siam_e2c_model — the e2c Stage-2 query-conditioned model, model-only.

Switchable per-cell aggregation (``supervlad`` soft / ``vlad``==``residual`` classic) via
a Registry, configurable vocabulary size ``k`` and output pyramid ``scales_cells``
(default the e2c ``(8,4,2,1)``). Self-contained: the Stage-2 core is VENDORED in
``siam_e2c_model.vendored`` (byte-faithful copy of siam_model_stage3's stage2_core) and
the residual arm is ported from the e2c script — no external RevisitAnything dependency.

Contains NO loss / batch (see ``geo_train_batching``), no split, no data IO — only the
model contract: ``build_V`` / ``encode_query`` / ``score`` / ``scales_cells`` / params.
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
