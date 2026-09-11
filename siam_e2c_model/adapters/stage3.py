"""Adapter to the VENDORED Stage-2 core (``siam_e2c_model.vendored``).

Self-contained: imports only the in-package vendored copy of the Stage-2
query-conditioned model + assignment loader (both numpy/torch only). No dependency on
the external ``siam_model_stage3`` / RevisitAnything repo. Imports are lazy so importing
the package stays cheap.
"""
from __future__ import annotations

from pathlib import Path


def build_core_model(**kwargs):
    """Build the CELL Stage-2 query-conditioned model (``Stage2QueryConditionedModel``)."""
    from ..vendored.stage2_core import build_stage2_query_conditioned_model
    return build_stage2_query_conditioned_model(**kwargs)


def build_concentric_core(**kwargs):
    """Build the CONCENTRIC-pyramid model (``ConcentricStage2Model``); same agg construction."""
    from ..concentric_core import build_concentric_model
    return build_concentric_model(**kwargs)


def load_assign_weight(path, *, k=None, d=None):
    """Load the recast SuperVLAD ``assign_weight`` (K, D) from a k*.pt blob."""
    from ..vendored.assign_io import load_assign_weight as _law
    return _law(Path(path).expanduser(), k=k, d=d)


def load_blob(path):
    """Load the full recast blob (has ``assign_weight`` and ``centroids``)."""
    import torch
    return torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)


def normalize_centroids(centroids):
    """L2-normalise centroids row-wise (matches the e2c residual arm)."""
    import torch
    import torch.nn.functional as F
    return F.normalize(torch.as_tensor(centroids).float(), dim=1)


def region_ids_cached(H: int, W: int, n: int, dev):
    """Token→cell id map for an n×n grid (vendored stage2_core helper)."""
    from ..vendored.stage2_core import _region_ids_cached
    return _region_ids_cached(H, W, n, dev)
