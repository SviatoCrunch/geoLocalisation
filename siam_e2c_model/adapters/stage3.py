"""The ONLY reach into ``siam_model_stage3`` — reused, never modified.

Thin wrappers over the Stage-2 query-conditioned factory + assignment loading, so the
rest of ``siam_e2c_model`` depends on stage3 through this one module. All imports are
lazy (inside functions) so importing the package is cheap and does not pull torch/stage3
until a model is actually built.
"""
from __future__ import annotations

from pathlib import Path


def build_core_model(**kwargs):
    """Build the Stage-2 query-conditioned model (``Stage2QueryConditionedModel``)."""
    from siam_model_stage3.model.stage2_core import build_stage2_query_conditioned_model
    return build_stage2_query_conditioned_model(**kwargs)


def load_assign_weight(path, *, k=None, d=None):
    """Load the recast SuperVLAD ``assign_weight`` (K, D) from a k*.pt blob."""
    from siam_model_stage3.model.factory import load_assign_weight as _law
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
    """Token→cell id map for an n×n grid (reused stage3 helper, read-only)."""
    from siam_model_stage3.model.stage2_core import _region_ids_cached
    return _region_ids_cached(H, W, n, dev)
