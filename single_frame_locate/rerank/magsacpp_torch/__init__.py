"""Vendored GPU MAGSAC++ core (faithful Torch port), copied verbatim from
geoLocalisation/experiments/magsacpp_torch (config/gamma/solver/estimator/batch). Runtime-only subset:
no oracle/cli/evaluate/search. Pure torch/numpy tensor ops — one import, no cross-repo dependency.

Used by service.rerank.reranker when SIAM_RERANK_VERIFY=torch to replace the cv2 MAGSAC verifier with
the batched GPU estimator. Default stays cv2. See service/rerank for the switch.
"""
from __future__ import annotations

from .batch import estimate_homography_magsacpp_batch
from .config import MagsacppConfig

__all__ = ["MagsacppConfig", "estimate_homography_magsacpp_batch"]
