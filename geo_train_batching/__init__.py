"""geo_train_batching — model-agnostic training batch construction + losses.

Holds ONLY the loss + batch-creation + relevance logic for retrieval training. It does
NOT contain a model: losses take score tensors / a ``score_fn``; the batcher takes
pair embeddings as plain arrays. The model lives in a separate package and is wired in
by the (future) training loop.

Integrates with ``geo_split_no_overlap``: :func:`build_split_relevance` reuses that
package's positive-selection snapshot (so training positives == the split's) and its
train/val/test assignment.

Layers:
    loss/       multi-positive CE, full-gallery NLL (dense+streaming), symmetric InfoNCE
    relevance/  RelevanceTable (pos/safe/ignore) — geometry or explicit
    batching/   canonical pairs, neighbour cache, logical-batch planner, Y[B,B]
    adapters/   split integration (the only cross-package dependency)
"""
from __future__ import annotations

from .loss import (TAU_LOSS_DEFAULT, masked_logsumexp, multipositive_ce,
                   symmetric_multipositive_ce, dense_full_gallery_nll,
                   streaming_full_gallery_nll, streaming_soft_relevance_ce, validate_masks,
                   symmetric_infonce, LOGIT_SCALE_DEFAULT)
from .relevance import RelevanceTable, GeometryRelevanceTable, ExplicitRelevanceTable
from .batching import (CanonicalPair, build_pair_pool, pair_pool_to_json, NeighbourCache,
                       build_neighbour_cache, BatchPlan, plan_logical_batch, microbatch_ranges,
                       build_cross_relevance)

__all__ = [
    # loss
    "TAU_LOSS_DEFAULT", "masked_logsumexp", "multipositive_ce", "symmetric_multipositive_ce",
    "dense_full_gallery_nll", "streaming_full_gallery_nll", "streaming_soft_relevance_ce",
    "validate_masks", "symmetric_infonce", "LOGIT_SCALE_DEFAULT",
    # relevance
    "RelevanceTable", "GeometryRelevanceTable", "ExplicitRelevanceTable",
    # batching
    "CanonicalPair", "build_pair_pool", "pair_pool_to_json", "NeighbourCache",
    "build_neighbour_cache", "BatchPlan", "plan_logical_batch", "microbatch_ranges",
    "build_cross_relevance",
]

__version__ = "0.1.0"
