"""DSS batch construction (model-agnostic): pairs, neighbour cache, planner, Y[B,B]."""
from __future__ import annotations

from .cross_relevance import build_cross_relevance, build_cross_weights
from .neighbours import NeighbourCache, build_neighbour_cache
from .pairs import CanonicalPair, build_pair_pool, pair_pool_to_json
from .planner import BatchPlan, plan_logical_batch, microbatch_ranges

__all__ = [
    "CanonicalPair", "build_pair_pool", "pair_pool_to_json",
    "NeighbourCache", "build_neighbour_cache",
    "BatchPlan", "plan_logical_batch", "microbatch_ranges",
    "build_cross_relevance", "build_cross_weights",
]
