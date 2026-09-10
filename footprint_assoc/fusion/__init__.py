"""Score fusion over pyramid levels: cascade (default) + rank_vote (experiment)."""
from __future__ import annotations

from .base import (Fusion, available_fusion, create_fusion, register_fusion,
                   softmax_masked, method_scores)
from .cascade import CascadeFusion
from .rank_vote import RankVoteFusion

__all__ = ["Fusion", "available_fusion", "create_fusion", "register_fusion",
           "softmax_masked", "method_scores", "CascadeFusion", "RankVoteFusion"]
