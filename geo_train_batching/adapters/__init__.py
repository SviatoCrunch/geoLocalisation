"""Adapters to external packages. Reach these only through the public API."""
from __future__ import annotations

from .split_relevance import SplitRelevance, build_split_relevance, build_pairs_from_split

__all__ = ["SplitRelevance", "build_split_relevance", "build_pairs_from_split"]
