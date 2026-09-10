"""Relevance tables (pos/safe/ignore) consumed by the batcher and losses."""
from __future__ import annotations

from .table import (RelevanceTable, GeometryRelevanceTable, ExplicitRelevanceTable)

__all__ = ["RelevanceTable", "GeometryRelevanceTable", "ExplicitRelevanceTable"]
