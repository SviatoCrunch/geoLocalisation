"""Per-cell aggregation strategies (SuperVLAD soft / classic-residual VLAD) + Registry."""
from __future__ import annotations

from .base import AggregationStrategy
from .registry import (register_aggregation, create_aggregation, available_aggregations)
from .supervlad import SuperVLADAggregation
from .vlad import VladAggregation

__all__ = ["AggregationStrategy", "register_aggregation", "create_aggregation",
           "available_aggregations", "SuperVLADAggregation", "VladAggregation"]
