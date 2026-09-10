"""SuperVLAD aggregation — soft assignment ``α = softmax(assign)``, value = x̄.

This is the built-in e2c arm: it delegates straight to the Stage-2 model's own
``build_V`` / ``encode_query_from_tokens`` (which use the frozen ``assign_weight``).
No centroids needed.
"""
from __future__ import annotations

from typing import Mapping


class SuperVLADAggregation:
    name = "supervlad"

    @classmethod
    def build(cls, *, blob=None, centroids=None, cfg=None) -> "SuperVLADAggregation":
        return cls()

    def build_V(self, core, grids):
        return core.build_V(grids)

    def encode_query(self, core, tokens):
        return core.encode_query_from_tokens(tokens)

    def resolved_config(self) -> Mapping[str, object]:
        return {"agg": "supervlad", "assign": "softmax(frozen)", "value": "soft_mean_xbar"}
