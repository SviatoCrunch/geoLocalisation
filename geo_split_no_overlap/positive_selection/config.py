"""Parsing of the ``positive_selection`` config block.

    positive_selection:
      strategy: current_rule
      params:
        query_size_m: 0.0

Only the shape is validated here; the *values* in ``params`` are validated by the
chosen strategy's factory (single source of truth for each rule's parameters).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class PositiveSelectionConfig:
    strategy: str
    params: Mapping[str, object] = field(default_factory=dict)

    @staticmethod
    def from_mapping(m: Mapping) -> "PositiveSelectionConfig":
        if not isinstance(m, Mapping):
            raise TypeError(f"positive_selection must be a mapping, got {type(m).__name__}")
        unknown = set(m) - {"strategy", "params"}
        if unknown:
            raise ValueError(f"positive_selection: unknown keys {sorted(unknown)} "
                             f"(allowed: strategy, params)")
        strategy = m.get("strategy")
        if not strategy or not isinstance(strategy, str):
            raise ValueError("positive_selection.strategy must be a non-empty string")
        params = m.get("params") or {}
        if not isinstance(params, Mapping):
            raise TypeError("positive_selection.params must be a mapping")
        return PositiveSelectionConfig(strategy=strategy, params=dict(params))
