"""Per-cell aggregation strategy contract.

Both arms share the SAME e2c pyramid / heads / scale-gate / rho (the Stage-2 model);
ONLY the per-cell aggregation differs. A strategy turns the frozen model + token grids
into the map cell-pyramid ``V`` and a query token set into the query embedding ``q``.
"""
from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable


@runtime_checkable
class AggregationStrategy(Protocol):
    @property
    def name(self) -> str:
        """Stable registry name (e.g. ``"supervlad"`` / ``"vlad"``)."""
        ...

    def build_V(self, core, grids):
        """Map token grids (M,H,W,D) → cell pyramid ``{n:(M,n²,d_out)}`` (via ``core``)."""
        ...

    def encode_query(self, core, tokens):
        """Query tokens (N,D) → query embedding ``(d_out,)`` (via ``core``)."""
        ...

    def resolved_config(self) -> Mapping[str, object]:
        """Params + variant identity, recorded in the model's resolved config."""
        ...
