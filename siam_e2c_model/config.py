"""Model configuration for the e2c Stage-2 query-conditioned model.

Only architecture knobs live here (aggregation, K/vocab, pyramid, dims). Data concerns
(grid resampling, H5 IO) and training (loss/batch/optimizer) are elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class E2cModelConfig:
    # ── aggregation (switchable) + vocabulary size ────────────────────────────
    agg: str = "supervlad"                  # supervlad | vlad | residual
    k: int = 32                             # n_groups (SuperVLAD/VLAD clusters); "the size"
    assign_path: str = ""                   # recast k*.pt: assign_weight [+ centroids for vlad]

    # ── output pyramid (default = e2c; changeable) ────────────────────────────
    scales_cells: tuple = (8, 4, 2, 1)

    # ── dims / vocab ──────────────────────────────────────────────────────────
    d_token: int = 1536
    d_group: int = 32                       # PerGroupProjection output dim
    d_out: int = 256                        # projection head output
    d_hidden: int = 512                     # projection head hidden
    dropout: float = 0.1
    tau_min: float = 0.01
    tau_init: float = 0.1
    intra: bool = True
    freeze_assignment: bool = True

    def validate(self) -> None:
        if self.agg not in ("supervlad", "vlad", "residual"):
            raise ValueError(f"unknown agg {self.agg!r} (supervlad|vlad|residual)")
        if int(self.k) <= 0:
            raise ValueError("k (n_groups) must be > 0")
        sc = tuple(int(n) for n in self.scales_cells)
        if not sc or any(n < 1 for n in sc):
            raise ValueError(f"scales_cells must be non-empty ints >= 1, got {self.scales_cells}")
        object.__setattr__(self, "scales_cells", sc)
        for name in ("d_token", "d_group", "d_out", "d_hidden"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["scales_cells"] = list(self.scales_cells)
        return d
