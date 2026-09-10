"""Configuration for the footprint-association testbed."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List


def geometric_scales(smin: float = 100.0, smax: float = 1000.0, ratio: float = 1.21) -> List[float]:
    """Geometric scale grid s_{i+1}=r·s_i in [smin, smax]. ratio≈1.21 → ≤~10% step
    (the doc's engineering starting point). Always includes smax."""
    if not (smin > 0 and smax > smin and ratio > 1.0):
        raise ValueError("need 0 < smin < smax and ratio > 1")
    out, s = [], float(smin)
    while s < smax * 0.999:
        out.append(round(s, 3))
        s *= ratio
    out.append(float(smax))
    return out


@dataclass
class FootprintConfig:
    # ── pyramid scale grid (metres) ───────────────────────────────────────────
    scale_min_m: float = 100.0
    scale_max_m: float = 1000.0
    scale_ratio: float = 1.21               # ~10% step
    grid_crs: str = "EPSG:3857"

    # ── estimators to run (first iteration: data-ready on tokens) ─────────────
    estimators: list = field(default_factory=lambda: ["global_vlad", "patch_overlap"])

    # patch_overlap params
    patch_ratio_test: float = 0.9           # mutual-NN ratio (1.0 = mutual-NN only)
    patch_trim: float = 0.1                 # trimmed-mean fraction each side

    # ── fusion ─────────────────────────────────────────────────────────────────
    fusion: str = "cascade"                 # cascade | rank_vote
    tau: float = 0.10                       # softmax temperature for the soft distribution
    accept_margin: float = 0.05             # min gap best-vs-2nd (fused prob) to ACCEPT
    agreement_levels: int = 1               # methods must agree within ±this many levels

    # ── footprint → gallery association ────────────────────────────────────────
    strong_cq: float = 0.7
    strong_ct: float = 0.3
    partial_min_iou: float = 0.05
    area_epsilon_m2: float = 1.0

    def scales(self) -> list:
        return geometric_scales(self.scale_min_m, self.scale_max_m, self.scale_ratio)

    def validate(self) -> None:
        self.scales()                       # raises on bad grid
        if self.fusion not in ("cascade", "rank_vote"):
            raise ValueError(f"unknown fusion {self.fusion!r}")
        for e in self.estimators:
            if e not in ("global_vlad", "patch_overlap"):
                raise ValueError(f"unknown estimator {e!r} (iteration 1: global_vlad|patch_overlap)")
        if not (0.0 <= self.patch_trim < 0.5):
            raise ValueError("patch_trim must be in [0, 0.5)")

    def to_dict(self) -> dict:
        return asdict(self)
