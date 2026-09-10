"""Configuration for the split layer.

The positive RULE is no longer configured here field-by-field — it is selected via the
``positive_selection`` block (strategy + params), resolved by the Registry. This
module keeps only geometry (tile-size fallback, CRS, area_epsilon), split targets,
policy and IO.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from .positive_selection import PositiveSelectionConfig


@dataclass
class Config:
    # ── inputs (existing artifacts) ───────────────────────────────────────────
    # Single-gallery mode: one gallery + one or more gt roots (balanced globally).
    gt: list = field(default_factory=list)      # ["city=dir", ...]
    tiles_h5: str = ""                          # multicity_vlad_*.h5
    # Per-city mode: one (points, H5) pair per city; each city is split independently
    # (train/val/test) then the per-city splits are unioned. Presence of `cities`
    # selects this mode.
    cities: list = field(default_factory=list)  # [{city, gt, tiles_h5}, ...]

    # ── positive-rule selection (strategy + params) ───────────────────────────
    positive_selection: dict = field(default_factory=lambda: {"strategy": "current_rule",
                                                              "params": {}})

    # ── geometry / leakage ────────────────────────────────────────────────────
    tile_size_m: float = 1000.0                 # fallback when a tile lacks window_size_m
    area_epsilon_m2: float = 1.0                # true-m^2 overlap at/below this is NOT leakage
    grid_crs: str = "EPSG:3857"

    # ── split targets ─────────────────────────────────────────────────────────
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    ratio_tolerance: float = 0.05
    seed: int = 0

    # ── policy ────────────────────────────────────────────────────────────────
    no_positive_policy: str = "exclude"         # "exclude" | "error"
    solver: str = "auto"                        # auto|exact|cpsat|milp|heuristic
    solver_time_limit_s: float = 60.0
    solver_workers: int = 8

    # ── output ────────────────────────────────────────────────────────────────
    out_dir: str = ""

    def ratios(self) -> dict:
        return {"train": self.train_ratio, "val": self.val_ratio, "test": self.test_ratio}

    def positive_selection_config(self) -> PositiveSelectionConfig:
        return PositiveSelectionConfig.from_mapping(self.positive_selection)

    def is_per_city(self) -> bool:
        return bool(self.cities)

    def validate(self) -> None:
        r = (self.train_ratio, self.val_ratio, self.test_ratio)
        if any(x < 0 for x in r) or abs(sum(r) - 1.0) > 1e-6:
            raise ValueError(f"ratios must be non-negative and sum to 1.0, got {r}")
        if not (0.0 <= self.ratio_tolerance < 1.0):
            raise ValueError(f"ratio_tolerance must be in [0,1), got {self.ratio_tolerance}")
        if self.area_epsilon_m2 < 0:
            raise ValueError("area_epsilon_m2 must be >= 0")
        if self.no_positive_policy not in ("exclude", "error"):
            raise ValueError("no_positive_policy must be 'exclude' or 'error'")
        if self.solver not in ("auto", "exact", "cpsat", "milp", "heuristic"):
            raise ValueError(f"unknown solver {self.solver!r}")
        if self.is_per_city():
            if self.gt or self.tiles_h5:
                raise ValueError("per-city mode uses `cities`; do not also set top-level gt/tiles_h5")
            for i, e in enumerate(self.cities):
                if not isinstance(e, dict):
                    raise ValueError(f"cities[{i}] must be a mapping {{city, gt, tiles_h5}}")
                if not e.get("gt") or not e.get("tiles_h5"):
                    raise ValueError(f"cities[{i}] needs both 'gt' and 'tiles_h5'")
        else:
            if not self.gt:
                raise ValueError("config.gt is empty (need at least one 'city=dir' entry)")
            if not self.tiles_h5:
                raise ValueError("config.tiles_h5 is required")
        self.positive_selection_config()        # shape-validate the block early

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path, overrides: Optional[dict] = None) -> Config:
    import yaml
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    known = {f for f in Config().__dataclass_fields__}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)} (allowed: {sorted(known)})")
    cfg = Config(**{k: v for k, v in raw.items() if k in known})
    if overrides:
        for k, v in overrides.items():
            if v is not None and k in known:
                setattr(cfg, k, v)
    cfg.validate()
    return cfg
