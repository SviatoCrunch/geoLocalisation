"""Typed data structures for the SPLIT layer (not the positive subsystem).

Point/tile domain models + the materialized positive snapshot live in
``geo_split_no_overlap.positive_selection`` (``GeoPoint``, ``GalleryTile``,
``GalleryIndex``, ``MaterializedPositiveSets``). This module only holds split-side
structures (components, split result, audit report).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Split labels — strings on purpose (they land verbatim in JSON output).
TRAIN, VAL, TEST, EXCLUDED = "train", "val", "test", "excluded"
KEPT_SPLITS = (TRAIN, VAL, TEST)


@dataclass
class Component:
    """An indivisible group of points fused by positive-tile conflict."""
    component_id: int
    point_ids: list                 # sorted list[str]
    tile_ids: list                  # sorted list[str] — union of positive tile ids
    size: int                       # == len(point_ids)
    assigned: str = ""              # one of KEPT_SPLITS or EXCLUDED
    exclusion_reason: str = ""


@dataclass
class SplitResult:
    components: list                 # list[Component]
    assignment: dict                 # component_id -> split/excluded
    seed: int
    ratios: dict                     # target fractions {train,val,test}
    actual: dict                     # actual fractions {train,val,test}
    counts: dict                     # point counts {train,val,test,excluded}
    backend: str
    optimality_proven: bool
    status: str                      # "valid" | "invalid" | "infeasible"
    phase: str
    deletion_used: bool
    message: str = ""
    stage_objectives: dict = field(default_factory=dict)


@dataclass
class AuditReport:
    status: str
    checks: dict
    counts: dict
    ratios: dict
    actual: dict
    max_ratio_deviation: float
    crs: str
    area_epsilon_m2: float
    area_units: str
    cross_split_conflicts: int
    fingerprint: str
    n_components: int
    component_size_hist: dict
    largest_components: list
    excluded_points: int
    excluded_components: int
    excluded_sizes: list

    @property
    def ok(self) -> bool:
        return self.status == "valid"
