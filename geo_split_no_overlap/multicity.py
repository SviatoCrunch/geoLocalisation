"""Per-city split: split EACH city into train/val/test independently, then union.

Input is one ``(points, H5 gallery)`` pair per city (``cfg.cities``). Each city's
positives + conflict components are built on its own gallery, and the optimizer runs
**per city** with the same target ratios — so every city is individually 70/15/15
(within tolerance). The per-city assignments are then unioned.

Because the galleries do not overlap geographically, components never cross city
boundaries; the union is leakage-free. A single COMBINED audit (over the merged
gallery + merged snapshot) re-verifies cross-city disjointness for free, and per-city
ratios are checked individually. Uses only the positive_selection public API — no
strategy internals, and the positive rule is identical to the single-gallery mode.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from . import optimizer
from .audit import audit
from .components import build_components
from .config import Config
from .io import build_split_result
from .optimizer import OptimizerResult
from .positive_selection import (GalleryIndex, build_gallery_index, build_geo_points,
                                 create_positive_selector, materialize_positive_sets)
from .schemas import EXCLUDED, KEPT_SPLITS, TRAIN, VAL, TEST


@dataclass
class PerCityRun:
    status: str
    per_city: dict = field(default_factory=dict)       # city -> {counts, actual, ...}
    infeasible: dict = field(default_factory=dict)      # city -> message
    merged_gallery: Optional[GalleryIndex] = None
    merged_mps: object = None
    all_points: tuple = ()
    point_split: dict = field(default_factory=dict)
    combined_audit: object = None
    split_result: object = None


def _gt_dir(gt) -> str:
    gt = str(gt)
    return gt.split("=", 1)[1] if "=" in gt else gt


def _load_city(cfg: Config, entry: dict):
    g = build_gallery_index(entry["tiles_h5"], cfg.grid_crs, cfg.tile_size_m)
    cs = g.cities()
    # single-city H5 -> use its own label so point.city matches tile.city; else the
    # configured name (same_city_only then disambiguates within the merged gallery).
    label = cs[0] if len(cs) == 1 else str(entry.get("city") or "city")
    pts = build_geo_points([f"{label}={_gt_dir(entry['gt'])}"], cfg.grid_crs)
    return g, pts, label, g.area_units


def load_merged(cfg: Config):
    """Load every city's (gallery, points), merge into one gallery + point list."""
    selector = create_positive_selector(cfg.positive_selection_config())
    all_tiles, all_points, area_units, seen = [], [], None, set()
    for e in cfg.cities:
        g, pts, label, units = _load_city(cfg, e)
        if label in seen:
            raise ValueError(f"duplicate city label {label!r} across cfg.cities entries")
        seen.add(label)
        area_units = area_units or units
        all_tiles.extend(g.all_tiles())
        all_points.extend(pts)
    merged = GalleryIndex(all_tiles, crs=cfg.grid_crs, area_units=area_units or "",
                          tile_size_fallback=cfg.tile_size_m)          # raises on tile_id collision
    mps = materialize_positive_sets(all_points, merged, selector)
    return merged, tuple(all_points), mps


def _group_components_by_city(components, all_points):
    pid2city = {p.point_id: p.city for p in all_points}
    by_city = defaultdict(list)
    for c in components:
        cities = {pid2city[pid] for pid in c.point_ids}
        if len(cities) != 1:
            raise ValueError(
                f"component {c.component_id} spans cities {sorted(cities)} — the galleries "
                f"overlap across cities, so a per-city union would NOT be leakage-safe")
        by_city[next(iter(cities))].append(c)
    return by_city


def run_per_city(cfg: Config) -> PerCityRun:
    merged, all_points, mps = load_merged(cfg)
    if mps.points_without_positives and cfg.no_positive_policy == "error":
        raise ValueError(f"{len(mps.points_without_positives)} point(s) have no positive "
                         f"and no_positive_policy='error'")

    components = build_components(merged, mps, cfg.area_epsilon_m2, cfg.tile_size_m)
    by_city = _group_components_by_city(components, all_points)

    assignment, infeasible = {}, {}
    backends, proven, deletion = set(), True, False
    for city in sorted(by_city):
        opt = optimizer.solve(by_city[city], cfg.ratios(), cfg.ratio_tolerance, seed=cfg.seed,
                              solver=cfg.solver, time_limit_s=cfg.solver_time_limit_s,
                              workers=cfg.solver_workers)
        if opt.status != "ok":
            infeasible[city] = opt.message
            continue
        assignment.update(opt.assignment)
        backends.add(opt.backend)
        proven = proven and opt.optimality_proven
        deletion = deletion or opt.deletion_used

    if infeasible:
        return PerCityRun(status="infeasible", infeasible=infeasible,
                          per_city={c: {"n_components": len(by_city[c])} for c in by_city})

    for c in components:
        c.assigned = assignment.get(c.component_id, EXCLUDED)
        c.exclusion_reason = "excluded_component" if c.assigned == EXCLUDED else ""

    per_city = {}
    for city, comps_c in by_city.items():
        counts = {TRAIN: 0, VAL: 0, TEST: 0, EXCLUDED: 0}
        for c in comps_c:
            counts[c.assigned] += c.size
        kept = counts[TRAIN] + counts[VAL] + counts[TEST]
        actual = {s: (counts[s] / kept if kept else 0.0) for s in KEPT_SPLITS}
        max_dev = max(abs(actual[s] - cfg.ratios()[s]) for s in KEPT_SPLITS) if kept else 1.0
        within = kept > 0 and all(counts[s] >= 1 for s in KEPT_SPLITS) \
            and max_dev <= cfg.ratio_tolerance + 1e-12
        per_city[city] = {"counts": counts, "actual": actual, "max_dev": max_dev,
                          "within_tol": within, "n_components": len(comps_c)}

    point_split = {pid: c.assigned for c in components for pid in c.point_ids}
    combined_audit = audit(cfg, merged, mps, point_split)
    all_within = all(v["within_tol"] for v in per_city.values())
    status = "valid" if (combined_audit.status == "valid" and all_within) else "invalid"

    optlike = OptimizerResult(
        assignment=assignment, backend="per_city(" + ",".join(sorted(backends)) + ")",
        status="ok", optimality_proven=proven, phase="per_city", deletion_used=deletion)
    sr = build_split_result(cfg, components, optlike)
    sr.status = status

    return PerCityRun(status=status, per_city=per_city, merged_gallery=merged, merged_mps=mps,
                      all_points=all_points, point_split=point_split,
                      combined_audit=combined_audit, split_result=sr)
