"""Independent audit — re-verifies a split WITHOUT trusting the optimizer.

Given a ``GalleryIndex`` + the frozen :class:`MaterializedPositiveSets` snapshot and a
``point_id -> split`` mapping, it recomputes components + geometric conflicts from
scratch and checks: partition/completeness, valid labels, atomic components, tile_id
disjointness across splits, geometric disjointness (> area_epsilon), and ratios within
tolerance. Any failed hard check -> status ``invalid``.
"""
from __future__ import annotations

from collections import defaultdict

from .config import Config
from .positive_selection import GalleryIndex, MaterializedPositiveSets
from .schemas import AuditReport, TRAIN, VAL, TEST, EXCLUDED, KEPT_SPLITS
from .components import build_components, size_histogram
from .spatial_conflicts import tile_bounds, overlap_true_m2, _candidate_pairs


def _counts(point_split: dict) -> dict:
    c = {TRAIN: 0, VAL: 0, TEST: 0, EXCLUDED: 0}
    for sp in point_split.values():
        if sp in c:
            c[sp] += 1
    return c


def _actual_fracs(counts: dict) -> dict:
    kept = counts[TRAIN] + counts[VAL] + counts[TEST]
    if kept == 0:
        return {s: 0.0 for s in KEPT_SPLITS}
    return {s: counts[s] / kept for s in KEPT_SPLITS}


def _cross_split_geo(cfg: Config, gallery: GalleryIndex, mps: MaterializedPositiveSets,
                     point_split: dict):
    """Return (n_conflicts, shared_tile_ids). Recomputes overlaps from geometry."""
    tile_splits = defaultdict(set)
    for pid, tiles in mps.point_to_tile_ids.items():
        sp = point_split.get(pid)
        if sp in KEPT_SPLITS:
            for t in tiles:
                tile_splits[t].add(sp)

    shared = sorted({t for t, ss in tile_splits.items() if len(ss) > 1})

    ids = sorted(tile_splits)
    bounds = {t: tile_bounds(gallery.get(t), cfg.tile_size_m) for t in ids}
    n_geo = 0
    for t1, t2 in _candidate_pairs(ids, bounds):
        s1, s2 = tile_splits[t1], tile_splits[t2]
        if not any(a != b for a in s1 for b in s2):
            continue
        if overlap_true_m2(gallery.get(t1), gallery.get(t2), cfg.tile_size_m) > cfg.area_epsilon_m2:
            n_geo += 1
    return n_geo + len(shared), shared


def audit(cfg: Config, gallery: GalleryIndex, mps: MaterializedPositiveSets,
          point_split: dict, claimed_status: str = "valid") -> AuditReport:
    checks = {}

    all_ids = set(mps.point_to_tile_ids)
    labelled = set(point_split)
    missing = all_ids - labelled
    extra = labelled - all_ids
    checks["partition_complete"] = {"ok": not missing and not extra,
                                    "detail": f"missing={len(missing)} extra={len(extra)}"}
    bad = {pid: sp for pid, sp in point_split.items()
           if sp not in (TRAIN, VAL, TEST, EXCLUDED)}
    checks["valid_labels"] = {"ok": not bad, "detail": f"{len(bad)} bad labels"}

    comps = build_components(gallery, mps, cfg.area_epsilon_m2, cfg.tile_size_m)
    split_comps = sum(1 for c in comps
                      if len({point_split.get(pid) for pid in c.point_ids}) > 1)
    checks["components_atomic"] = {"ok": split_comps == 0,
                                   "detail": f"{split_comps} component(s) split across splits"}

    n_conflicts, shared = _cross_split_geo(cfg, gallery, mps, point_split)
    checks["tile_id_disjoint"] = {"ok": not shared,
                                  "detail": f"{len(shared)} tile_id(s) shared across splits"}
    checks["geo_disjoint"] = {"ok": n_conflicts == 0,
                              "detail": f"{n_conflicts} cross-split geometric conflict(s)"}

    counts = _counts(point_split)
    actual = _actual_fracs(counts)
    ratios = cfg.ratios()
    kept = counts[TRAIN] + counts[VAL] + counts[TEST]
    max_dev = max(abs(actual[s] - ratios[s]) for s in KEPT_SPLITS) if kept else 1.0
    nonempty = all(counts[s] >= 1 for s in KEPT_SPLITS)
    checks["ratios_within_tolerance"] = {
        "ok": nonempty and max_dev <= cfg.ratio_tolerance + 1e-12,
        "detail": f"max_dev={max_dev:.4f} tol={cfg.ratio_tolerance} nonempty={nonempty}"}

    excluded_comps = [c for c in comps if all(point_split.get(pid) == EXCLUDED for pid in c.point_ids)]
    excluded_sizes = sorted((c.size for c in excluded_comps), reverse=True)

    hard_ok = all(v["ok"] for v in checks.values())
    if claimed_status == "infeasible":
        status = "infeasible"
    else:
        status = "valid" if hard_ok else "invalid"

    largest = sorted(comps, key=lambda c: (-c.size, c.component_id))[:10]
    return AuditReport(
        status=status, checks=checks, counts=counts, ratios=ratios, actual=actual,
        max_ratio_deviation=float(max_dev), crs=cfg.grid_crs,
        area_epsilon_m2=cfg.area_epsilon_m2, area_units=gallery.area_units,
        cross_split_conflicts=int(n_conflicts), fingerprint=mps.fingerprint,
        n_components=len(comps), component_size_hist=size_histogram(comps),
        largest_components=[{"component_id": c.component_id, "size": c.size,
                             "assigned": point_split.get(c.point_ids[0], "?")} for c in largest],
        excluded_points=counts[EXCLUDED],
        excluded_components=len(excluded_comps), excluded_sizes=excluded_sizes)
