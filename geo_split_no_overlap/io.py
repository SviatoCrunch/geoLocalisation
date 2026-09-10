"""Read/write the run artifacts (split.json, components.json, audit.json, ...).

The resolved positive-selection config + fingerprint (from the materialized snapshot)
are persisted so a run is fully reproducible and auditable.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from .config import Config
from .positive_selection import MaterializedPositiveSets
from .schemas import (AuditReport, SplitResult, TRAIN, VAL, TEST, EXCLUDED, KEPT_SPLITS)


def point_split_map(components) -> dict:
    """point_id -> split label for all points-with-positives (from component assignment)."""
    m = {}
    for c in components:
        for pid in c.point_ids:
            m[pid] = c.assigned or EXCLUDED
    return m


def build_split_result(cfg: Config, components, opt_result) -> SplitResult:
    for c in components:
        sp = opt_result.assignment.get(c.component_id, EXCLUDED)
        c.assigned = sp
        c.exclusion_reason = "excluded_component" if sp == EXCLUDED else ""

    counts = {TRAIN: 0, VAL: 0, TEST: 0, EXCLUDED: 0}
    for c in components:
        counts[c.assigned] += c.size
    kept = counts[TRAIN] + counts[VAL] + counts[TEST]
    actual = {s: (counts[s] / kept if kept else 0.0) for s in KEPT_SPLITS}

    return SplitResult(
        components=components, assignment=dict(opt_result.assignment), seed=cfg.seed,
        ratios=cfg.ratios(), actual=actual, counts=counts, backend=opt_result.backend,
        optimality_proven=opt_result.optimality_proven, status="valid",
        phase=opt_result.phase, deletion_used=opt_result.deletion_used,
        message=opt_result.message, stage_objectives=opt_result.stage_objectives)


def write_run(out_dir, cfg: Config, mps: MaterializedPositiveSets, sr: SplitResult,
              audit: AuditReport) -> Path:
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    buckets = {TRAIN: [], VAL: [], TEST: []}
    for c in sr.components:
        if c.assigned in buckets:
            buckets[c.assigned].extend(c.point_ids)
    for k in buckets:
        buckets[k].sort()

    excluded = []
    for c in sr.components:
        if c.assigned == EXCLUDED:
            for pid in c.point_ids:
                excluded.append({"point_id": pid, "reason": "excluded_component",
                                 "component": c.component_id})
    for pid in mps.points_without_positives:
        excluded.append({"point_id": pid, "reason": "no_positive_tile", "component": None})
    excluded.sort(key=lambda e: e["point_id"])

    positive_selection = {
        "strategy": mps.strategy_name, "version": mps.strategy_version,
        "resolved_params": dict(mps.resolved_params), "stats": dict(mps.stats)}

    split_json = {
        "status": audit.status, "seed": sr.seed,
        "train": buckets[TRAIN], "val": buckets[VAL], "test": buckets[TEST],
        "excluded": excluded,
        "target_ratios": sr.ratios, "actual_ratios": sr.actual, "counts": sr.counts,
        "backend": sr.backend, "optimality_proven": sr.optimality_proven,
        "phase": sr.phase, "deletion_used": sr.deletion_used,
        "fingerprint": mps.fingerprint,
        "positive_selection": positive_selection,
        "meta": {"crs": cfg.grid_crs, "area_epsilon_m2": cfg.area_epsilon_m2,
                 "area_units": audit.area_units, "ratio_tolerance": cfg.ratio_tolerance,
                 "tiles_h5": str(cfg.tiles_h5),
                 "no_positive_count": len(mps.points_without_positives)},
    }

    components_json = [
        {"component_id": c.component_id, "point_ids": c.point_ids, "size": c.size,
         "assigned": c.assigned, "exclusion_reason": c.exclusion_reason,
         "n_positive_tiles": len(c.tile_ids)}
        for c in sr.components]

    excluded_components_json = [
        {"component_id": c.component_id, "size": c.size, "point_ids": c.point_ids,
         "reason": c.exclusion_reason or "excluded_component"}
        for c in sr.components if c.assigned == EXCLUDED]

    _dump(out / "split.json", split_json)
    _dump(out / "components.json", components_json)
    _dump(out / "excluded_components.json", excluded_components_json)
    _dump(out / "audit.json", asdict(audit))
    _write_resolved_config(out / "resolved_config.yaml", cfg, mps)
    (out / "summary.md").write_text(_summary_md(cfg, sr, audit, mps), encoding="utf-8")
    return out


def write_per_city_run(cfg: Config, run) -> Path:
    """Write the combined per-city artifacts + a per-city subdir for each city."""
    out = write_run(cfg.out_dir, cfg, run.merged_mps, run.split_result, run.combined_audit)

    data = read_split_json(out / "split.json")
    data["mode"] = "per_city"
    data["per_city"] = {
        city: {"counts": v["counts"], "actual_ratios": v["actual"],
               "max_ratio_deviation": v["max_dev"], "within_tolerance": v["within_tol"],
               "n_components": v["n_components"]}
        for city, v in run.per_city.items()}
    _dump(out / "split.json", data)

    buckets = defaultdict(lambda: {TRAIN: [], VAL: [], TEST: [], EXCLUDED: []})
    for c in run.split_result.components:
        city = c.point_ids[0].split(":", 1)[0]
        for pid in c.point_ids:
            buckets[city][c.assigned].append(pid)
    for pid in run.merged_mps.points_without_positives:
        buckets[pid.split(":", 1)[0]][EXCLUDED].append(pid)

    for city, b in buckets.items():
        pc = run.per_city.get(city, {})
        sub = {"city": city, "train": sorted(b[TRAIN]), "val": sorted(b[VAL]),
               "test": sorted(b[TEST]), "excluded": sorted(b[EXCLUDED]),
               "target_ratios": cfg.ratios(), "actual_ratios": pc.get("actual", {}),
               "within_tolerance": pc.get("within_tol")}
        cdir = out / "cities" / city
        cdir.mkdir(parents=True, exist_ok=True)
        _dump(cdir / "split.json", sub)
    return out


def read_split_json(path) -> dict:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def point_split_for_viz(data: dict) -> dict:
    """point_id -> split incl. ALL excluded (component + no-positive) for KMZ colouring."""
    m = {}
    for s in (TRAIN, VAL, TEST):
        for pid in data.get(s, []):
            m[pid] = s
    for e in data.get("excluded", []):
        m[e["point_id"]] = EXCLUDED
    return m


def point_split_from_split_json(data: dict) -> dict:
    m = {}
    for s in (TRAIN, VAL, TEST):
        for pid in data.get(s, []):
            m[pid] = s
    for e in data.get("excluded", []):
        if e.get("reason") == "excluded_component":
            m[e["point_id"]] = EXCLUDED
    return m


def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_resolved_config(path: Path, cfg: Config, mps: MaterializedPositiveSets) -> None:
    import yaml
    d = cfg.to_dict()
    d["positive_selection"] = {
        "strategy": mps.strategy_name,
        "params": dict(cfg.positive_selection.get("params", {})),
        "resolved_params": dict(mps.resolved_params),
        "version": mps.strategy_version,
    }
    path.write_text(yaml.safe_dump(d, sort_keys=True, allow_unicode=True), encoding="utf-8")


def _summary_md(cfg: Config, sr: SplitResult, audit: AuditReport,
                mps: MaterializedPositiveSets) -> str:
    lines = [
        "# geo_split_no_overlap — run summary", "",
        f"- **status**: `{audit.status}`",
        f"- **positive strategy**: `{mps.strategy_name}` v{mps.strategy_version}",
        f"- **backend**: `{sr.backend}` (optimality_proven={sr.optimality_proven})",
        f"- **phase**: `{sr.phase}`  deletion_used={sr.deletion_used}",
        f"- **fingerprint**: `{mps.fingerprint[:16]}…`",
        f"- **CRS**: {audit.crs}  |  **area_epsilon**: {audit.area_epsilon_m2} {audit.area_units}",
        "",
        "## Points",
        f"- with positives: {sum(sr.counts.values())}  |  no-positive: "
        f"{len(mps.points_without_positives)}",
        f"- train / val / test / excluded: "
        f"{sr.counts[TRAIN]} / {sr.counts[VAL]} / {sr.counts[TEST]} / {sr.counts[EXCLUDED]}",
        f"- positives/point: {mps.stats.get('positives_per_point_hist')}",
        "",
        "## Ratios (of kept)",
        f"- target: {sr.ratios}",
        f"- actual: { {k: round(v, 4) for k, v in sr.actual.items()} }",
        f"- max deviation: {audit.max_ratio_deviation:.4f} (tol {cfg.ratio_tolerance})",
        "",
        "## Components",
        f"- total: {audit.n_components}  |  size histogram: {audit.component_size_hist}",
        f"- largest: {[(c['size'], c['assigned']) for c in audit.largest_components[:5]]}",
        f"- excluded components: {audit.excluded_components}  sizes: {audit.excluded_sizes}",
        "",
        "## Independent audit",
    ]
    for name, res in audit.checks.items():
        lines.append(f"- [{'PASS' if res['ok'] else 'FAIL'}] **{name}**: {res['detail']}")
    lines.append(f"- cross-split geometric conflicts: **{audit.cross_split_conflicts}**")
    lines.append("")
    return "\n".join(lines)
