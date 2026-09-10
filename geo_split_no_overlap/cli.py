"""CLI: build a leakage-free split, audit one, export a KMZ, or list strategies.

    python -m geo_split_no_overlap.cli build  --config cfg.yaml [--dry-run] [--kmz]
    python -m geo_split_no_overlap.cli audit  --split run/split.json --config cfg.yaml
    python -m geo_split_no_overlap.cli export-kmz --config cfg.yaml --split run/split.json --out s.kmz [--tiles]
    python -m geo_split_no_overlap.cli list-positive-strategies

Two build modes, chosen by the config: single-gallery (`gt` + `tiles_h5`, balanced
globally) or per-city (`cities: [{city, gt, tiles_h5}]`, each city split
independently then unioned). The positive RULE is chosen by the `positive_selection`
block + Registry — no rule logic here. Run from the repo root.
Exit codes: 0 valid · 1 invalid · 2 infeasible/error.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import audit as audit_mod
from . import io as run_io
from . import kmz as kmz_mod
from . import multicity
from . import optimizer
from .components import build_components, size_histogram
from .config import load_config
from .positive_selection import (available_positive_selectors, build_gallery_index,
                                 build_geo_points, create_positive_selector,
                                 materialize_positive_sets)
from .schemas import EXCLUDED


def _overrides(args) -> dict:
    ov = {}
    for src, dst in (("seed", "seed"), ("out", "out_dir"), ("area_epsilon", "area_epsilon_m2"),
                     ("tolerance", "ratio_tolerance"), ("solver", "solver")):
        v = getattr(args, src, None)
        if v is not None:
            ov[dst] = v
    if getattr(args, "ratios", None):
        parts = [float(x) for x in args.ratios.split(",")]
        if len(parts) != 3:
            raise SystemExit("--ratios must be train,val,test")
        ov["train_ratio"], ov["val_ratio"], ov["test_ratio"] = parts
    return ov


def _inputs(cfg):
    """(gallery, points, mps) for the current mode (single gallery or merged per-city)."""
    if cfg.is_per_city():
        return multicity.load_merged(cfg)
    gallery = build_gallery_index(cfg.tiles_h5, cfg.grid_crs, cfg.tile_size_m)
    points = build_geo_points(cfg.gt, cfg.grid_crs)
    selector = create_positive_selector(cfg.positive_selection_config())
    mps = materialize_positive_sets(points, gallery, selector)
    return gallery, points, mps


def _check_no_positive(cfg, mps) -> None:
    if mps.points_without_positives and cfg.no_positive_policy == "error":
        sample = ", ".join(mps.points_without_positives[:10])
        raise SystemExit(f"[error] {len(mps.points_without_positives)} point(s) have no "
                         f"positive and no_positive_policy='error': {sample}"
                         + (" ..." if len(mps.points_without_positives) > 10 else ""))


def _emit_kmz(path, cfg, points, gallery, point_split, mps, draw_tiles) -> Path:
    viz = dict(point_split)
    for pid in mps.points_without_positives:
        viz.setdefault(pid, EXCLUDED)
    return kmz_mod.write_kmz(path, points, gallery, viz,
                             point_to_tiles=dict(mps.point_to_tile_ids),
                             draw_tiles=draw_tiles, tile_size_fallback=cfg.tile_size_m,
                             name=Path(cfg.out_dir or "geo_split").name)


# ── dry-run ──────────────────────────────────────────────────────────────────────
def _dry_run(cfg) -> int:
    gallery, points, mps = _inputs(cfg)
    comps = build_components(gallery, mps, cfg.area_epsilon_m2, cfg.tile_size_m)
    mode = "per-city" if cfg.is_per_city() else "single-gallery"
    print(f"[dry-run] inputs validated OK  (mode: {mode})")
    print(f"  gallery            : {len(gallery)} tiles, CRS {gallery.crs}, cities {list(gallery.cities())}")
    print(f"  strategy           : {mps.strategy_name} v{mps.strategy_version} "
          f"params={dict(mps.resolved_params)}")
    print(f"  points w/ positives: {mps.stats['n_points_with_positives']}   "
          f"no-positive: {mps.stats['n_points_without_positives']} (policy={cfg.no_positive_policy})")
    print(f"  unique positive tiles: {mps.stats['unique_positive_tiles']}   components: {len(comps)}")
    if cfg.is_per_city():
        by = {}
        for c in comps:
            by.setdefault(c.point_ids[0].split(":", 1)[0], [0, 0])
        for c in comps:
            e = by[c.point_ids[0].split(":", 1)[0]]
            e[0] += 1; e[1] += c.size
        for city in sorted(by):
            print(f"    - {city}: {by[city][0]} components, {by[city][1]} points")
    print(f"  fingerprint        : {mps.fingerprint}")
    if mps.points_without_positives and cfg.no_positive_policy == "error":
        print("  [error] no_positive_policy='error' and some points have no positive")
        return 2
    print("[dry-run] no split written.")
    return 0


# ── build ────────────────────────────────────────────────────────────────────────
def _build(args) -> int:
    cfg = load_config(args.config, _overrides(args))
    if not cfg.out_dir and not args.dry_run:
        raise SystemExit("out_dir is required (config.out_dir or --out) unless --dry-run")
    if args.dry_run:
        return _dry_run(cfg)
    return _build_per_city(cfg, args) if cfg.is_per_city() else _build_single(cfg, args)


def _build_single(cfg, args) -> int:
    gallery, points, mps = _inputs(cfg)
    _check_no_positive(cfg, mps)

    components = build_components(gallery, mps, cfg.area_epsilon_m2, cfg.tile_size_m)
    opt = optimizer.solve(components, cfg.ratios(), cfg.ratio_tolerance, seed=cfg.seed,
                          solver=cfg.solver, time_limit_s=cfg.solver_time_limit_s,
                          workers=cfg.solver_workers)
    if opt.status != "ok":
        return _write_infeasible(cfg, opt.message, mps.fingerprint)

    sr = run_io.build_split_result(cfg, components, opt)
    point_split = run_io.point_split_map(components)
    report = audit_mod.audit(cfg, gallery, mps, point_split, claimed_status=sr.status)
    sr.status = report.status
    out = run_io.write_run(cfg.out_dir, cfg, mps, sr, report)

    print(f"[{report.status.upper()}] {out}   (single-gallery)")
    print(f"  strategy={mps.strategy_name} v{mps.strategy_version} fp={mps.fingerprint[:12]}")
    print(f"  backend={sr.backend} proven={sr.optimality_proven} phase={sr.phase} deletion={sr.deletion_used}")
    print(f"  counts={sr.counts}  actual={ {k: round(v,4) for k,v in sr.actual.items()} }")
    print(f"  cross-split conflicts={report.cross_split_conflicts}")
    if args.kmz:
        p = _emit_kmz(out / "split.kmz", cfg, points, gallery, point_split, mps, args.kmz_tiles)
        print(f"  kmz={p}")
    return 0 if report.status == "valid" else 1


def _build_per_city(cfg, args) -> int:
    run = multicity.run_per_city(cfg)
    if run.status == "infeasible":
        msg = "; ".join(f"{c}: {m}" for c, m in run.infeasible.items())
        return _write_infeasible(cfg, "per-city infeasible -> " + msg, "")

    out = run_io.write_per_city_run(cfg, run)
    a = run.combined_audit
    print(f"[{run.status.upper()}] {out}   (per-city: {len(run.per_city)} cities)")
    print(f"  strategy={run.merged_mps.strategy_name} fp={run.merged_mps.fingerprint[:12]}")
    print(f"  combined counts={run.split_result.counts}  "
          f"global actual={ {k: round(v,4) for k,v in run.split_result.actual.items()} }")
    for city in sorted(run.per_city):
        v = run.per_city[city]
        ok = "ok" if v["within_tol"] else "OUT-OF-TOL"
        print(f"    - {city}: {v['counts']}  actual={ {k: round(x,4) for k,x in v['actual'].items()} } [{ok}]")
    print(f"  cross-split conflicts={a.cross_split_conflicts}")
    if args.kmz:
        p = _emit_kmz(out / "split.kmz", cfg, run.all_points, run.merged_gallery,
                      run.point_split, run.merged_mps, args.kmz_tiles)
        print(f"  kmz={p}")
    return 0 if run.status == "valid" else 1


def _write_infeasible(cfg, message, fingerprint) -> int:
    out = Path(cfg.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    run_io._dump(out / "split.json", {"status": "infeasible", "message": message,
                                      "seed": cfg.seed, "train": [], "val": [], "test": [],
                                      "excluded": [], "fingerprint": fingerprint})
    (out / "summary.md").write_text(
        f"# geo_split_no_overlap\n\n**status: infeasible**\n\n{message}\n", encoding="utf-8")
    print(f"[INFEASIBLE] {message}")
    return 2


# ── audit ────────────────────────────────────────────────────────────────────────
def _audit(args) -> int:
    cfg = load_config(args.config, _overrides(args))
    data = run_io.read_split_json(args.split)
    gallery, _points, mps = _inputs(cfg)                 # recompute independently
    point_split = run_io.point_split_from_split_json(data)
    report = audit_mod.audit(cfg, gallery, mps, point_split, claimed_status=data.get("status", "valid"))

    fp_ok = data.get("fingerprint") in (None, mps.fingerprint)
    print(f"[audit] status={report.status}  fingerprint_match={fp_ok}")
    for name, res in report.checks.items():
        print(f"  {'OK ' if res['ok'] else 'FAIL'} {name}: {res['detail']}")
    print(f"  cross-split geometric conflicts: {report.cross_split_conflicts}")
    if not fp_ok:
        print("  [warn] split fingerprint != recomputed inputs — inputs/strategy may have changed")
    if report.status == "infeasible":
        return 2
    return 0 if report.status == "valid" else 1


# ── export-kmz ─────────────────────────────────────────────────────────────────────
def _export_kmz(args) -> int:
    cfg = load_config(args.config, _overrides(args))
    data = run_io.read_split_json(args.split)
    gallery, points, mps = _inputs(cfg)
    point_split = run_io.point_split_for_viz(data)
    p = kmz_mod.write_kmz(args.out, points, gallery, point_split,
                          point_to_tiles=dict(mps.point_to_tile_ids),
                          draw_tiles=args.tiles, tile_size_fallback=cfg.tile_size_m,
                          name=Path(args.out).stem)
    print(f"[kmz] {p}  (points={len(points)}, tiles={'yes' if args.tiles else 'no'})")
    return 0


def _list_strategies(args) -> int:
    for name in available_positive_selectors():
        print(name)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="geo_split_no_overlap",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build a leakage-free split (single-gallery or per-city)")
    b.add_argument("--config", required=True, type=Path)
    b.add_argument("--dry-run", action="store_true", help="validate + report, write nothing")
    b.add_argument("--out", type=str)
    b.add_argument("--seed", type=int)
    b.add_argument("--ratios", type=str, help="train,val,test override")
    b.add_argument("--tolerance", type=float)
    b.add_argument("--area-epsilon", dest="area_epsilon", type=float)
    b.add_argument("--solver", choices=["auto", "exact", "cpsat", "milp", "heuristic"])
    b.add_argument("--kmz", action="store_true", help="also write split.kmz")
    b.add_argument("--kmz-tiles", action="store_true", help="include positive-tile footprints in the KMZ")
    b.set_defaults(func=_build)

    a = sub.add_parser("audit", help="independently audit an existing split.json")
    a.add_argument("--config", required=True, type=Path)
    a.add_argument("--split", required=True, type=Path)
    a.add_argument("--area-epsilon", dest="area_epsilon", type=float)
    a.add_argument("--tolerance", type=float)
    a.set_defaults(func=_audit)

    k = sub.add_parser("export-kmz", help="write a KMZ visualization of an existing split")
    k.add_argument("--config", required=True, type=Path)
    k.add_argument("--split", required=True, type=Path)
    k.add_argument("--out", required=True, type=Path)
    k.add_argument("--tiles", action="store_true", help="include positive-tile footprints")
    k.set_defaults(func=_export_kmz)

    ls = sub.add_parser("list-positive-strategies", help="print registered positive strategies")
    ls.set_defaults(func=_list_strategies)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
